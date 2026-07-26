"""Validate SWE-rebench task metadata, images, tests, and parsers.

The inexpensive metadata mode scans task definitions without starting
containers. Execution mode performs two clean evaluations for each selected
task:

* base commit + held-out test patch: F2P must be observed failing and P2P pass;
* gold patch + held-out test patch: every F2P and P2P test must pass.

Results are written as JSON Lines so a long Slurm shard retains completed work
if it is preempted. Runtime failures are records, not process failures; use
``--fail-on-invalid`` when a nonzero final status is useful.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

import pyarrow.parquet as pq

import log_parsers
from dataset_store import TaskDataset
from sandbox_backends import EnrootSandbox, LocalRunResult, local_image_reference
from scoring import normalize_test_name
from task_commands import build_test_command_script, normalize_test_commands


EXECUTION_COLUMNS = [
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "image_name",
    "language",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "install_config",
]

METADATA_COLUMNS = EXECUTION_COLUMNS
_PASSED = log_parsers.TestStatus.PASSED.value
_EXPLICIT_FAILURES = {
    log_parsers.TestStatus.FAILED.value,
    log_parsers.TestStatus.ERROR.value,
}


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _bounded_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    omitted = len(output) - max_chars
    return (
        output[:half]
        + f"\n... [preflight truncated {omitted} characters] ...\n"
        + output[-(max_chars - half) :]
    )


def _parse_index_spec(value: str) -> list[int]:
    """Parse comma-separated indices and inclusive ranges."""
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            index = int(item)
            if index < 0:
                raise ValueError("indices must be nonnegative")
            indices.add(index)
            continue
        start_text, end_text = item.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < start:
            raise ValueError(f"invalid index range: {item!r}")
        indices.update(range(start, end + 1))
    return sorted(indices)


def select_indices(
    num_tasks: int,
    *,
    index_spec: str | None,
    shard_id: int,
    num_shards: int,
    max_tasks: int | None,
) -> list[int]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"shard_id must be in 0..{num_shards - 1}; got {shard_id}"
        )
    candidates: Iterable[int]
    if index_spec:
        candidates = _parse_index_spec(index_spec)
    else:
        candidates = range(num_tasks)

    selected = [
        index
        for index in candidates
        if index < num_tasks and index % num_shards == shard_id
    ]
    if index_spec:
        out_of_range = [
            index for index in _parse_index_spec(index_spec) if index >= num_tasks
        ]
        if out_of_range:
            raise IndexError(
                f"task indices outside 0..{num_tasks - 1}: "
                + ", ".join(map(str, out_of_range[:10]))
            )
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        selected = selected[:max_tasks]
    return selected


def stratified_sample(
    dataset: TaskDataset,
    indices: Sequence[int],
    *,
    per_language: int | None,
    seed: int,
    cached_image_dir: Path | None,
    require_valid_metadata: bool = False,
) -> list[int]:
    """Filter cached images and deterministically sample each language."""
    if per_language is not None and per_language < 1:
        raise ValueError("sample_per_language must be at least 1")
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, row in _iter_selected_rows(
        dataset,
        indices,
        columns=(
            METADATA_COLUMNS
            if require_valid_metadata
            else ["language", "image_name"]
        ),
    ):
        if require_valid_metadata and not validate_metadata(
            index,
            dataset.raw_index(index),
            row,
        )["valid"]:
            continue
        if cached_image_dir is not None:
            image = row.get("image_name")
            if not isinstance(image, str):
                continue
            digest = hashlib.sha256(
                local_image_reference(image).encode()
            ).hexdigest()
            image_path = cached_image_dir / f"{digest}.sqsh"
            if not image_path.exists() or image_path.stat().st_size == 0:
                continue
        candidates[str(row.get("language") or "unknown")].append(index)

    rng = random.Random(seed)
    selected: list[int] = []
    for language in sorted(candidates):
        language_indices = sorted(candidates[language])
        if per_language is not None and len(language_indices) > per_language:
            language_indices = rng.sample(language_indices, per_language)
        selected.extend(language_indices)
    return sorted(selected)


def _iter_selected_rows(
    dataset: TaskDataset,
    indices: Sequence[int],
    *,
    columns: Sequence[str],
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read selected rows once per parquet row group, not once per task."""
    public_by_raw = {
        dataset.raw_index(public_index): public_index for public_index in indices
    }
    wanted = set(public_by_raw)
    for shard in dataset.shards:
        parquet_file = pq.ParquetFile(shard.path)
        row_group_offset = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group_index).num_rows
            raw_start = shard.offset + row_group_offset
            raw_end = raw_start + row_count
            relevant = sorted(
                raw_index
                for raw_index in wanted
                if raw_start <= raw_index < raw_end
            )
            if relevant:
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=list(columns),
                )
                for raw_index in relevant:
                    row_offset = raw_index - raw_start
                    row = {
                        column: table.column(column)[row_offset].as_py()
                        for column in columns
                    }
                    yield (
                        public_by_raw[raw_index],
                        TaskDataset._normalize_row(row),
                    )
            row_group_offset += row_count


def validate_metadata(
    index: int,
    raw_index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    install_config = row.get("install_config")
    if not isinstance(install_config, dict):
        errors.append("install_config is not an object")
        install_config = {}

    try:
        test_commands = normalize_test_commands(install_config.get("test_cmd"))
    except ValueError as error:
        errors.append(str(error))
        test_commands = []

    parser_name = install_config.get("log_parser")
    if not isinstance(parser_name, str) or not callable(
        getattr(log_parsers, parser_name, None)
    ):
        errors.append(f"unknown log parser: {parser_name!r}")

    raw_fail_to_pass = [
        str(test_id) for test_id in row.get("FAIL_TO_PASS", [])
    ]
    raw_pass_to_pass = [
        str(test_id) for test_id in row.get("PASS_TO_PASS", [])
    ]
    fail_to_pass = [
        normalize_test_name(test_id) for test_id in raw_fail_to_pass
    ]
    pass_to_pass = [
        normalize_test_name(test_id) for test_id in raw_pass_to_pass
    ]
    if not fail_to_pass:
        errors.append("FAIL_TO_PASS is empty")
    raw_overlap = sorted(set(raw_fail_to_pass) & set(raw_pass_to_pass))
    if raw_overlap:
        errors.append(
            f"{len(raw_overlap)} test(s) occur verbatim in both "
            "FAIL_TO_PASS and PASS_TO_PASS"
        )
    # PHPUnit identifiers include durations. The same logical test can appear
    # in both groups with different timing suffixes when upstream collected
    # repeated suite output. These aliases remain scoreable: at submission
    # time both groups require PASSED, while the base preflight treats either
    # observed status as evidence for the shared logical identifier.
    normalized_cross_group_aliases = sorted(
        (set(fail_to_pass) & set(pass_to_pass))
        - {normalize_test_name(test_id) for test_id in raw_overlap}
    )
    for field in ("instance_id", "base_commit", "image_name", "patch", "test_patch"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{field} is empty or not a string")

    return {
        "schema_version": 1,
        "mode": "metadata",
        "index": index,
        "raw_index": raw_index,
        "instance_id": row.get("instance_id"),
        "image_name": row.get("image_name"),
        "language": row.get("language"),
        "parser": parser_name,
        "test_command_count": len(test_commands),
        "fail_to_pass_count": len(fail_to_pass),
        "pass_to_pass_count": len(pass_to_pass),
        "normalized_cross_group_alias_count": len(
            normalized_cross_group_aliases
        ),
        "normalized_cross_group_aliases": normalized_cross_group_aliases[:20],
        "valid": not errors,
        "errors": errors,
    }


async def _run_checked(
    sandbox: EnrootSandbox,
    command: str,
    *,
    timeout: float | None = None,
    stdin_data: bytes | None = None,
) -> LocalRunResult:
    result = await sandbox.run(
        command,
        timeout=timeout,
        stdin_data=stdin_data,
    )
    if result.return_code != 0:
        raise RuntimeError(
            f"command exited {result.return_code}: "
            f"{_bounded_output(result.output, 4000)}"
        )
    return result


async def _apply_patch(
    sandbox: EnrootSandbox,
    workdir: str,
    patch: str,
    label: str,
) -> None:
    patch_bytes = patch.encode("utf-8")
    command_prefix = f"cd {_shell_quote(workdir)} && "
    first = await sandbox.run(
        command_prefix + "git apply -",
        stdin_data=patch_bytes,
    )
    if first.return_code == 0:
        return
    second = await sandbox.run(
        command_prefix + "git apply --3way -",
        stdin_data=patch_bytes,
    )
    if second.return_code != 0:
        raise RuntimeError(
            f"failed to apply {label} patch with git apply "
            f"(exit {first.return_code}) and --3way "
            f"(exit {second.return_code}):\n"
            + _bounded_output(first.output + "\n" + second.output, 8000)
        )


def assess_phase(
    phase: str,
    test_results: dict[str, str],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    test_exit_code: int,
) -> dict[str, Any]:
    expected = {
        normalize_test_name(test_id): test_results.get(
            normalize_test_name(test_id), "NOT_FOUND"
        )
        for test_id in [*fail_to_pass, *pass_to_pass]
    }
    normalized_f2p = [normalize_test_name(test_id) for test_id in fail_to_pass]
    normalized_p2p = [normalize_test_name(test_id) for test_id in pass_to_pass]
    shared_expected = set(normalized_f2p) & set(normalized_p2p)
    missing = [test_id for test_id, status in expected.items() if status == "NOT_FOUND"]
    # Match the reference verifier: 0 and 1 are ordinary test-suite outcomes.
    # Exit codes above 1 indicate invocation/infrastructure failure. Expected
    # parsed IDs, not unrelated suite failures, determine task validity.
    ordinary_test_exit = 0 <= test_exit_code <= 1

    if phase == "base":
        wrong_f2p = [
            test_id
            for test_id in normalized_f2p
            if test_id not in shared_expected
            if expected[test_id] not in _EXPLICIT_FAILURES
        ]
        wrong_p2p = [
            test_id
            for test_id in normalized_p2p
            if test_id not in shared_expected
            if expected[test_id] != _PASSED
        ]
        wrong_shared = [
            test_id
            for test_id in sorted(shared_expected)
            if expected[test_id] not in {*_EXPLICIT_FAILURES, _PASSED}
        ]
        valid = (
            ordinary_test_exit
            and not missing
            and not wrong_f2p
            and not wrong_p2p
            and not wrong_shared
        )
    elif phase == "gold":
        wrong_f2p = [
            test_id
            for test_id in normalized_f2p
            if expected[test_id] != _PASSED
        ]
        wrong_p2p = [
            test_id
            for test_id in normalized_p2p
            if expected[test_id] != _PASSED
        ]
        wrong_shared = []
        valid = (
            ordinary_test_exit
            and not missing
            and not wrong_f2p
            and not wrong_p2p
        )
    else:
        raise ValueError(f"unknown phase: {phase!r}")

    return {
        "valid": valid,
        "test_exit_code": test_exit_code,
        "parsed_test_count": len(test_results),
        "missing_count": len(missing),
        "wrong_fail_to_pass_count": len(wrong_f2p),
        "wrong_pass_to_pass_count": len(wrong_p2p),
        "shared_expected_count": len(shared_expected),
        "wrong_shared_count": len(wrong_shared),
        "missing": missing,
        "wrong_fail_to_pass": wrong_f2p,
        "wrong_pass_to_pass": wrong_p2p,
        "shared_expected": sorted(shared_expected),
        "wrong_shared": wrong_shared,
        "expected_results": expected,
    }


async def _run_phase(
    row: dict[str, Any],
    phase: str,
    *,
    test_timeout: float,
    diagnostic_chars: int,
) -> tuple[dict[str, Any], Path]:
    sandbox = EnrootSandbox(row["image_name"])
    image_path = sandbox._cached_image_path()
    started = time.monotonic()
    try:
        await sandbox.start()
        pwd = await _run_checked(sandbox, "pwd")
        workdir = pwd.output.strip()
        if not workdir:
            raise RuntimeError("container returned an empty workdir")
        await _run_checked(
            sandbox,
            f"cd {_shell_quote(workdir)} && "
            "git config --global --add safe.directory '*' && "
            "git config user.email 'preflight@openreward.dev' && "
            "git config user.name 'OpenReward Preflight' && "
            f"git checkout --force {_shell_quote(row['base_commit'])}",
        )
        if phase == "gold":
            await _apply_patch(
                sandbox,
                workdir,
                row["patch"],
                "gold",
            )
        await _apply_patch(
            sandbox,
            workdir,
            row["test_patch"],
            "held-out test",
        )

        install_config = row["install_config"]
        test_script = build_test_command_script(install_config["test_cmd"])
        test_result = await sandbox.run(
            f"cd {_shell_quote(workdir)} && {test_script}",
            timeout=test_timeout,
        )
        parser_name = install_config["log_parser"]
        parser = getattr(log_parsers, parser_name)
        parsed = {
            normalize_test_name(test_id): status
            for test_id, status in parser(test_result.output).items()
        }
        assessment = assess_phase(
            phase,
            parsed,
            row["FAIL_TO_PASS"],
            row["PASS_TO_PASS"],
            test_result.return_code,
        )
        assessment["duration_seconds"] = time.monotonic() - started
        if not assessment["valid"]:
            assessment["test_output_excerpt"] = _bounded_output(
                test_result.output,
                diagnostic_chars,
            )
        return assessment, image_path
    finally:
        await sandbox.stop()


def _remove_cached_image(image_path: Path) -> None:
    """Remove one image under the same lock used by lazy import."""
    lock_path = image_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        image_path.unlink(missing_ok=True)


async def execute_task(
    dataset: TaskDataset,
    index: int,
    *,
    test_timeout: float,
    diagnostic_chars: int,
    delete_image: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    raw_index = dataset.raw_index(index)
    row = await asyncio.to_thread(
        dataset.get_row,
        index,
        columns=EXECUTION_COLUMNS,
    )
    metadata = validate_metadata(index, raw_index, row)
    result: dict[str, Any] = {
        **metadata,
        "mode": "execute",
        "phases": {},
    }
    image_path: Path | None = EnrootSandbox(
        row["image_name"]
    )._cached_image_path()
    if not metadata["valid"]:
        result["duration_seconds"] = time.monotonic() - started
        return result

    for phase in ("base", "gold"):
        try:
            phase_result, image_path = await _run_phase(
                row,
                phase,
                test_timeout=test_timeout,
                diagnostic_chars=diagnostic_chars,
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
                raise
            phase_result = {
                "valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": _bounded_output(traceback.format_exc(), 8000),
            }
        result["phases"][phase] = phase_result

    result["valid"] = all(
        phase_result.get("valid", False)
        for phase_result in result["phases"].values()
    )
    result["duration_seconds"] = time.monotonic() - started
    if delete_image and image_path is not None:
        try:
            await asyncio.to_thread(_remove_cached_image, image_path)
            result["cached_image_deleted"] = True
        except Exception as error:
            result["cached_image_deleted"] = False
            result["cache_cleanup_error"] = str(error)
            result["valid"] = False
    return result


def _existing_results(path: Path) -> dict[int, bool]:
    """Return the latest validity value for each completed JSONL index."""
    completed: dict[int, bool] = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            completed[int(record["index"])] = bool(record.get("valid", False))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


class _ResultWriter:
    def __init__(self, output: str, resume: bool) -> None:
        self.path = None if output == "-" else Path(output)
        if self.path is None:
            self.stream: TextIO = sys.stdout
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a" if resume else "w")

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self.path is not None:
            self.stream.close()


async def _execute_selected(
    dataset: TaskDataset,
    indices: Sequence[int],
    writer: _ResultWriter,
    *,
    workers: int,
    test_timeout: float,
    diagnostic_chars: int,
    delete_images: bool,
) -> tuple[int, int]:
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    for index in indices:
        queue.put_nowait(index)
    for _ in range(workers):
        queue.put_nowait(None)

    counts = {"completed": 0, "invalid": 0}
    write_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            index = await queue.get()
            try:
                if index is None:
                    return
                try:
                    record = await execute_task(
                        dataset,
                        index,
                        test_timeout=test_timeout,
                        diagnostic_chars=diagnostic_chars,
                        delete_image=delete_images,
                    )
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
                        raise
                    record = {
                        "schema_version": 1,
                        "mode": "execute",
                        "index": index,
                        "raw_index": dataset.raw_index(index),
                        "valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": _bounded_output(
                            traceback.format_exc(),
                            8000,
                        ),
                    }
                async with write_lock:
                    writer.write(record)
                    counts["completed"] += 1
                    counts["invalid"] += not record.get("valid", False)
                    print(
                        f"[{counts['completed']}/{len(indices)}] "
                        f"index={index} valid={record.get('valid', False)}",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    try:
        await queue.join()
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return counts["completed"], counts["invalid"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "/orwd_data")),
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Defaults to DATA_DIR/task_index.json",
    )
    parser.add_argument(
        "--mode",
        choices=("metadata", "execute"),
        default="metadata",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="JSONL output path, or - for stdout",
    )
    parser.add_argument(
        "--indices",
        default=os.getenv("SWE_PREFLIGHT_INDICES"),
        help="Comma-separated public indices and inclusive ranges",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--sample-per-language",
        type=int,
        default=None,
        help="Deterministically retain at most this many tasks per language",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--cached-image-dir",
        type=Path,
        default=None,
        help="Restrict selection to images already present in this SQSH cache",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--test-timeout",
        type=float,
        default=float(os.getenv("SWE_TEST_TIMEOUT_SECONDS", "1200")),
    )
    parser.add_argument("--diagnostic-chars", type=int, default=12000)
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="Delete imported SQSH files after each task; use only with a dedicated cache",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    dataset = TaskDataset(args.data_dir, index_path=args.index_path)
    # Sample globally before assigning shards. Sampling each shard separately
    # would multiply the requested per-language bound by num_shards.
    indices = select_indices(
        dataset.num_tasks,
        index_spec=args.indices,
        shard_id=0,
        num_shards=1,
        max_tasks=None,
    )
    if args.sample_per_language is not None or args.cached_image_dir is not None:
        indices = stratified_sample(
            dataset,
            indices,
            per_language=args.sample_per_language,
            seed=args.sample_seed,
            cached_image_dir=args.cached_image_dir,
            require_valid_metadata=args.mode == "execute",
        )
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(
            f"--shard-id must be in 0..{args.num_shards - 1}"
        )
    indices = [
        index for index in indices if index % args.num_shards == args.shard_id
    ]
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise SystemExit("--max-tasks must be at least 1")
        indices = indices[: args.max_tasks]
    selected_count = len(indices)
    output_path = None if args.output == "-" else Path(args.output)
    completed = 0
    invalid = 0
    if args.resume and output_path is not None:
        selected_set = set(indices)
        existing = {
            index: valid
            for index, valid in _existing_results(output_path).items()
            if index in selected_set
        }
        completed = len(existing)
        invalid = sum(not valid for valid in existing.values())
        indices = [index for index in indices if index not in existing]

    writer = _ResultWriter(args.output, args.resume)
    started = time.monotonic()
    try:
        if args.mode == "metadata":
            for index, row in _iter_selected_rows(
                dataset,
                indices,
                columns=METADATA_COLUMNS,
            ):
                record = validate_metadata(
                    index,
                    dataset.raw_index(index),
                    row,
                )
                writer.write(record)
                completed += 1
                invalid += not record["valid"]
        else:
            new_completed, new_invalid = asyncio.run(
                _execute_selected(
                    dataset,
                    indices,
                    writer,
                    workers=args.workers,
                    test_timeout=args.test_timeout,
                    diagnostic_chars=args.diagnostic_chars,
                    delete_images=args.delete_images,
                )
            )
            completed += new_completed
            invalid += new_invalid
    finally:
        writer.close()

    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "selected": selected_count,
        "completed": completed,
        "invalid": invalid,
        "duration_seconds": time.monotonic() - started,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if output_path is not None:
        summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.fail_on_invalid and invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
