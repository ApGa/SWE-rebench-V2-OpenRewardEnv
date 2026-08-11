"""OpenReward environment for SWE-rebench-V2."""
import asyncio
import base64
import functools
import os
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from openreward import AsyncOpenReward, SandboxSettings
from openreward.api.sandboxes.types import MachineSize
from openreward.environments import Environment, Server, tool
from openreward.environments.types import (
    Blocks,
    JSONObject,
    ListToolsOutput,
    TextBlock,
    ToolOutput,
    ToolSpec,
)
from pydantic import BaseModel, Field

from dataset_store import TaskDataset
from scoring import normalize_test_name, score_test_results
from sandbox_backends import create_local_sandbox
from task_commands import build_test_command_script

# ---------------------------------------------------------------------------
# Dataset loading — lazy row-group reads from one file or multiple shards
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "/orwd_data"))
TASK_INDEX = Path(os.getenv("TASK_INDEX", DATA_DIR / "task_index.json"))
_TASK_DATASET = TaskDataset(DATA_DIR, index_path=TASK_INDEX)


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

class InstallConfig(BaseModel):
    test_cmd: str | list[str]
    log_parser: str
    install: str | list[str] = ""
    base_image_name: str = ""


class TaskSpec(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    problem_statement: str
    image_name: str
    language: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    install_config: InstallConfig


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------

ENVIRONMENT_NAME = "nebius/SWE-rebench-V2"
TOOL_ROUTING_SCHEMA_KEY = "x-openhands-tool-routing"


def _with_direct_tool_routing(
    spec: ToolSpec,
    capabilities: tuple[str, ...],
) -> ToolSpec:
    """Attach task-runtime routing as an ignorable JSON-Schema extension."""

    schema = dict(spec.input_schema or {"type": "object", "properties": {}})
    schema[TOOL_ROUTING_SCHEMA_KEY] = {
        "version": 1,
        "execution_domain": "task",
        "capabilities": list(capabilities),
        "invocation": {"kind": "direct"},
    }
    return spec.model_copy(update={"input_schema": schema})


class BashInput(BaseModel):
    """Input for bash command execution."""
    command: str = Field(..., description="Bash command to run in container")
    description: str = Field(..., description="Why I'm running this command")


class StrReplaceInput(BaseModel):
    """Input for string replacement in files."""
    path: str = Field(..., description="Path to the file to edit")
    old_str: str = Field(..., description="String to replace (must be unique in file)")
    new_str: str = Field(default="", description="String to replace with (empty to delete)")
    description: str = Field(..., description="Why I'm making this edit")


class ViewInput(BaseModel):
    """Input for viewing files and directories."""
    path: str = Field(..., description="Absolute path to file or directory")
    view_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional line range for text files. Format: [start_line, end_line] where lines are indexed starting at 1. Use [start_line, -1] to view from start_line to end."
    )
    description: str = Field(..., description="Why I need to view this")


class CreateFileInput(BaseModel):
    """Input for creating new files."""
    description: str = Field(..., description="Why I'm creating this file")
    path: str = Field(..., description="Path to the file to create")
    file_text: str = Field(..., description="Content to write to the file")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_output(
    text: str,
    finished: bool = False,
    *,
    metadata: JSONObject | None = None,
    reward: float | None = None,
) -> ToolOutput:
    return ToolOutput(
        blocks=[TextBlock(text=text)],
        finished=finished,
        metadata=metadata,
        reward=reward,
    )


def _command_output(
    text: str,
    exit_code: int,
    *,
    finished: bool = False,
    reward: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ToolOutput:
    metadata: dict[str, Any] = {
        "ok": exit_code == 0,
        "exit_code": exit_code,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return _text_output(
        f"{text}\nExit code: {exit_code}",
        finished,
        metadata=metadata,
        reward=reward,
    )


def _tool_error(
    tool_name: str,
    error: BaseException | str,
    *,
    finished: bool = False,
    reward: float | None = None,
    invalid: bool = False,
) -> ToolOutput:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error) or repr(error)
    else:
        error_type = "ToolError"
        message = error
    metadata: dict[str, Any] = {
        "ok": False,
        "tool": tool_name,
        "error_type": error_type,
        "error": message,
        "exit_code": 1,
    }
    if invalid:
        metadata.update(
            {
                "invalid": True,
                "failure_class": "infrastructure",
            }
        )
    return _text_output(
        (
            f"Tool error: {tool_name}\n"
            f"Error type: {error_type}\n"
            f"Message: {message}\n"
            "Exit code: 1"
        ),
        finished,
        metadata=metadata,
        reward=reward,
    )


def _result_values(result: Any) -> tuple[str, int]:
    """Normalize hosted and local sandbox command results."""
    output = getattr(result, "output", None)
    if output is not None:
        return str(output), int(
            getattr(result, "return_code", getattr(result, "exit_code", 0))
        )
    output, return_code = result
    return str(output), int(return_code)


def _bounded_text(output: str, max_chars: int, *, label: str) -> str:
    if max_chars < 1:
        return ""
    if len(output) <= max_chars:
        return output
    omitted = len(output) - max_chars
    start_chars = max_chars // 2
    end_chars = max_chars - start_chars
    return (
        output[:start_chars]
        + f"\n\n... [{label}: {omitted} characters omitted] ...\n\n"
        + output[-end_chars:]
    )


def _bounded_output(output: str) -> str:
    return _bounded_text(
        output,
        int(os.getenv("SWE_TOOL_OUTPUT_MAX_CHARS", "50000")),
        label="tool output truncated",
    )


def _status_detail(
    test_ids: list[str],
    test_results: dict[str, str],
    *,
    only_nonpassing: bool,
) -> tuple[list[str], int]:
    """Return a bounded status listing and its unabridged item count."""
    entries = [
        (
            test_id,
            test_results.get(normalize_test_name(test_id), "NOT_FOUND"),
        )
        for test_id in test_ids
    ]
    if only_nonpassing:
        entries = [
            (test_id, status)
            for test_id, status in entries
            if status != "PASSED"
        ]
    total = len(entries)
    limit = max(
        0,
        int(os.getenv("SWE_GRADER_EXPECTED_DETAIL_LIMIT", "200")),
    )
    lines = [f"  {test_id}: {status}" for test_id, status in entries[:limit]]
    if total > limit:
        lines.append(f"  ... {total - limit} additional result(s) omitted")
    return lines, total


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _get_log_parser(parser_name: str):
    """Import and return the log parser function by name."""
    import log_parsers
    fn = getattr(log_parsers, parser_name, None)
    if fn is None:
        raise ValueError(f"Unknown log parser: {parser_name}")
    return fn


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SWERebenchV2(Environment):
    """OpenReward environment for SWE-rebench-V2 tasks."""

    _TOOL_CAPABILITIES = {
        "bash": (
            "filesystem.read",
            "filesystem.write",
            "system.execute",
            "python.execute",
        ),
        "view": ("filesystem.read",),
        "str_replace": ("filesystem.read", "filesystem.write"),
        "create_file": ("filesystem.write",),
        "submit_answer": ("task.submit",),
    }

    @classmethod
    @functools.cache
    def list_tools(cls) -> ListToolsOutput:
        tools = super().list_tools()
        return ListToolsOutput(
            tools=[
                _with_direct_tool_routing(
                    spec,
                    cls._TOOL_CAPABILITIES[spec.name],
                )
                if spec.name in cls._TOOL_CAPABILITIES
                else spec
                for spec in tools.tools
            ]
        )

    def __init__(
        self,
        task_spec: JSONObject,
        secrets: dict[str, str] | None = None,
    ) -> None:
        super().__init__(task_spec)
        self.parsed = TaskSpec.model_validate(task_spec)

        secrets = secrets or {}
        self.workdir: str | None = None  # resolved in setup() from container WORKDIR
        runtime = os.getenv("SWE_SANDBOX_RUNTIME", "hosted").strip().lower()
        if runtime == "hosted":
            self.or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
            self.sandbox_settings = SandboxSettings(
                environment=ENVIRONMENT_NAME,
                image=self.parsed.image_name,
                machine_size=cast(
                    MachineSize,
                    os.getenv("SWE_SANDBOX_MACHINE_SIZE", "2:4"),
                ),
            )
            self.sandbox: Any = self.or_client.sandbox(self.sandbox_settings)
        else:
            self.or_client = None
            self.sandbox_settings = None
            self.sandbox = create_local_sandbox(runtime, self.parsed.image_name)

    # ----- splits / tasks (class methods) -----

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        raise NotImplementedError(
            "Dataset has 32K+ tasks — use num_tasks/get_task instead"
        )

    @classmethod
    async def num_tasks(cls, split: str) -> int:
        if split != "train":
            raise ValueError(f"Unknown split: {split!r}")
        return _TASK_DATASET.num_tasks

    @classmethod
    async def get_task(cls, split: str, index: int) -> JSONObject:
        if split != "train":
            raise ValueError(f"Unknown split: {split!r}")
        row = await asyncio.to_thread(_TASK_DATASET.get_task, index)
        row["FAIL_TO_PASS"] = [
            normalize_test_name(t) for t in row["FAIL_TO_PASS"]
        ]
        row["PASS_TO_PASS"] = [
            normalize_test_name(t) for t in row["PASS_TO_PASS"]
        ]
        return row

    # ----- lifecycle -----

    async def setup(self):
        try:
            await self.sandbox.start()
            # SWE-rebench V2 images use /{project_name} as WORKDIR (not /testbed).
            # Query the container's actual WORKDIR so we don't have to guess.
            res = await self.sandbox.run("pwd")
            output, exit_code = _result_values(res)
            if exit_code != 0 or not output.strip():
                raise RuntimeError(
                    f"Could not determine sandbox workdir: {output.strip()}"
                )
            self.workdir = output.strip()

            setup_commands = [
                (
                    "configure git",
                    f"cd {_shell_quote(self.workdir)} && "
                    "git config --global --add safe.directory '*' && "
                    "git config user.email 'agent@openreward.dev' && "
                    "git config user.name 'Agent'",
                ),
                (
                    "checkout base commit",
                    f"cd {_shell_quote(self.workdir)} && "
                    f"git checkout {_shell_quote(self.parsed.base_commit)}",
                ),
                (
                    "remove hidden git history",
                    f"cd {_shell_quote(self.workdir)} && "
                    "git reflog expire --expire=now --all && "
                    "git gc --prune=now --quiet",
                ),
            ]
            for label, command in setup_commands:
                result = await self.sandbox.run(command)
                output, exit_code = _result_values(result)
                if exit_code != 0:
                    raise RuntimeError(
                        f"Failed to {label} (exit {exit_code}): "
                        f"{output.strip()}"
                    )
        except Exception:
            await self.sandbox.stop()
            raise

    async def teardown(self):
        await self.sandbox.stop()

    def get_prompt(self) -> Blocks:
        text = (
            f"You are a software engineer working on the repository **{self.parsed.repo}** "
            f"(language: {self.parsed.language}).\n\n"
            f"## Problem Statement\n\n{self.parsed.problem_statement}\n\n"
            f"## Instructions\n\n"
            f"The repository is cloned at `{self.workdir}` and checked out to the commit "
            f"before the fix. Your task is to modify the code so that the failing tests pass.\n\n"
            f"Use the available tools to explore the codebase, understand the problem, "
            f"make edits, and then call `submit_answer` when you are done.\n\n"
            f"Do NOT modify or create tests — only fix the source code."
        )
        return [TextBlock(text=text)]

    async def _stage_text_file(self, content: str) -> str:
        """Write content to a private sandbox file without placing it in argv.

        Local Docker and Enroot backends have a real stdin pipe. The hosted
        OpenReward API does not currently expose stdin, so use bounded chunks
        whose individual command arguments remain safely below ARG_MAX.
        """
        sandbox_path = f"/tmp/.openreward-upload-{uuid4().hex}"
        encoded = content.encode("utf-8")

        if self.or_client is None:
            result = await self.sandbox.run(
                f"umask 077; cat > {_shell_quote(sandbox_path)}",
                stdin_data=encoded,
            )
            output, exit_code = _result_values(result)
            if exit_code != 0:
                detail = output.strip() or "(no output)"
                raise RuntimeError(
                    f"Could not stage file in sandbox (exit {exit_code}): "
                    f"{detail}"
                )
            return sandbox_path

        # OpenReward 0.1.137's upload helper embeds the complete file in one
        # command and therefore has the same ARG_MAX failure mode. Hosted
        # sandboxes do not expose stdin yet, so append bounded base64 chunks.
        # 32 KiB of raw input yields a command under 44 KiB.
        chunk_size = 32 * 1024
        try:
            initialize = await self.sandbox.run(
                f"umask 077; : > {_shell_quote(sandbox_path)}"
            )
            initialize_output, initialize_code = _result_values(initialize)
            if initialize_code != 0:
                raise RuntimeError(
                    "Could not initialize hosted sandbox staging file "
                    f"(exit {initialize_code}): "
                    f"{initialize_output.strip() or '(no output)'}"
                )

            for offset in range(0, len(encoded), chunk_size):
                chunk = base64.b64encode(
                    encoded[offset : offset + chunk_size]
                ).decode("ascii")
                append = await self.sandbox.run(
                    f"printf '%s' {_shell_quote(chunk)} | base64 -d >> "
                    f"{_shell_quote(sandbox_path)}"
                )
                append_output, append_code = _result_values(append)
                if append_code != 0:
                    raise RuntimeError(
                        "Could not append hosted sandbox staging chunk "
                        f"(exit {append_code}): "
                        f"{append_output.strip() or '(no output)'}"
                    )

            verify = await self.sandbox.run(
                f"test \"$(wc -c < {_shell_quote(sandbox_path)})\" "
                f"-eq {len(encoded)}"
            )
            verify_output, verify_code = _result_values(verify)
            if verify_code != 0:
                raise RuntimeError(
                    "Hosted sandbox staging file failed size verification "
                    f"(expected {len(encoded)} bytes): "
                    f"{verify_output.strip() or '(no output)'}"
                )
        except Exception:
            try:
                await self.sandbox.run(
                    f"rm -f -- {_shell_quote(sandbox_path)}"
                )
            except Exception:
                pass
            raise
        return sandbox_path

    async def _install_staged_file(
        self,
        staged_path: str,
        target_path: str,
        *,
        create_parent: bool,
    ) -> tuple[str, int]:
        """Atomically install a staged file while preserving target mode."""
        parent_setup = (
            'mkdir -p -- "$parent"\n'
            if create_parent
            else '[ -d "$parent" ] || { echo "Parent directory does not exist: '
            '$parent"; exit 1; }\n'
        )
        command = (
            "set -eu\n"
            f"source_file={_shell_quote(staged_path)}\n"
            f"target={_shell_quote(target_path)}\n"
            'parent=$(dirname -- "$target")\n'
            f"{parent_setup}"
            'if [ -L "$target" ]; then\n'
            '  cat -- "$source_file" > "$target"\n'
            '  rm -f -- "$source_file"\n'
            "  exit 0\n"
            "fi\n"
            'temporary=$(mktemp "$parent/.openreward-write.XXXXXX")\n'
            'cleanup() { rm -f -- "$temporary" "$source_file"; }\n'
            "trap cleanup EXIT HUP INT TERM\n"
            'cat -- "$source_file" > "$temporary"\n'
            'if [ -e "$target" ]; then\n'
            '  mode=$(stat -c "%a" -- "$target")\n'
            '  chmod "$mode" "$temporary"\n'
            "else\n"
            '  chmod 644 "$temporary"\n'
            "fi\n"
            'mv -f -- "$temporary" "$target"\n'
            'rm -f -- "$source_file"\n'
            "trap - EXIT HUP INT TERM\n"
        )
        result = await self.sandbox.run(command)
        return _result_values(result)

    # ----- tools -----

    @tool
    async def bash(self, input: BashInput) -> ToolOutput:
        """Run a bash command in the container."""
        try:
            assert self.workdir is not None, "setup() must run before tools"
            cmd = f"cd {_shell_quote(self.workdir)} && {input.command}"
            result = await self.sandbox.run(cmd)
            output, exit_code = _result_values(result)
            output = _bounded_output(output)
            s = output if output else "(no output)"
            return _command_output(s, exit_code)
        except Exception as error:
            return _tool_error("bash", error)

    @tool
    async def str_replace(self, input: StrReplaceInput) -> ToolOutput:
        """Replace a unique string in a file with another string."""
        try:
            if self.or_client is None:
                res = await self.sandbox.run(
                    f"cat -- {_shell_quote(input.path)}"
                )
                content, exit_code = _result_values(res)
                error_output = content
            else:
                # Hosted sandbox output defaults to 50 KB, which silently
                # truncates source files before replacement. Base64 also
                # preserves trailing newlines that the hosted SDK strips from
                # ordinary command output.
                res = await self.sandbox.run(
                    f"base64 {_shell_quote(input.path)}",
                    max_bytes=None,
                    sanitise=False,
                )
                encoded_content, exit_code = _result_values(res)
                error_output = encoded_content
                content = ""
                if exit_code == 0:
                    try:
                        content = base64.b64decode(
                            encoded_content.encode("ascii"),
                            validate=False,
                        ).decode("utf-8")
                    except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as error:
                        raise RuntimeError(
                            f"Could not decode source file {input.path}: {error}"
                        ) from error
            if exit_code != 0:
                s = error_output if error_output else "(no output)"
                return _command_output(
                    s,
                    exit_code,
                    extra_metadata={"tool": "str_replace"},
                )

            count = content.count(input.old_str)
            if count == 0:
                return _tool_error(
                    "str_replace",
                    f"The string to replace was not found in {input.path}",
                )
            if count > 1:
                return _tool_error(
                    "str_replace",
                    (
                        f"The string to replace appears {count} times in "
                        f"{input.path}. It must be unique."
                    ),
                )

            new_content = content.replace(input.old_str, input.new_str, 1)
            staged_path = await self._stage_text_file(new_content)
            output, exit_code = await self._install_staged_file(
                staged_path,
                input.path,
                create_parent=False,
            )
            s = (
                output
                if output
                else f"Successfully replaced string in {input.path}"
            )
            return _command_output(
                s,
                exit_code,
                extra_metadata={"tool": "str_replace"},
            )
        except Exception as error:
            return _tool_error("str_replace", error)

    @tool
    async def view(self, input: ViewInput) -> ToolOutput:
        """View file contents or directory listings."""
        try:
            res = await self.sandbox.run(
                f"test -d {_shell_quote(input.path)} && "
                "echo 'dir' || echo 'file'"
            )
            output, probe_exit_code = _result_values(res)
            if probe_exit_code != 0:
                return _command_output(
                    output or "(no output)",
                    probe_exit_code,
                    extra_metadata={"tool": "view"},
                )
            is_dir = output.strip() == "dir"

            if is_dir:
                cmd = (
                    f"find {_shell_quote(input.path)} -maxdepth 2 "
                    "-not -path '*/\\.*' "
                    "-not -path '*/node_modules/*' | head -100"
                )
            else:
                if input.view_range:
                    start, end = input.view_range
                    if end == -1:
                        cmd = (
                            f"cat -n {_shell_quote(input.path)} "
                            f"| tail -n +{start}"
                        )
                    else:
                        cmd = (
                            f"cat -n {_shell_quote(input.path)} "
                            f"| sed -n '{start},{end}p'"
                        )
                else:
                    cmd = f"cat -n {_shell_quote(input.path)}"

            res = await self.sandbox.run(cmd)
            output, exit_code = _result_values(res)

            if len(output) > 16000:
                lines = output.split('\n')
                mid = len(lines) // 2
                keep_start = mid // 2
                keep_end = mid // 2
                output = (
                    '\n'.join(lines[:keep_start])
                    + (
                        f"\n\n... [truncated "
                        f"{len(lines) - keep_start - keep_end} lines] ...\n\n"
                    )
                    + '\n'.join(lines[-keep_end:])
                )

            s = output if output else "(no output)"
            return _command_output(
                s,
                exit_code,
                extra_metadata={"tool": "view"},
            )
        except Exception as error:
            return _tool_error("view", error)

    @tool
    async def create_file(self, input: CreateFileInput) -> ToolOutput:
        """Create a new file with the specified content."""
        try:
            staged_path = await self._stage_text_file(input.file_text)
            output, exit_code = await self._install_staged_file(
                staged_path,
                input.path,
                create_parent=True,
            )
            s = output if output else f"Successfully created {input.path}"
            return _command_output(
                s,
                exit_code,
                extra_metadata={"tool": "create_file"},
            )
        except Exception as error:
            return _tool_error("create_file", error)

    @tool
    async def submit_answer(self) -> ToolOutput:
        """Submit your solution. Applies the test patch, runs the test suite, and scores."""
        try:
            return await self._grade_submission()
        except Exception as error:
            return _tool_error(
                "submit_answer",
                error,
                finished=True,
                reward=0.0,
                invalid=True,
            )

    async def _capture_submission_diagnostics(
        self,
    ) -> tuple[str, dict[str, Any]]:
        """Capture agent changes before the hidden test patch mutates the tree."""
        assert self.workdir is not None, "setup() must run before tools"
        status_begin = "__OPENREWARD_STATUS_BEGIN__"
        status_end = "__OPENREWARD_STATUS_END__"
        count_prefix = "__OPENREWARD_CHANGED_ENTRY_COUNT__="
        hash_prefix = "__OPENREWARD_TRACKED_DIFF_HASH__="
        command = (
            f"cd {_shell_quote(self.workdir)}\n"
            f"printf '%s\\n' {status_begin}\n"
            "git status --porcelain=v1 --untracked-files=normal "
            "| sed -n '1,200p'\n"
            f"printf '%s\\n' {status_end}\n"
            "changed_count=$(git status --porcelain=v1 "
            "--untracked-files=normal | wc -l)\n"
            f"printf '{count_prefix}%s\\n' \"$changed_count\"\n"
            "diff_hash=$(git diff --binary --no-ext-diff HEAD -- "
            "| git hash-object --stdin)\n"
            f"printf '{hash_prefix}%s\\n' \"$diff_hash\"\n"
            "printf '%s\\n' '__OPENREWARD_DIFF_STAT_BEGIN__'\n"
            "git diff --stat --no-ext-diff HEAD -- | sed -n '1,200p'\n"
            "printf '%s\\n' '__OPENREWARD_DIFF_STAT_END__'\n"
        )
        try:
            result = await self.sandbox.run(command)
            output, exit_code = _result_values(result)
        except Exception as error:
            message = f"Could not capture repository diagnostics: {error}"
            return message, {
                "repository_diagnostics_ok": False,
                "repository_diagnostics_error": str(error),
            }

        bounded = _bounded_text(
            output.strip() or "(no repository diagnostic output)",
            int(os.getenv("SWE_GRADER_DIAGNOSTIC_MAX_CHARS", "12000")),
            label="repository diagnostics truncated",
        )
        metadata: dict[str, Any] = {
            "repository_diagnostics_ok": exit_code == 0,
            "repository_diagnostics_exit_code": exit_code,
        }
        lines = output.splitlines()
        try:
            start = lines.index(status_begin) + 1
            end = lines.index(status_end, start)
            metadata["changed_files_preview"] = lines[start:end]
        except ValueError:
            metadata["changed_files_preview"] = []
        for line in lines:
            if line.startswith(count_prefix):
                try:
                    metadata["changed_entry_count"] = int(
                        line.removeprefix(count_prefix).strip()
                    )
                except ValueError:
                    pass
            elif line.startswith(hash_prefix):
                metadata["tracked_diff_hash"] = line.removeprefix(
                    hash_prefix
                ).strip()
        return bounded, metadata

    async def _grade_submission(self) -> ToolOutput:
        assert self.workdir is not None, "setup() must run before tools"
        submission_diagnostics, diagnostic_metadata = (
            await self._capture_submission_diagnostics()
        )

        # 1. Apply the held-out patch. Local runtimes can stream it directly
        # over stdin, keeping the hidden tests out of both argv and /tmp. The
        # hosted API has no stdin channel, so it uses the bounded staging
        # transport and removes the private file in the same command.
        if self.or_client is None:
            patch_bytes = self.parsed.test_patch.encode("utf-8")
            apply_result = await self.sandbox.run(
                f"cd {_shell_quote(self.workdir)} && git apply -",
                stdin_data=patch_bytes,
            )
            apply_output, apply_code = _result_values(apply_result)
            if apply_code != 0:
                three_way_result = await self.sandbox.run(
                    f"cd {_shell_quote(self.workdir)} && git apply --3way -",
                    stdin_data=patch_bytes,
                )
                three_way_output, apply_code = _result_values(
                    three_way_result
                )
                apply_output = "\n".join(
                    output
                    for output in (apply_output, three_way_output)
                    if output
                )
        else:
            patch_path = await self._stage_text_file(self.parsed.test_patch)
            apply_result = await self.sandbox.run(
                "set -u\n"
                f"cd {_shell_quote(self.workdir)}\n"
                f"patch_file={_shell_quote(patch_path)}\n"
                'cleanup() { rm -f -- "$patch_file"; }\n'
                "trap cleanup EXIT HUP INT TERM\n"
                'if git apply "$patch_file"; then\n'
                "  :\n"
                "else\n"
                '  git apply --3way "$patch_file"\n'
                "fi\n"
            )
            apply_output, apply_code = _result_values(apply_result)
        if apply_code != 0:
            detail = apply_output if apply_output else "(no output)"
            return _command_output(
                (
                    "Pre-submission repository state:\n"
                    f"{submission_diagnostics}\n\n"
                    f"Failed to apply test patch:\n{detail}"
                ),
                apply_code,
                reward=0.0,
                finished=True,
                extra_metadata={
                    "tool": "submit_answer",
                    "stage": "apply_test_patch",
                    "test_patch_apply_exit_code": apply_code,
                    **diagnostic_metadata,
                },
            )

        # 2. Run test command
        test_script = build_test_command_script(
            self.parsed.install_config.test_cmd
        )
        res = await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && {test_script}",
            timeout=float(os.getenv("SWE_TEST_TIMEOUT_SECONDS", "600")),
        )
        test_output, test_code = _result_values(res)

        # 3. Parse test output
        parser_name = self.parsed.install_config.log_parser
        try:
            parser_fn = _get_log_parser(parser_name)
            test_results = {
                normalize_test_name(test_id): status
                for test_id, status in parser_fn(test_output).items()
            }
        except Exception as error:
            raw_output = _bounded_text(
                test_output,
                int(os.getenv("SWE_GRADER_RAW_OUTPUT_MAX_CHARS", "20000")),
                label="raw test output truncated",
            )
            result = _tool_error(
                "submit_answer",
                (
                    "Pre-submission repository state:\n"
                    f"{submission_diagnostics}\n\n"
                    f"Log parser error ({parser_name}): {error}\n\n"
                    f"Raw test output (bounded):\n{raw_output}"
                ),
                reward=0.0,
                finished=True,
                invalid=True,
            )
            assert result.metadata is not None
            result.metadata = {
                **result.metadata,
                "stage": "parse_test_output",
                "parser": parser_name,
                "test_exit_code": test_code,
                "test_patch_apply_exit_code": apply_code,
                **diagnostic_metadata,
            }
            return result

        # 4. Check FAIL_TO_PASS and PASS_TO_PASS
        score = score_test_results(
            test_results,
            self.parsed.FAIL_TO_PASS,
            self.parsed.PASS_TO_PASS,
            reward_mode=os.getenv("OPENREWARD_REWARD_MODE", "binary"),
        )
        f2p_passed = score.fail_to_pass_passed
        p2p_passed = score.pass_to_pass_passed
        ordinary_test_exit = 0 <= test_code <= 1
        reward = score.reward if ordinary_test_exit else 0.0

        # Build a bounded forensic summary. Tool outputs are retained in
        # rollout event artifacts, which makes parser misses, test-runner
        # aborts, and compilation failures distinguishable after the fact.
        f2p_detail, _ = _status_detail(
            self.parsed.FAIL_TO_PASS,
            test_results,
            only_nonpassing=False,
        )
        p2p_detail, p2p_nonpassing = _status_detail(
            self.parsed.PASS_TO_PASS,
            test_results,
            only_nonpassing=True,
        )
        p2p_total = len(self.parsed.PASS_TO_PASS)
        f2p_not_found = sum(
            test_results.get(
                normalize_test_name(test_id),
                "NOT_FOUND",
            )
            == "NOT_FOUND"
            for test_id in self.parsed.FAIL_TO_PASS
        )
        p2p_not_found = sum(
            test_results.get(
                normalize_test_name(test_id),
                "NOT_FOUND",
            )
            == "NOT_FOUND"
            for test_id in self.parsed.PASS_TO_PASS
        )
        raw_output = _bounded_text(
            test_output,
            int(os.getenv("SWE_GRADER_RAW_OUTPUT_MAX_CHARS", "20000")),
            label="raw test output truncated",
        )
        p2p_detail_text = (
            "\n".join(p2p_detail)
            if p2p_detail
            else "  (all expected PASS_TO_PASS tests passed)"
        )

        summary = (
            "Pre-submission repository state:\n"
            f"{submission_diagnostics}\n\n"
            f"Test patch apply exit code: {apply_code}\n"
            f"Test command exit code: {test_code}\n"
            f"Ordinary test exit (0 or 1): {ordinary_test_exit}\n"
            f"Log parser: {parser_name}\n"
            f"Parsed tests: {len(test_results)}\n"
            f"FAIL_TO_PASS: {f2p_passed}/{len(self.parsed.FAIL_TO_PASS)} passed\n"
            f"FAIL_TO_PASS NOT_FOUND: {f2p_not_found}\n"
            f"FAIL_TO_PASS detail:\n" +
            "\n".join(f2p_detail) + "\n"
            f"PASS_TO_PASS: {p2p_passed}/{p2p_total} passed\n"
            f"PASS_TO_PASS nonpassing: {p2p_nonpassing}\n"
            f"PASS_TO_PASS NOT_FOUND: {p2p_not_found}\n"
            f"PASS_TO_PASS nonpassing detail:\n"
            f"{p2p_detail_text}\n"
            f"Reward: {reward}\n\n"
            f"Raw test output (bounded):\n{raw_output}"
        )

        return _text_output(
            summary,
            reward=reward,
            finished=True,
            metadata={
                "ok": ordinary_test_exit,
                "tool": "submit_answer",
                "test_patch_apply_exit_code": apply_code,
                "test_exit_code": test_code,
                "ordinary_test_exit": ordinary_test_exit,
                "parser": parser_name,
                "parsed_test_count": len(test_results),
                "fail_to_pass_passed": f2p_passed,
                "fail_to_pass_total": len(self.parsed.FAIL_TO_PASS),
                "fail_to_pass_not_found": f2p_not_found,
                "pass_to_pass_passed": p2p_passed,
                "pass_to_pass_total": p2p_total,
                "pass_to_pass_nonpassing": p2p_nonpassing,
                "pass_to_pass_not_found": p2p_not_found,
                **diagnostic_metadata,
            },
        )


if __name__ == "__main__":
    port = int(os.getenv("OPENREWARD_PORT", os.getenv("PORT", "8080")))
    Server(environments=[SWERebenchV2]).run(
        host=os.getenv("OPENREWARD_HOST", "0.0.0.0"),
        port=port,
    )
