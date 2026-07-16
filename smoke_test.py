"""Smoke test: verify reward=0 before patch, reward=1 after gold patch.

Usage:
    python smoke_test.py --data-dir ~/data/SWE-rebench-V2
    python smoke_test.py --data-dir ~/data/SWE-rebench-V2 --index 5 -v
"""
import argparse
import base64
import os
from pathlib import Path

from openreward.api.environments.types import TextBlock

from dataset_store import TaskDataset
from task_commands import build_test_command_script

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--data-dir", type=Path, required=True,
    help="Directory containing data.parquet and task_index.json",
)
parser.add_argument(
    "--index", type=int, default=0,
    help="Task index to run (default: 0)",
)
parser.add_argument(
    "--verbose", "-v", action="store_true",
    help="Apply gold+test patches manually, run tests, and show parser diagnostics",
)
parser.add_argument(
    "--base-url",
    default=os.getenv("OPENREWARD_LOCAL_URL", "http://localhost:8080"),
    help="Local environment server URL (default: http://localhost:8080)",
)
args = parser.parse_args()

os.environ["DATA_DIR"] = str(args.data_dir)
os.environ["TASK_INDEX"] = str(args.data_dir / "task_index.json")
os.environ["OPENREWARD_API_URL"] = args.base_url
os.environ["OPENREWARD_SESSION_URL"] = args.base_url

from openreward import OpenReward  # noqa: E402

# Read just this task's evaluation fields from its parquet row group.
dataset = TaskDataset(
    args.data_dir,
    index_path=args.data_dir / "task_index.json",
)
raw_idx = dataset.raw_index(args.index)
row = dataset.get_row(
    args.index,
    columns=[
        "instance_id",
        "patch",
        "test_patch",
        "install_config",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ],
)
instance_id = row["instance_id"]
gold_patch = row["patch"]
test_patch = row["test_patch"]
install_config = row["install_config"]
fail_to_pass = row["FAIL_TO_PASS"]
pass_to_pass = row["PASS_TO_PASS"]

print(f"=== Task {args.index} (raw={raw_idx}): {instance_id} ===")
print(f"Gold patch: {len(gold_patch)} bytes")
print(f"Test patch: {len(test_patch)} bytes")
print(f"Test cmd:   {install_config.get('test_cmd')}")
print(f"Log parser: {install_config.get('log_parser')}")
print(f"FAIL_TO_PASS ({len(fail_to_pass)}): {fail_to_pass[:3]}{'...' if len(fail_to_pass) > 3 else ''}")
print(f"PASS_TO_PASS: {len(pass_to_pass)} tests\n")

or_client = OpenReward(base_url=args.base_url)
environment = or_client.environments.get(name="nebius/SWE-rebench-V2")

ok = True


def _block_text(result) -> str:
    block = result.blocks[0]
    assert isinstance(block, TextBlock)
    return block.text


def apply_patch(session, patch_bytes: str, label: str) -> bool:
    """Apply a patch via bash, with --3way fallback. Returns success."""
    encoded = base64.b64encode(patch_bytes.encode("utf-8")).decode("ascii")
    result = session.call_tool("bash", {
        "command": f"echo '{encoded}' | base64 -d > /tmp/{label}.patch && git apply /tmp/{label}.patch",
        "description": f"Apply {label}",
    })
    text = _block_text(result)
    if "Exit code: 0" in text:
        return True
    # Fallback
    result = session.call_tool("bash", {
        "command": f"git apply --3way /tmp/{label}.patch",
        "description": f"Apply {label} with --3way",
    })
    return "Exit code: 0" in _block_text(result)


# --- Phase 1: submit WITHOUT patch, expect reward=0 ---
print("=" * 60)
print("PHASE 1: Submit without fix (expect reward=0)")
print("=" * 60)

with environment.session(split="train", index=args.index) as session:
    prompt = session.get_prompt()
    first_prompt_block = prompt[0]
    assert isinstance(first_prompt_block, TextBlock)
    print(f"Repo cloned at: {first_prompt_block.text.split('cloned at `')[1].split('`')[0]}")

    result = session.call_tool("submit_answer", {})
    print(_block_text(result))
    pre_reward = result.reward
    print(f"\nReward: {pre_reward}")

    if pre_reward == 0.0:
        print("✓ Correctly fails before patch\n")
    else:
        print("✗ Should have failed before patch!\n")
        ok = False

# -v: diagnose in a disposable session so test-patch files cannot leak into
# the session used for the real submission.
if args.verbose:
    print("=" * 60)
    print("DIAGNOSTIC: Apply gold + test patches and inspect parser output")
    print("=" * 60)
    with environment.session(split="train", index=args.index) as session:
        if not apply_patch(session, gold_patch, "gold"):
            print("✗ Failed to apply gold patch")
            ok = False

        print("\n--- Applying test patch (replicating submit_answer) ---")
        if not apply_patch(session, test_patch, "test"):
            print("✗ Failed to apply test patch")
            ok = False

        test_script = build_test_command_script(install_config["test_cmd"])
        print(f"\n--- Running configured test command(s):\n{test_script} ---")
        result = session.call_tool("bash", {
            "command": test_script,
            "description": "Run tests for diagnostic",
        })
        raw_output = _block_text(result)
        print(raw_output[:5000])
        if len(raw_output) > 5000:
            print(f"\n... ({len(raw_output)} total chars, truncated)")

        print(f"\n--- Parser diagnostic ({install_config['log_parser']}) ---")
        try:
            import log_parsers
            parser_fn = getattr(log_parsers, install_config["log_parser"])
            clean_output = raw_output.rsplit("\nExit code:", 1)[0]
            parsed = parser_fn(clean_output)
            print(f"Parser returned {len(parsed)} test results")
            for key, value in list(parsed.items())[:5]:
                print(f"  {key!r}: {value!r}")
            if len(parsed) > 5:
                print(f"  ... and {len(parsed) - 5} more")
            for test_id in fail_to_pass:
                clean_id = log_parsers.ansi_escape(test_id)
                status = parsed.get(clean_id, "NOT_FOUND")
                print(f"  FAIL_TO_PASS {clean_id!r} -> {status}")
        except Exception as error:
            print(f"Parser error: {error}")
            ok = False

# --- Phase 2: apply gold patch, then submit, expect reward=1 ---
print("=" * 60)
print("PHASE 2: Apply gold patch, then submit (expect reward=1)")
print("=" * 60)

with environment.session(split="train", index=args.index) as session:
    if not apply_patch(session, gold_patch, "gold"):
        print("✗ Failed to apply gold patch")
        ok = False

    # Submit applies the held-out test patch internally, then runs and scores it.
    print("\n--- Submitting ---")
    result = session.call_tool("submit_answer", {})
    print(_block_text(result))
    post_reward = result.reward
    print(f"\nReward: {post_reward}")

    if post_reward == 1.0:
        print("✓ Gold patch passed!\n")
    else:
        print("✗ Gold patch did NOT pass\n")
        ok = False

# --- Summary ---
print("=" * 60)
if ok:
    print(f"✓ {instance_id}: reward 0→1 as expected")
else:
    print(f"✗ {instance_id}: UNEXPECTED RESULTS (pre={pre_reward}, post={post_reward})")

or_client.close()
if not ok:
    raise SystemExit(1)
