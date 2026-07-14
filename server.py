"""OpenReward environment for SWE-rebench-V2."""
import asyncio
import base64
import os
import re
from pathlib import Path
from typing import Any, cast

from openreward import AsyncOpenReward, SandboxSettings
from openreward.api.sandboxes.types import MachineSize
from openreward.environments import Environment, Server, tool
from openreward.environments.types import Blocks, JSONObject, TextBlock, ToolOutput
from pydantic import BaseModel, Field

from dataset_store import TaskDataset
from log_parsers import TestStatus
from sandbox_backends import create_local_sandbox

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
    test_cmd: str
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

def _text_output(text: str, finished: bool = False) -> ToolOutput:
    return ToolOutput(blocks=[TextBlock(text=text)], finished=finished)


def _result_values(result: Any) -> tuple[str, int]:
    """Normalize hosted and local sandbox command results."""
    output = getattr(result, "output", None)
    if output is not None:
        return str(output), int(
            getattr(result, "return_code", getattr(result, "exit_code", 0))
        )
    output, return_code = result
    return str(output), int(return_code)


def _bounded_output(output: str) -> str:
    max_chars = int(os.getenv("SWE_TOOL_OUTPUT_MAX_CHARS", "50000"))
    if len(output) <= max_chars:
        return output
    omitted = len(output) - max_chars
    start_chars = max_chars // 2
    end_chars = max_chars - start_chars
    return (
        output[:start_chars]
        + f"\n\n... [truncated {omitted} characters] ...\n\n"
        + output[-end_chars:]
    )


# Same pattern as ANSI_ESCAPE_RE in log_parsers.py
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", s).strip()


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
        row["FAIL_TO_PASS"] = [_strip_ansi(t) for t in row["FAIL_TO_PASS"]]
        row["PASS_TO_PASS"] = [_strip_ansi(t) for t in row["PASS_TO_PASS"]]
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

    # ----- tools -----

    @tool
    async def bash(self, input: BashInput) -> ToolOutput:
        """Run a bash command in the container."""
        assert self.workdir is not None, "setup() must run before tools"
        cmd = f"cd {_shell_quote(self.workdir)} && {input.command}"
        result = await self.sandbox.run(cmd)
        output, exit_code = _result_values(result)
        output = _bounded_output(output)
        s = output if output else "(no output)"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def str_replace(self, input: StrReplaceInput) -> ToolOutput:
        """Replace a unique string in a file with another string."""
        res = await self.sandbox.run(f"cat -- {_shell_quote(input.path)}")
        content, exit_code = _result_values(res)
        if exit_code != 0:
            s = content if content else "(no output)"
            return _text_output(f"{s}\nExit code: {exit_code}")

        count = content.count(input.old_str)
        if count == 0:
            return _text_output(f"Error: The string to replace was not found in {input.path}\nExit code: 1")
        if count > 1:
            return _text_output(f"Error: The string to replace appears {count} times in {input.path}. It must be unique.\nExit code: 1")

        new_content = content.replace(input.old_str, input.new_str, 1)
        encoded = base64.b64encode(new_content.encode('utf-8')).decode('ascii')
        write_cmd = f"echo '{encoded}' | base64 -d > {_shell_quote(input.path)}"
        result = await self.sandbox.run(write_cmd)
        output, exit_code = _result_values(result)

        s = output if output else f"Successfully replaced string in {input.path}"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def view(self, input: ViewInput) -> ToolOutput:
        """View file contents or directory listings."""
        res = await self.sandbox.run(f"test -d {_shell_quote(input.path)} && echo 'dir' || echo 'file'")
        output, _ = _result_values(res)
        is_dir = output.strip() == "dir"

        if is_dir:
            cmd = f"find {_shell_quote(input.path)} -maxdepth 2 -not -path '*/\\.*' -not -path '*/node_modules/*' | head -100"
        else:
            if input.view_range:
                start, end = input.view_range
                if end == -1:
                    cmd = f"cat -n {_shell_quote(input.path)} | tail -n +{start}"
                else:
                    cmd = f"cat -n {_shell_quote(input.path)} | sed -n '{start},{end}p'"
            else:
                cmd = f"cat -n {_shell_quote(input.path)}"

        res = await self.sandbox.run(cmd)
        output, exit_code = _result_values(res)

        if len(output) > 16000:
            lines = output.split('\n')
            mid = len(lines) // 2
            keep_start = mid // 2
            keep_end = mid // 2
            output = '\n'.join(lines[:keep_start]) + \
                    f"\n\n... [truncated {len(lines) - keep_start - keep_end} lines] ...\n\n" + \
                    '\n'.join(lines[-keep_end:])

        s = output if output else "(no output)"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def create_file(self, input: CreateFileInput) -> ToolOutput:
        """Create a new file with the specified content."""
        parent_dir = "/".join(input.path.rsplit("/", 1)[:-1])
        if parent_dir:
            await self.sandbox.run(f"mkdir -p {_shell_quote(parent_dir)}")

        encoded = base64.b64encode(input.file_text.encode('utf-8')).decode('ascii')
        write_cmd = f"echo '{encoded}' | base64 -d > {_shell_quote(input.path)}"
        result = await self.sandbox.run(write_cmd)
        output, exit_code = _result_values(result)

        s = output if output else f"Successfully created {input.path}"
        return _text_output(f"{s}\nExit code: {exit_code}")

    @tool
    async def submit_answer(self) -> ToolOutput:
        """Submit your solution. Applies the test patch, runs the test suite, and scores."""
        assert self.workdir is not None, "setup() must run before tools"
        # 1. Write test_patch to a file and apply it
        test_patch_encoded = base64.b64encode(
            self.parsed.test_patch.encode('utf-8')
        ).decode('ascii')
        await self.sandbox.run(
            f"echo '{test_patch_encoded}' | base64 -d > /tmp/test_patch.diff"
        )
        apply_result = await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && git apply /tmp/test_patch.diff"
        )
        apply_output, apply_code = _result_values(apply_result)
        if apply_code != 0:
            # Try with --3way as fallback
            apply_result = await self.sandbox.run(
                f"cd {_shell_quote(self.workdir)} && git apply --3way /tmp/test_patch.diff"
            )
            apply_output, apply_code = _result_values(apply_result)
            if apply_code != 0:
                return ToolOutput(
                    blocks=[TextBlock(text=f"Failed to apply test patch:\n{apply_output}")],
                    reward=0.0,
                    finished=True,
                )

        # 2. Run test command
        test_cmd = self.parsed.install_config.test_cmd
        res = await self.sandbox.run(
            f"cd {_shell_quote(self.workdir)} && {test_cmd}",
            timeout=float(os.getenv("SWE_TEST_TIMEOUT_SECONDS", "600")),
        )
        test_output, test_code = _result_values(res)

        # 3. Parse test output
        parser_name = self.parsed.install_config.log_parser
        try:
            parser_fn = _get_log_parser(parser_name)
            test_results = parser_fn(test_output)
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Log parser error ({parser_name}): {e}\n\nRaw output:\n{test_output[:4000]}")],
                reward=0.0,
                finished=True,
            )

        # 4. Check FAIL_TO_PASS and PASS_TO_PASS
        f2p_passed = sum(
            test_results.get(t) == TestStatus.PASSED.value
            for t in self.parsed.FAIL_TO_PASS
        )
        p2p_passed = sum(
            test_results.get(t) == TestStatus.PASSED.value
            for t in self.parsed.PASS_TO_PASS
        )
        fail_to_pass_ok = f2p_passed == len(self.parsed.FAIL_TO_PASS)
        pass_to_pass_ok = p2p_passed == len(self.parsed.PASS_TO_PASS)

        binary_reward = 1.0 if (fail_to_pass_ok and pass_to_pass_ok) else 0.0
        reward_mode = os.getenv(
            "OPENREWARD_REWARD_MODE", "binary"
        ).strip().lower()
        if reward_mode in {"partial", "fractional"}:
            group_scores = []
            if self.parsed.FAIL_TO_PASS:
                group_scores.append(
                    f2p_passed / len(self.parsed.FAIL_TO_PASS)
                )
            if self.parsed.PASS_TO_PASS:
                group_scores.append(
                    p2p_passed / len(self.parsed.PASS_TO_PASS)
                )
            reward = (
                sum(group_scores) / len(group_scores)
                if group_scores
                else binary_reward
            )
        else:
            reward = binary_reward

        # Build summary
        f2p_detail = []
        for t in self.parsed.FAIL_TO_PASS:
            status = test_results.get(t, "NOT_FOUND")
            f2p_detail.append(f"  {t}: {status}")
        p2p_total = len(self.parsed.PASS_TO_PASS)

        summary = (
            f"Test command exit code: {test_code}\n"
            f"FAIL_TO_PASS: {f2p_passed}/{len(self.parsed.FAIL_TO_PASS)} passed\n"
            f"FAIL_TO_PASS detail:\n" +
            "\n".join(f2p_detail) + "\n"
            f"PASS_TO_PASS: {p2p_passed}/{p2p_total} passed\n"
            f"Reward: {reward}"
        )

        return ToolOutput(
            blocks=[TextBlock(text=summary)],
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    port = int(os.getenv("OPENREWARD_PORT", os.getenv("PORT", "8080")))
    Server(environments=[SWERebenchV2]).run(
        host=os.getenv("OPENREWARD_HOST", "0.0.0.0"),
        port=port,
    )
