import asyncio
import base64
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dataset_store
from sandbox_backends import DockerSandbox, LocalRunResult, _run_process


os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
with mock.patch.object(dataset_store, "TaskDataset"):
    server = importlib.import_module("server")


def _task_spec(*, test_patch: str = ""):
    return server.TaskSpec.model_validate(
        {
            "instance_id": "transport-test",
            "repo": "example/repo",
            "base_commit": "abc123",
            "test_patch": test_patch,
            "problem_statement": "Fix the bug",
            "image_name": "example/image:latest",
            "language": "python",
            "FAIL_TO_PASS": ["test_bug"],
            "PASS_TO_PASS": ["test_existing"],
            "install_config": {
                "test_cmd": "run-tests",
                "log_parser": "test_parser",
            },
        }
    )


class RecordingSandbox:
    def __init__(self, *, source_content: str = "") -> None:
        self.source_content = source_content
        self.calls: list[tuple[str, float | None, bytes | None, dict]] = []

    async def run(
        self,
        command: str,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
        **kwargs,
    ) -> LocalRunResult:
        self.calls.append((command, timeout, stdin_data, kwargs))
        if command.startswith("cat -- "):
            return LocalRunResult(self.source_content, 0)
        return LocalRunResult("", 0)


class ExplodingSandbox:
    async def run(self, *args, **kwargs):
        raise RuntimeError("sandbox connection closed")


class HostShellSandbox:
    async def run(
        self,
        command: str,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
        **_kwargs,
    ) -> LocalRunResult:
        return await _run_process(
            ["/bin/sh", "-c", command],
            timeout=timeout,
            stdin_data=stdin_data,
        )


class GradingSandbox:
    async def run(
        self,
        command: str,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
        **_kwargs,
    ) -> LocalRunResult:
        del timeout, stdin_data
        if "__OPENREWARD_STATUS_BEGIN__" in command:
            return LocalRunResult(
                "\n".join(
                    [
                        "__OPENREWARD_STATUS_BEGIN__",
                        " M source.py",
                        "?? new-file.txt",
                        "__OPENREWARD_STATUS_END__",
                        "__OPENREWARD_CHANGED_ENTRY_COUNT__=2",
                        (
                            "__OPENREWARD_TRACKED_DIFF_HASH__="
                            "0123456789abcdef"
                        ),
                        "__OPENREWARD_DIFF_STAT_BEGIN__",
                        " source.py | 2 +-",
                        "__OPENREWARD_DIFF_STAT_END__",
                    ]
                ),
                0,
            )
        if "OPENREWARD_TEST_COMMAND_START" in command:
            return LocalRunResult(
                "raw-output-start\n"
                + ("x" * 300)
                + "\nraw-output-end",
                1,
            )
        return LocalRunResult("", 0)


class NonordinaryTestExitSandbox(GradingSandbox):
    async def run(
        self,
        command: str,
        timeout: float | None = None,
        stdin_data: bytes | None = None,
        **kwargs,
    ) -> LocalRunResult:
        result = await super().run(
            command,
            timeout=timeout,
            stdin_data=stdin_data,
            **kwargs,
        )
        if "OPENREWARD_TEST_COMMAND_START" in command:
            return LocalRunResult(result.output, 7)
        return result


def _environment(sandbox, *, test_patch: str = "", hosted: bool = False):
    environment = object.__new__(server.SWERebenchV2)
    environment.parsed = _task_spec(test_patch=test_patch)
    environment.workdir = "/workspace"
    environment.or_client = object() if hosted else None
    environment.sandbox_settings = None
    environment.sandbox = sandbox
    return environment


class ProcessStdinTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_process_streams_large_stdin_without_argv_payload(self) -> None:
        payload = b"large-payload-marker:" + (b"x" * 512_000)
        result = await _run_process(
            ["/bin/sh", "-c", "wc -c"],
            stdin_data=payload,
        )
        self.assertEqual(result.return_code, 0)
        self.assertEqual(int(result.output.strip()), len(payload))

    async def test_docker_exec_uses_interactive_stdin_only_when_needed(self) -> None:
        sandbox = DockerSandbox("unused")
        sandbox.started = True
        payload = b"content"
        with mock.patch(
            "sandbox_backends._run_process",
            new=mock.AsyncMock(return_value=LocalRunResult("", 0)),
        ) as run_process:
            await sandbox.run("cat > /tmp/file", stdin_data=payload)

        await_args = run_process.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        args, kwargs = await_args
        self.assertEqual(args[0][:3], ["docker", "exec", "-i"])
        self.assertEqual(kwargs["stdin_data"], payload)


class EnvironmentFileTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_hosted_transport_uses_bounded_verified_chunks(self) -> None:
        content = "hosted-large-payload-marker:" + ("h" * 256_000)
        sandbox = RecordingSandbox()
        environment = _environment(sandbox, hosted=True)

        staged_path = await environment._stage_text_file(content)

        append_commands = [
            command
            for command, _timeout, _stdin_data, _kwargs in sandbox.calls
            if "| base64 -d >>" in command
        ]
        self.assertGreater(len(append_commands), 1)
        self.assertLess(max(map(len, append_commands)), 45_000)
        self.assertNotIn(
            "hosted-large-payload-marker",
            "\n".join(call[0] for call in sandbox.calls),
        )
        decoded_chunks = []
        for command in append_commands:
            encoded_chunk = command.split("printf '%s' '", 1)[1].split(
                "' | base64", 1
            )[0]
            decoded_chunks.append(base64.b64decode(encoded_chunk))
        self.assertEqual(b"".join(decoded_chunks), content.encode())
        self.assertTrue(
            any(
                command.startswith('test "$(wc -c < ')
                and f"-eq {len(content.encode())}" in command
                for command, _timeout, _stdin_data, _kwargs in sandbox.calls
            )
        )
        self.assertTrue(staged_path.startswith("/tmp/.openreward-upload-"))

    async def test_atomic_install_preserves_mode_and_exact_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "source.py"
            target.write_text("replace-me\n")
            target.chmod(0o755)
            environment = _environment(HostShellSandbox())

            result = await environment.str_replace(
                server.StrReplaceInput(
                    path=str(target),
                    old_str="replace-me",
                    new_str="replacement",
                    description="test",
                )
            )

            self.assertTrue(result.metadata["ok"])
            self.assertEqual(target.read_text(), "replacement\n")
            self.assertEqual(target.stat().st_mode & 0o777, 0o755)
            self.assertEqual(
                list(Path(temporary_directory).glob(".openreward-write.*")),
                [],
            )

    async def test_large_str_replace_uses_stdin_not_command_argv(self) -> None:
        old_content = "replace-me\n" + ("x" * 512_000)
        new_content = "replacement\n" + ("x" * 512_000)
        sandbox = RecordingSandbox(source_content=old_content)
        environment = _environment(sandbox)

        result = await environment.str_replace(
            server.StrReplaceInput(
                path="/workspace/source.py",
                old_str="replace-me",
                new_str="replacement",
                description="test",
            )
        )

        self.assertTrue(result.metadata["ok"])
        stdin_payloads = [
            stdin_data
            for _command, _timeout, stdin_data, _kwargs in sandbox.calls
            if stdin_data is not None
        ]
        self.assertEqual(stdin_payloads, [new_content.encode()])
        commands = "\n".join(call[0] for call in sandbox.calls)
        self.assertNotIn("large-payload-marker", commands)
        self.assertNotIn("base64", commands)
        self.assertLess(max(map(len, (call[0] for call in sandbox.calls))), 5000)

    async def test_large_create_file_uses_stdin_not_command_argv(self) -> None:
        content = "large-payload-marker:" + ("y" * 512_000)
        sandbox = RecordingSandbox()
        environment = _environment(sandbox)

        result = await environment.create_file(
            server.CreateFileInput(
                path="/workspace/new/source.py",
                file_text=content,
                description="test",
            )
        )

        self.assertTrue(result.metadata["ok"])
        self.assertEqual(
            [call[2] for call in sandbox.calls if call[2] is not None],
            [content.encode()],
        )
        commands = "\n".join(call[0] for call in sandbox.calls)
        self.assertNotIn("large-payload-marker", commands)
        self.assertNotIn("base64", commands)

    async def test_large_hidden_patch_uses_stdin_not_command_argv(self) -> None:
        patch = "large-payload-marker:" + ("z" * 512_000)
        sandbox = RecordingSandbox()
        environment = _environment(sandbox, test_patch=patch)

        with mock.patch.object(
            server,
            "_get_log_parser",
            return_value=lambda _output: {
                "test_bug": "PASSED",
                "test_existing": "PASSED",
            },
        ):
            result = await environment.submit_answer()

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(
            [call[2] for call in sandbox.calls if call[2] is not None],
            [patch.encode()],
        )
        commands = "\n".join(call[0] for call in sandbox.calls)
        self.assertNotIn("large-payload-marker", commands)
        self.assertNotIn("base64", commands)
        self.assertIn("git apply", commands)

    async def test_all_tool_exceptions_return_structured_outputs(self) -> None:
        environment = _environment(
            ExplodingSandbox(),
            test_patch="test patch",
        )
        results = [
            await environment.bash(
                server.BashInput(command="true", description="test")
            ),
            await environment.str_replace(
                server.StrReplaceInput(
                    path="/workspace/source.py",
                    old_str="old",
                    new_str="new",
                    description="test",
                )
            ),
            await environment.view(
                server.ViewInput(path="/workspace/source.py", description="test")
            ),
            await environment.create_file(
                server.CreateFileInput(
                    path="/workspace/new.py",
                    file_text="content",
                    description="test",
                )
            ),
            await environment.submit_answer(),
        ]

        for result in results:
            with self.subTest(result=result):
                self.assertIsNotNone(result.metadata)
                self.assertFalse(result.metadata["ok"])
                self.assertEqual(result.metadata["exit_code"], 1)
                self.assertEqual(result.metadata["error_type"], "RuntimeError")
                self.assertIn(
                    "sandbox connection closed",
                    result.blocks[0].text,
                )
        self.assertTrue(results[-1].metadata["invalid"])
        self.assertEqual(
            results[-1].metadata["failure_class"],
            "infrastructure",
        )

    async def test_submission_retains_bounded_forensic_diagnostics(self) -> None:
        environment = _environment(
            GradingSandbox(),
            test_patch="held-out patch",
        )
        with (
            mock.patch.object(
                server,
                "_get_log_parser",
                return_value=lambda _output: {
                    "test_bug": "FAILED",
                    "test_existing": "PASSED",
                },
            ),
            mock.patch.dict(
                os.environ,
                {
                    "SWE_GRADER_RAW_OUTPUT_MAX_CHARS": "100",
                    "SWE_GRADER_DIAGNOSTIC_MAX_CHARS": "2000",
                },
            ),
        ):
            result = await environment.submit_answer()

        self.assertEqual(result.reward, 0.0)
        assert result.metadata is not None
        self.assertEqual(result.metadata["changed_entry_count"], 2)
        self.assertEqual(
            result.metadata["changed_files_preview"],
            [" M source.py", "?? new-file.txt"],
        )
        self.assertEqual(
            result.metadata["tracked_diff_hash"],
            "0123456789abcdef",
        )
        self.assertEqual(result.metadata["parsed_test_count"], 2)
        self.assertEqual(result.metadata["fail_to_pass_not_found"], 0)
        self.assertEqual(result.metadata["pass_to_pass_not_found"], 0)
        text = result.blocks[0].text
        self.assertIn(" M source.py", text)
        self.assertIn("source.py | 2 +-", text)
        self.assertIn("raw-output-start", text)
        self.assertIn("raw-output-end", text)
        self.assertIn("raw test output truncated", text)
        self.assertLess(len(text), 3000)

    async def test_nonordinary_test_exit_cannot_receive_success_reward(
        self,
    ) -> None:
        environment = _environment(
            NonordinaryTestExitSandbox(),
            test_patch="held-out patch",
        )
        with mock.patch.object(
            server,
            "_get_log_parser",
            return_value=lambda _output: {
                "test_bug": "PASSED",
                "test_existing": "PASSED",
            },
        ):
            result = await environment.submit_answer()

        self.assertEqual(result.reward, 0.0)
        assert result.metadata is not None
        self.assertFalse(result.metadata["ok"])
        self.assertFalse(result.metadata["ordinary_test_exit"])
        self.assertEqual(result.metadata["test_exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
