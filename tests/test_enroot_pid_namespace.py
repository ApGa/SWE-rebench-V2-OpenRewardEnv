import asyncio
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import sandbox_backends
from sandbox_backends import EnrootSandbox, LocalRunResult


class EnrootNamespaceContractTest(unittest.TestCase):
    def test_uses_required_unshare_and_nsenter_contract(self) -> None:
        sandbox = EnrootSandbox("example.invalid/task:latest")
        sandbox._unshare_path = "/usr/bin/unshare"
        launch_args = sandbox._namespace_launch_args(
            Path("/control/launcher.py"),
            Path("/control/reaper.py"),
            Path("/control/ready"),
        )
        self.assertEqual(
            launch_args,
            [
                sandbox_backends.sys.executable,
                "/control/launcher.py",
                str(os.getpid()),
                "/usr/bin/unshare",
                "--user",
                "--map-current-user",
                "--pid",
                "--fork",
                "--kill-child=SIGKILL",
                "--mount-proc",
                "--",
                sandbox_backends.sys.executable,
                "/control/reaper.py",
                "/control/ready",
            ],
        )

        sandbox._nsenter_path = "/usr/bin/nsenter"
        sandbox._pid_namespace_anchor_pid = 4321
        with mock.patch.object(sandbox, "_require_pid_namespace_anchor"):
            self.assertEqual(
                sandbox._namespace_enter_args(["/bin/true"]),
                [
                    "/usr/bin/nsenter",
                    "--target",
                    "4321",
                    "--user",
                    "--preserve-credentials",
                    "--mount",
                    "--pid",
                    "--",
                    "/bin/true",
                ],
            )

    def test_reaper_protects_pid_one_and_launcher_sets_parent_death_signal(
        self,
    ) -> None:
        reaper = sandbox_backends._PID_NAMESPACE_REAPER
        for protected_signal in ("signal.SIGHUP", "signal.SIGINT", "signal.SIGTERM"):
            self.assertIn(protected_signal, reaper)
        self.assertIn("signal.SIG_IGN", reaper)
        self.assertIn("os.waitpid(-1, os.WNOHANG)", reaper)

        launcher = sandbox_backends._PID_NAMESPACE_LAUNCHER
        self.assertIn("PR_SET_PDEATHSIG", launcher)
        self.assertIn("signal.SIGKILL", launcher)
        self.assertIn("os.getppid() != expected_parent", launcher)

    def test_missing_namespace_tool_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = EnrootSandbox("example.invalid/task:latest")
            sandbox.namespace_control_dir = Path(tmp)

            def which(name: str) -> str | None:
                if name == "unshare":
                    return "/usr/bin/unshare"
                if name == "nsenter":
                    return None
                raise AssertionError(name)

            async def exercise() -> None:
                with mock.patch.object(sandbox_backends.shutil, "which", which):
                    with self.assertRaisesRegex(RuntimeError, "nsenter"):
                        await sandbox._start_pid_namespace()
                self.assertIsNone(sandbox._pid_namespace_process)

            asyncio.run(exercise())

    def test_denied_unshare_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            denied_unshare = root / "unshare"
            denied_unshare.write_text("#!/bin/sh\nprintf 'unshare denied'\nexit 1\n")
            denied_unshare.chmod(0o700)
            sandbox = EnrootSandbox("example.invalid/task:latest")
            sandbox.namespace_control_dir = root / "control"
            sandbox.namespace_control_dir.mkdir()

            real_nsenter = shutil.which("nsenter")

            def which(name: str) -> str | None:
                if name == "unshare":
                    return str(denied_unshare)
                if name == "nsenter":
                    return real_nsenter
                raise AssertionError(name)

            async def exercise() -> None:
                try:
                    with mock.patch.object(sandbox_backends.shutil, "which", which):
                        with self.assertRaisesRegex(RuntimeError, "unshare denied"):
                            await sandbox._start_pid_namespace()
                finally:
                    await sandbox._stop_pid_namespace()

            asyncio.run(exercise())

    def test_missing_entry_handshake_is_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = EnrootSandbox("example.invalid/task:latest")
            sandbox.started = True
            sandbox.session_tmp_dir = root / "task-tmp"
            sandbox.session_tmp_dir.mkdir()
            sandbox.namespace_control_dir = root / "control"
            sandbox.namespace_control_dir.mkdir()
            marker = sandbox.namespace_control_dir / "never-created"

            async def exercise() -> None:
                with mock.patch.object(
                    sandbox,
                    "_namespace_command_args",
                    return_value=(["/bin/sh", "-c", "exit 77"], marker),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "refusing to treat the command as executed"
                    ):
                        await sandbox.run("echo must-not-run")

            asyncio.run(exercise())

    def test_cancelled_stop_finishes_namespace_before_enroot_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = EnrootSandbox("example.invalid/task:latest")
            sandbox.started = True
            sandbox._container_created = True
            sandbox.session_tmp_dir = root / "task-tmp"
            sandbox.session_tmp_dir.mkdir()
            sandbox.namespace_control_dir = root / "control"
            sandbox.namespace_control_dir.mkdir()
            events: list[str] = []

            async def exercise() -> None:
                namespace_stopping = asyncio.Event()
                allow_namespace_stop = asyncio.Event()

                async def slow_namespace_stop() -> None:
                    events.append("namespace-start")
                    namespace_stopping.set()
                    await allow_namespace_stop.wait()
                    events.append("namespace-finished")

                async def fake_run_process(
                    args: list[str], **_kwargs: object
                ) -> LocalRunResult:
                    self.assertEqual(args[:3], ["enroot", "remove", "-f"])
                    events.append("enroot-remove")
                    return LocalRunResult("", 0)

                with (
                    mock.patch.object(
                        sandbox, "_stop_pid_namespace", new=slow_namespace_stop
                    ),
                    mock.patch.object(
                        sandbox_backends, "_run_process", new=fake_run_process
                    ),
                ):
                    stop_task = asyncio.create_task(sandbox.stop())
                    await namespace_stopping.wait()
                    stop_task.cancel()
                    allow_namespace_stop.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await stop_task

            asyncio.run(exercise())
            self.assertEqual(
                events,
                ["namespace-start", "namespace-finished", "enroot-remove"],
            )
            self.assertFalse(sandbox.started)
            self.assertIsNone(sandbox.session_tmp_dir)
            self.assertIsNone(sandbox.namespace_control_dir)


@unittest.skipUnless(
    os.path.exists("/proc/self/ns/pid")
    and shutil.which("unshare") is not None
    and shutil.which("nsenter") is not None,
    "Linux user/PID namespace tools are unavailable",
)
class EnrootNamespaceIntegrationTest(unittest.TestCase):
    def _run_namespace_test(self, body) -> None:
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sandbox = EnrootSandbox("example.invalid/task:latest")
                sandbox.namespace_control_dir = root / "control"
                sandbox.namespace_control_dir.mkdir()
                try:
                    try:
                        await sandbox._start_pid_namespace()
                    except RuntimeError as error:
                        if "Operation not permitted" in str(error):
                            self.skipTest(str(error))
                        raise
                    await body(sandbox, root)
                finally:
                    await sandbox._stop_pid_namespace()

        asyncio.run(exercise())

    def test_anchor_persists_background_process_and_ignores_term_signals(
        self,
    ) -> None:
        async def body(sandbox: EnrootSandbox, _root: Path) -> None:
            anchor_pid = sandbox._pid_namespace_anchor_pid
            namespace_links = sandbox._pid_namespace_links
            assert anchor_pid is not None

            started = await sandbox_backends._run_process(
                sandbox._namespace_enter_args(
                    [
                        "/bin/sh",
                        "-c",
                        "sleep 30 </dev/null >/dev/null 2>&1 & echo $!",
                    ]
                ),
                timeout=5,
            )
            self.assertEqual(started.return_code, 0, started.output)
            background_pid = int(started.output.strip())

            for protected_signal in (
                signal.SIGHUP,
                signal.SIGINT,
                signal.SIGTERM,
            ):
                os.kill(anchor_pid, protected_signal)
            await asyncio.sleep(0.1)

            checked = await sandbox_backends._run_process(
                sandbox._namespace_enter_args(
                    ["/bin/sh", "-c", f"kill -0 {background_pid}"]
                ),
                timeout=5,
            )
            self.assertEqual(checked.return_code, 0, checked.output)
            self.assertEqual(sandbox._pid_namespace_anchor_pid, anchor_pid)
            self.assertEqual(sandbox._pid_namespace_links, namespace_links)
            sandbox._require_pid_namespace_anchor()
            with self.assertRaisesRegex(RuntimeError, "already been started"):
                await sandbox._start_pid_namespace()
            self.assertEqual(sandbox._pid_namespace_anchor_pid, anchor_pid)
            sandbox._require_pid_namespace_anchor()

        self._run_namespace_test(body)

    def test_dead_anchor_fails_closed(self) -> None:
        async def body(sandbox: EnrootSandbox, root: Path) -> None:
            anchor_pid = sandbox._pid_namespace_anchor_pid
            assert anchor_pid is not None
            sandbox.session_tmp_dir = root / "task-tmp"
            sandbox.session_tmp_dir.mkdir()
            sandbox.started = True
            os.kill(anchor_pid, signal.SIGKILL)
            for _ in range(100):
                if not sandbox._namespace_anchor_identity_matches():
                    break
                await asyncio.sleep(0.02)
            with self.assertRaisesRegex(RuntimeError, "anchor is not alive"):
                await sandbox.run("touch /tmp/must-not-exist")

        self._run_namespace_test(body)

    def test_server_parent_death_kills_namespace_anchor(self) -> None:
        child_code = """
import asyncio
import os
import sys
from pathlib import Path

from sandbox_backends import EnrootSandbox


async def main():
    sandbox = EnrootSandbox("parent-death-probe")
    sandbox.namespace_control_dir = Path(sys.argv[1])
    sandbox.namespace_control_dir.mkdir()
    await sandbox._start_pid_namespace()
    Path(sys.argv[2]).write_text(str(sandbox._pid_namespace_anchor_pid))
    os._exit(0)


asyncio.run(main())
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "anchor-pid"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(root / "control"),
                    str(marker),
                ],
                cwd=Path(__file__).resolve().parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
                check=False,
            )
            if (
                completed.returncode != 0
                and "Operation not permitted" in completed.stdout
            ):
                self.skipTest(completed.stdout.strip())
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertTrue(marker.exists(), completed.stdout)
            anchor_pid = int(marker.read_text())
            deadline = time.monotonic() + 5
            while Path(f"/proc/{anchor_pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            try:
                self.assertFalse(
                    Path(f"/proc/{anchor_pid}").exists(),
                    "PID namespace anchor survived its server parent",
                )
            finally:
                if Path(f"/proc/{anchor_pid}").exists():
                    os.kill(anchor_pid, signal.SIGKILL)

    @unittest.skipUnless(
        os.getenv("SWE_ENROOT_EXISTING_SMOKE_CONTAINER"),
        "set SWE_ENROOT_EXISTING_SMOKE_CONTAINER for an actual Enroot probe",
    )
    def test_actual_enroot_runs_inside_persistent_namespace(self) -> None:
        container_name = os.environ["SWE_ENROOT_EXISTING_SMOKE_CONTAINER"]

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sandbox = EnrootSandbox("existing-smoke-container")
                sandbox.name = container_name
                sandbox.session_tmp_dir = root / "task-tmp"
                sandbox.session_tmp_dir.mkdir()
                rc_local = sandbox.session_tmp_dir / ".openreward-enroot-rc.local"
                rc_local.write_text("")
                rc_local.chmod(0o600)
                sandbox.namespace_control_dir = root / "control"
                sandbox.namespace_control_dir.mkdir()
                try:
                    await sandbox._start_pid_namespace()
                    sandbox.started = True
                    result = await sandbox.run(
                        "echo task-pid=$$; echo pid1=$(cat /proc/1/comm)"
                    )
                    self.assertEqual(result.return_code, 0, result.output)
                    self.assertIn("task-pid=", result.output)
                    self.assertIn("pid1=", result.output)
                    sandbox._require_pid_namespace_anchor()
                finally:
                    sandbox.started = False
                    await sandbox._stop_pid_namespace()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
