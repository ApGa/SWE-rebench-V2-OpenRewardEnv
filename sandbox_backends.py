"""Local sandbox backends for SWE-rebench-V2.

The environment only needs three sandbox operations: start an image, run a
shell command while preserving filesystem changes, and remove the sandbox.
This module provides that small interface for Docker and Enroot so local
self-hosting does not depend on OpenReward's hosted sandbox service.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol
from uuid import uuid4


_PID_NAMESPACE_LAUNCHER = """\
import ctypes
import os
import signal
import sys

PR_SET_PDEATHSIG = 1

expected_parent = int(sys.argv[1])
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))

# PR_SET_PDEATHSIG has a race if the parent exits between fork and prctl.
if os.getppid() != expected_parent:
    os.kill(os.getpid(), signal.SIGKILL)

os.execvp(sys.argv[2], sys.argv[2:])
"""


_PID_NAMESPACE_REAPER = """\
import os
import signal
import sys


def reap_children(_signum=None, _frame=None):
    while True:
        try:
            child_pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except InterruptedError:
            continue
        if child_pid == 0:
            return


# A process inside a PID namespace can signal PID 1 only for signals for
# which PID 1 installed a handler. SIG_IGN deliberately does not provide a
# task-controlled handler that could terminate this reaper. The host-side
# namespace owner uses SIGKILL for teardown.
for protected_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(protected_signal, signal.SIG_IGN)
signal.signal(signal.SIGCHLD, reap_children)

with open(sys.argv[1], "x", encoding="utf-8") as ready_file:
    ready_file.write("ready\\n")

while True:
    signal.pause()
"""


_PID_NAMESPACE_COMMAND_GATE = """\
import os
import sys

marker_fd = os.open(
    sys.argv[1],
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
    0o600,
)
os.write(marker_fd, b"entered\\n")
os.close(marker_fd)
os.execvp(sys.argv[2], sys.argv[2:])
"""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class LocalRunResult:
    output: str
    return_code: int

    @property
    def exit_code(self) -> int:
        return self.return_code

    def __iter__(self) -> Iterator[str | int]:
        yield self.output
        yield self.return_code


class SandboxBackend(Protocol):
    async def start(self) -> None: ...

    async def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> LocalRunResult: ...

    async def stop(self) -> None: ...


async def _run_process(
    args: list[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> LocalRunResult:
    """Run a host command and terminate its process group if interrupted."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return LocalRunResult(
            output=stdout.decode(errors="replace"),
            return_code=proc.returncode or 0,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as error:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Reap the child before returning or propagating cancellation so an
        # interrupted request cannot leave a host-side process behind.
        communicate_task = asyncio.create_task(proc.communicate())
        cancellation_during_reap = False
        while not communicate_task.done():
            try:
                await asyncio.shield(communicate_task)
            except asyncio.CancelledError:
                cancellation_during_reap = True
        stdout, _ = communicate_task.result()

        if isinstance(error, asyncio.CancelledError) or cancellation_during_reap:
            raise asyncio.CancelledError()

        output = stdout.decode(errors="replace")
        if output and not output.endswith("\n"):
            output += "\n"
        output += f"Command timed out after {timeout:g} seconds"
        return LocalRunResult(output=output, return_code=124)


def _process_state_and_start_time(pid: int) -> tuple[str, str] | None:
    """Return a process state and stable identity from Linux procfs."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], fields[19]


def _process_identity_matches(pid: int, start_time: str) -> bool:
    """Return whether ``pid`` still names the same process, including zombies."""
    identity = _process_state_and_start_time(pid)
    return identity is not None and identity[1] == start_time


def _process_pid_namespace_id(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    for line in lines:
        if line.startswith("NSpid:"):
            values = line.split()[1:]
            if values:
                return int(values[-1])
    return None


def _process_namespace_links(pid: int) -> tuple[str, str, str] | None:
    try:
        return tuple(
            os.readlink(f"/proc/{pid}/ns/{namespace}")
            for namespace in ("user", "mnt", "pid")
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _direct_child_pids(pid: int) -> list[int]:
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(child) for child in children.split()]


class DockerSandbox:
    """A persistent writable task container managed by a local Docker daemon."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.name = f"swe-rebench-{uuid4().hex[:16]}"
        self.default_timeout = float(
            os.getenv("SWE_SANDBOX_COMMAND_TIMEOUT_SECONDS", "600")
        )
        self.started = False

    async def start(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("SWE_SANDBOX_RUNTIME=docker requires the docker CLI")

        create_args = [
            "docker",
            "create",
            "--name",
            self.name,
            "--network",
            os.getenv("SWE_DOCKER_NETWORK", "none"),
        ]
        if cpus := os.getenv("SWE_DOCKER_CPUS"):
            create_args += ["--cpus", cpus]
        if memory := os.getenv("SWE_DOCKER_MEMORY"):
            create_args += ["--memory", memory]
        create_args += [
            "--entrypoint",
            "/bin/sh",
            self.image,
            "-c",
            "trap 'exit 0' TERM INT; while :; do sleep 3600; done",
        ]

        result = await _run_process(
            create_args,
            timeout=float(os.getenv("SWE_SANDBOX_CREATE_TIMEOUT_SECONDS", "1800")),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"docker create failed for {self.image}: {result.output.strip()}"
            )

        result = await _run_process(["docker", "start", self.name], timeout=60)
        if result.return_code != 0:
            await self.stop()
            raise RuntimeError(
                f"docker start failed for {self.image}: {result.output.strip()}"
            )
        self.started = True

    async def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> LocalRunResult:
        if not self.started:
            raise RuntimeError("Docker sandbox has not been started")
        result = await _run_process(
            ["docker", "exec", self.name, "/bin/sh", "-c", command],
            timeout=self.default_timeout if timeout is None else timeout,
        )
        if result.return_code == 124:
            # Killing docker exec does not reliably kill its process in the
            # container. Restart the container to terminate the command while
            # preserving the writable filesystem.
            await _run_process(["docker", "kill", self.name], timeout=60)
            restart = await _run_process(["docker", "start", self.name], timeout=60)
            if restart.return_code != 0:
                self.started = False
                result.output += (
                    "\nSandbox restart after timeout failed: " + restart.output.strip()
                )
        return result

    async def stop(self) -> None:
        await _run_process(["docker", "rm", "-f", self.name], timeout=60)
        self.started = False


_DOCKER_HUB_REGISTRIES = frozenset(
    {"docker.io", "index.docker.io", "registry-1.docker.io"}
)


def _enroot_import_uri(image: str) -> str:
    if image.startswith(("dockerd://", "podman://")):
        return image
    if image.startswith("docker://"):
        target = image.removeprefix("docker://")
        for registry in _DOCKER_HUB_REGISTRIES:
            for delimiter in ("#", "/"):
                prefix = f"{registry}{delimiter}"
                if target.startswith(prefix):
                    return f"docker://{target.removeprefix(prefix)}"
        return image

    image = image.removeprefix("https://").removeprefix("http://")
    first, separator, rest = image.partition("/")
    # Enroot 3.5 expects Docker Hub's native shorthand. Treating docker.io as
    # a custom registry (docker://docker.io#...) can fail to parse multi-arch
    # manifests even though docker://namespace/image succeeds.
    if separator and first in _DOCKER_HUB_REGISTRIES:
        return f"docker://{rest}"
    if separator and ("." in first or ":" in first or first == "localhost"):
        return f"docker://{first}#{rest}"
    return f"docker://{image}"


class EnrootSandbox:
    """A persistent writable task sandbox backed by host Enroot."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.name = f"swe-rebench-{uuid4().hex[:16]}"
        cache_default = (
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "swe-rebench-v2"
            / "enroot-images"
        )
        self.image_cache = Path(
            os.getenv("SWE_ENROOT_IMAGE_CACHE", str(cache_default))
        ).expanduser()
        self.default_timeout = float(
            os.getenv("SWE_SANDBOX_COMMAND_TIMEOUT_SECONDS", "600")
        )
        self.root_remap = _env_bool("SWE_ENROOT_ROOT_REMAP", True)
        self.session_tmp_dir: Path | None = None
        self.namespace_control_dir: Path | None = None
        self.started = False
        self._pid_namespace_process: asyncio.subprocess.Process | None = None
        self._pid_namespace_anchor_pid: int | None = None
        self._pid_namespace_anchor_start_time: str | None = None
        self._pid_namespace_links: tuple[str, str, str] | None = None
        self._unshare_path: str | None = None
        self._nsenter_path: str | None = None
        self._stopping = False
        self._container_created = False
        self._stop_task: asyncio.Task[None] | None = None

    def _cached_image_path(self) -> Path:
        digest = hashlib.sha256(self.image.encode()).hexdigest()
        return self.image_cache / f"{digest}.sqsh"

    def _ensure_image_sync(self) -> Path:
        self.image_cache.mkdir(parents=True, exist_ok=True)
        image_path = self._cached_image_path()
        lock_path = image_path.with_suffix(".lock")

        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if image_path.exists() and image_path.stat().st_size > 0:
                return image_path

            with tempfile.NamedTemporaryFile(
                prefix=image_path.stem + ".",
                suffix=".sqsh",
                dir=self.image_cache,
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            tmp_path.unlink(missing_ok=True)

            try:
                result = subprocess.run(
                    [
                        "enroot",
                        "import",
                        "--output",
                        str(tmp_path),
                        _enroot_import_uri(self.image),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=float(
                        os.getenv("SWE_SANDBOX_CREATE_TIMEOUT_SECONDS", "1800")
                    ),
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"enroot import failed for {self.image}: "
                        f"{result.stdout.strip()}"
                    )
                os.replace(tmp_path, image_path)
                return image_path
            finally:
                tmp_path.unlink(missing_ok=True)

    def _namespace_anchor_identity(self) -> tuple[str, str] | None:
        anchor_pid = self._pid_namespace_anchor_pid
        expected_start_time = self._pid_namespace_anchor_start_time
        if anchor_pid is None or expected_start_time is None:
            return None
        process_identity = _process_state_and_start_time(anchor_pid)
        if process_identity is None or process_identity[1] != expected_start_time:
            return None
        return process_identity

    def _namespace_anchor_identity_matches(self) -> bool:
        # A zombie still occupies a PID and must remain visible to teardown.
        # Treating it as gone lets an outer PID-namespace init accumulate one
        # unreaped process per sandbox session.
        return self._namespace_anchor_identity() is not None

    def _require_pid_namespace_anchor(self) -> None:
        process = self._pid_namespace_process
        anchor_pid = self._pid_namespace_anchor_pid
        anchor_identity = self._namespace_anchor_identity()
        if self._stopping:
            raise RuntimeError(
                "Enroot sandbox is stopping; refusing to start a command"
            )
        if (
            process is None
            or process.returncode is not None
            or anchor_pid is None
            or anchor_identity is None
            or anchor_identity[0] == "Z"
            or anchor_pid not in _direct_child_pids(process.pid)
            or _process_pid_namespace_id(anchor_pid) != 1
            or _process_namespace_links(anchor_pid) != self._pid_namespace_links
        ):
            raise RuntimeError(
                "Enroot sandbox PID namespace anchor is not alive; refusing "
                "to execute a task command"
            )
        if self._nsenter_path is None or not os.access(self._nsenter_path, os.X_OK):
            raise RuntimeError(
                "Enroot sandbox requires an executable nsenter; refusing "
                "to execute a task command"
            )

    def _namespace_enter_args(self, command: list[str]) -> list[str]:
        self._require_pid_namespace_anchor()
        assert self._nsenter_path is not None
        assert self._pid_namespace_anchor_pid is not None
        return [
            self._nsenter_path,
            "--target",
            str(self._pid_namespace_anchor_pid),
            "--user",
            "--preserve-credentials",
            "--mount",
            "--pid",
            "--",
            *command,
        ]

    def _namespace_command_args(self, command: list[str]) -> tuple[list[str], Path]:
        if self.namespace_control_dir is None:
            raise RuntimeError("Enroot sandbox namespace control dir is missing")
        marker_path = self.namespace_control_dir / f"entered-{uuid4().hex}"
        return (
            self._namespace_enter_args(
                [
                    sys.executable,
                    "-c",
                    _PID_NAMESPACE_COMMAND_GATE,
                    str(marker_path),
                    *command,
                ]
            ),
            marker_path,
        )

    def _namespace_launch_args(
        self,
        launcher_path: Path,
        reaper_path: Path,
        ready_path: Path,
    ) -> list[str]:
        if self._unshare_path is None:
            raise RuntimeError("Enroot sandbox requires the unshare CLI")
        return [
            sys.executable,
            str(launcher_path),
            str(os.getpid()),
            self._unshare_path,
            "--user",
            "--map-current-user",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "--mount-proc",
            "--",
            sys.executable,
            str(reaper_path),
            str(ready_path),
        ]

    async def _wait_for_pid_namespace_anchor(
        self,
        ready_path: Path,
        timeout: float,
    ) -> int:
        process = self._pid_namespace_process
        assert process is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if process.returncode is not None:
                try:
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=1)
                    detail = stdout.decode(errors="replace").strip()
                except asyncio.TimeoutError:
                    detail = ""
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    "Failed to create Enroot sandbox PID namespace" + suffix
                )

            candidates = [
                child_pid
                for child_pid in _direct_child_pids(process.pid)
                if _process_pid_namespace_id(child_pid) == 1
                and ((identity := _process_state_and_start_time(child_pid)) is not None)
                and identity[0] != "Z"
            ]
            if ready_path.exists() and len(candidates) == 1:
                return candidates[0]
            await asyncio.sleep(0.05)

        raise RuntimeError(
            f"Timed out after {timeout:g} seconds while creating Enroot "
            "sandbox PID namespace"
        )

    async def _start_pid_namespace(self) -> None:
        if self.namespace_control_dir is None:
            raise RuntimeError("Enroot sandbox namespace control dir is missing")
        if self._pid_namespace_process is not None:
            raise RuntimeError("Enroot sandbox PID namespace has already been started")

        self._unshare_path = shutil.which("unshare")
        self._nsenter_path = shutil.which("nsenter")
        missing = [
            name
            for name, path in (
                ("unshare", self._unshare_path),
                ("nsenter", self._nsenter_path),
            )
            if path is None
        ]
        if missing:
            raise RuntimeError(
                "SWE_SANDBOX_RUNTIME=enroot requires the host "
                + " and ".join(missing)
                + " CLI"
            )
        assert self._unshare_path is not None

        launcher_path = self.namespace_control_dir / "launcher.py"
        reaper_path = self.namespace_control_dir / "reaper.py"
        ready_path = self.namespace_control_dir / "ready"
        launcher_path.write_text(_PID_NAMESPACE_LAUNCHER)
        reaper_path.write_text(_PID_NAMESPACE_REAPER)
        launcher_path.chmod(0o600)
        reaper_path.chmod(0o600)

        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self._namespace_launch_args(launcher_path, reaper_path, ready_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        )
        cancelled_during_spawn = False
        while not spawn_task.done():
            try:
                await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                # asyncio may already have forked before create_subprocess_exec
                # returns its handle. Let the spawn finish so ordered teardown
                # can kill the namespace init before its unshare owner.
                cancelled_during_spawn = True
        self._pid_namespace_process = spawn_task.result()
        if cancelled_during_spawn:
            raise asyncio.CancelledError()

        startup_timeout = float(
            os.getenv("SWE_ENROOT_NAMESPACE_START_TIMEOUT_SECONDS", "15")
        )
        anchor_pid = await self._wait_for_pid_namespace_anchor(
            ready_path, startup_timeout
        )
        process_identity = _process_state_and_start_time(anchor_pid)
        namespace_links = _process_namespace_links(anchor_pid)
        host_namespace_links = _process_namespace_links(os.getpid())
        if process_identity is None or namespace_links is None:
            raise RuntimeError("Could not verify Enroot sandbox PID namespace anchor")
        if host_namespace_links is None or any(
            child == host for child, host in zip(namespace_links, host_namespace_links)
        ):
            raise RuntimeError(
                "Enroot sandbox namespace setup did not create distinct "
                "user, mount, and PID namespaces"
            )

        self._pid_namespace_anchor_pid = anchor_pid
        self._pid_namespace_anchor_start_time = process_identity[1]
        self._pid_namespace_links = namespace_links

        probe = await _run_process(
            self._namespace_enter_args(["/bin/true"]),
            timeout=min(startup_timeout, 5),
        )
        if probe.return_code != 0:
            raise RuntimeError(
                "Could not enter Enroot sandbox PID namespace: " + probe.output.strip()
            )
        self._require_pid_namespace_anchor()

        # The running interpreters no longer need these files. Removing them
        # leaves only per-command entry markers in the private control dir.
        launcher_path.unlink(missing_ok=True)
        reaper_path.unlink(missing_ok=True)
        ready_path.unlink(missing_ok=True)

    async def _stop_pid_namespace(self) -> None:
        process = self._pid_namespace_process
        anchor_pid = self._pid_namespace_anchor_pid
        anchor_identity = self._namespace_anchor_identity()
        teardown_identity: tuple[int, str] | None = None
        stop_timeout = float(
            os.getenv("SWE_ENROOT_NAMESPACE_STOP_TIMEOUT_SECONDS", "10")
        )
        stop_deadline = asyncio.get_running_loop().time() + stop_timeout

        if process is not None and process.returncode is None:
            if anchor_pid is not None and anchor_identity is not None:
                if anchor_pid not in _direct_child_pids(process.pid):
                    raise RuntimeError(
                        "Enroot sandbox PID namespace owner no longer owns its init; "
                        "refusing unsafe teardown"
                    )
                teardown_identity = (anchor_pid, anchor_identity[1])
            elif anchor_pid is None:
                # Startup cancellation can arrive before the ready handshake
                # records the init. Wait for the launcher to fork so teardown
                # can still kill the child first and let unshare reap it.
                while process.returncode is None:
                    children = _direct_child_pids(process.pid)
                    if len(children) > 1:
                        raise RuntimeError(
                            "Enroot sandbox PID namespace owner has an unexpected "
                            f"process tree: {children}"
                        )
                    if children:
                        child_identity = _process_state_and_start_time(children[0])
                        if child_identity is not None:
                            teardown_identity = (children[0], child_identity[1])
                            break
                    if asyncio.get_running_loop().time() >= stop_deadline:
                        raise RuntimeError(
                            "Timed out while waiting for the Enroot sandbox PID "
                            "namespace owner to create its init during cleanup"
                        )
                    await asyncio.sleep(0.01)

            if teardown_identity is not None:
                teardown_pid, teardown_start_time = teardown_identity
                identity = _process_state_and_start_time(teardown_pid)
                if (
                    identity is not None
                    and identity[1] == teardown_start_time
                    and identity[0] != "Z"
                ):
                    try:
                        os.kill(teardown_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            # Do not kill the unshare owner first. It is the only process able
            # to wait(2) its PID-namespace init; killing it first reparents a
            # zombie to the production server when that server is outer PID 1.
            try:
                remaining = max(0.0, stop_deadline - asyncio.get_running_loop().time())
                await asyncio.wait_for(process.wait(), timeout=remaining)
            except asyncio.TimeoutError as error:
                raise RuntimeError(
                    "Timed out while waiting for the Enroot sandbox PID "
                    "namespace owner to reap its init"
                ) from error

        if process is not None:
            # Drain its diagnostic pipe after wait() has reaped the owner.
            await process.communicate()

        identities_to_verify: list[tuple[int, str]] = []
        if teardown_identity is not None:
            identities_to_verify.append(teardown_identity)
        if anchor_pid is not None and self._pid_namespace_anchor_start_time is not None:
            identities_to_verify.append(
                (anchor_pid, self._pid_namespace_anchor_start_time)
            )

        deadline = asyncio.get_running_loop().time() + 5
        while any(
            _process_identity_matches(pid, start_time)
            for pid, start_time in identities_to_verify
        ):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    "Enroot sandbox PID namespace init survived teardown or "
                    "remained as a zombie"
                )
            await asyncio.sleep(0.05)

        self._pid_namespace_process = None
        self._pid_namespace_anchor_pid = None
        self._pid_namespace_anchor_start_time = None
        self._pid_namespace_links = None
        self._unshare_path = None
        self._nsenter_path = None

    def _start_args(self, command: str) -> list[str]:
        if self.session_tmp_dir is None:
            raise RuntimeError("Enroot sandbox session tmp has not been created")
        args = [
            "enroot",
            "start",
            "--rw",
        ]
        if self.root_remap:
            args.append("--root")

        mounts = os.getenv("SWE_ENROOT_MOUNTS", "")
        for mount in mounts.split(";"):
            mount = mount.strip()
            if mount:
                args += ["--mount", mount]

        # Enroot's generated /etc/rc establishes the image WORKDIR before it
        # executes our explicit command. Image ENTRYPOINT logic lives in
        # /etc/rc.local, so replace only that file with an empty session-local
        # file: this preserves WORKDIR semantics without running a stale image
        # entrypoint.
        rc_local_path = self.session_tmp_dir / ".openreward-enroot-rc.local"
        args += ["--mount", f"{rc_local_path}:/etc/rc.local"]
        args += ["--mount", f"{self.session_tmp_dir}:/tmp"]
        # Enroot's standard configuration bind-mounts the host home directory
        # into every container. Parallel SWE sessions would therefore make
        # `git config --global` race on the same /root/.gitconfig.lock. Keep
        # Git's global config in the already session-private /tmp mount and
        # export the override for every command, including later agent tools.
        isolated_command = (
            "export GIT_CONFIG_GLOBAL=/tmp/.gitconfig-openreward; " + command
        )
        args += [self.name, "/bin/sh", "-c", isolated_command]
        return args

    async def start(self) -> None:
        if shutil.which("enroot") is None:
            raise RuntimeError("SWE_SANDBOX_RUNTIME=enroot requires the enroot CLI")
        if self.started or self._container_created:
            raise RuntimeError("Enroot sandbox has already been started")

        try:
            image_path = await asyncio.to_thread(self._ensure_image_sync)
            tmp_root_value = os.getenv("SWE_ENROOT_SESSION_TMP_ROOT")
            tmp_root = Path(tmp_root_value).expanduser() if tmp_root_value else None
            if tmp_root is not None:
                tmp_root.mkdir(parents=True, exist_ok=True)
            self.session_tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{self.name}-tmp-",
                    dir=tmp_root,
                )
            )
            self.namespace_control_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{self.name}-pidns-",
                    dir=tmp_root,
                )
            )
            self.namespace_control_dir.chmod(0o700)
            rc_local_path = self.session_tmp_dir / ".openreward-enroot-rc.local"
            rc_local_path.write_text("")
            rc_local_path.chmod(0o600)
            # Mark removal necessary before spawning create: cancellation can
            # arrive after Enroot creates the named rootfs but before the
            # subprocess result reaches this coroutine.
            self._container_created = True
            result = await _run_process(
                ["enroot", "create", "--name", self.name, str(image_path)],
                timeout=float(os.getenv("SWE_SANDBOX_CREATE_TIMEOUT_SECONDS", "1800")),
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"enroot create failed for {self.image}: {result.output.strip()}"
                )
            await self._start_pid_namespace()
            self.started = True
        except BaseException as start_error:
            try:
                await self.stop()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Enroot sandbox startup failed and cleanup did not "
                    f"complete: {cleanup_error}"
                ) from start_error
            raise

    async def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> LocalRunResult:
        if not self.started:
            raise RuntimeError("Enroot sandbox has not been started")
        namespace_args, entered_marker = self._namespace_command_args(
            self._start_args(command)
        )
        try:
            try:
                result = await _run_process(
                    namespace_args,
                    timeout=self.default_timeout if timeout is None else timeout,
                    # Some cluster installations enable an Enroot hook that
                    # bind-mounts the caller's home directory by default.
                    # Never let an untrusted task inherit that host path. An
                    # explicit empty value disables the hook even when the
                    # parent environment sets ENROOT_MOUNT_HOME.
                    env={**os.environ, "ENROOT_MOUNT_HOME": ""},
                )
            except (FileNotFoundError, PermissionError, OSError) as error:
                raise RuntimeError(
                    "Could not enter Enroot sandbox PID namespace; refusing "
                    "to execute a task command"
                ) from error

            if not entered_marker.exists():
                detail = result.output.strip()
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    "Could not enter Enroot sandbox PID namespace; refusing "
                    "to treat the command as executed" + suffix
                )
            self._require_pid_namespace_anchor()
            return result
        finally:
            entered_marker.unlink(missing_ok=True)

    async def _cleanup_resources(self) -> None:
        try:
            await self._stop_pid_namespace()
        except Exception as error:
            # The namespace must be gone before touching the writable rootfs.
            # Preserve all ownership state so a subsequent stop() can retry.
            self.started = False
            raise RuntimeError(
                f"Enroot sandbox namespace cleanup failed: {error}"
            ) from error

        if self._container_created:
            try:
                result = await _run_process(
                    ["enroot", "remove", "-f", self.name],
                    timeout=120,
                )
            except Exception as error:
                self.started = False
                raise RuntimeError(
                    f"Enroot sandbox removal failed: {error}"
                ) from error
            if result.return_code != 0:
                self.started = False
                raise RuntimeError(
                    "Enroot sandbox removal failed: " + result.output.strip()
                )

        if self.session_tmp_dir is not None:
            shutil.rmtree(self.session_tmp_dir, ignore_errors=True)
            self.session_tmp_dir = None
        if self.namespace_control_dir is not None:
            shutil.rmtree(self.namespace_control_dir, ignore_errors=True)
            self.namespace_control_dir = None
        self._container_created = False
        self.started = False
        self._stopping = False

    async def stop(self) -> None:
        if self._stop_task is None or self._stop_task.done():
            self._stopping = True
            self._stop_task = asyncio.create_task(self._cleanup_resources())
        stop_task = self._stop_task
        was_cancelled = False
        try:
            while not stop_task.done():
                try:
                    await asyncio.shield(stop_task)
                except asyncio.CancelledError:
                    # Cleanup owns the namespace and Enroot filesystem. Do
                    # not let request cancellation strand either resource.
                    was_cancelled = True

            stop_task.result()
            if was_cancelled:
                raise asyncio.CancelledError()
        finally:
            if stop_task.done() and self._stop_task is stop_task:
                self._stop_task = None


def create_local_sandbox(runtime: str, image: str) -> SandboxBackend:
    if runtime == "docker":
        return DockerSandbox(image)
    if runtime == "enroot":
        return EnrootSandbox(image)
    raise ValueError(
        f"Unsupported local sandbox runtime {runtime!r}; expected 'docker' or 'enroot'"
    )
