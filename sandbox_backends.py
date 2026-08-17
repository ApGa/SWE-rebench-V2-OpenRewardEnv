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
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol
from uuid import uuid4


_CARGO_ENV_PATH = b"/usr/local/cargo/env"
_CARGO_ENV_MISSING_ERRORS = frozenset(
    {
        b"cat: no matches for /usr/local/cargo/env\n",
        b"cat: no matches for /usr/local/cargo\n",
    }
)
_CARGO_SOURCE_LINE = re.compile(rb"[ \t]*(?:\.|source)[ \t]+/usr/local/cargo/env[ \t]*")


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
        stdin_data: bytes | None = None,
    ) -> LocalRunResult: ...

    async def stop(self) -> None: ...


async def _run_process(
    args: list[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    stdin_data: bytes | None = None,
) -> LocalRunResult:
    """Run a host command and terminate its process group if interrupted.

    ``stdin_data`` is passed through a pipe instead of being embedded in
    ``args``. This matters for source files and patches, which can readily
    exceed the host's argv size limit.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout,
        )
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


def _process_state_parent_and_start_time(pid: int) -> tuple[str, int, str] | None:
    """Return state, parent PID, and stable identity in one procfs read."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    command_end = stat.rfind(")")
    if command_end < 0:
        return None
    fields = stat[command_end + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return fields[0], int(fields[1]), fields[19]
    except ValueError:
        return None


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
        self.image = local_image_reference(image)
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
        stdin_data: bytes | None = None,
    ) -> LocalRunResult:
        if not self.started:
            raise RuntimeError("Docker sandbox has not been started")
        exec_args = ["docker", "exec"]
        if stdin_data is not None:
            exec_args.append("-i")
        exec_args += [self.name, "/bin/sh", "-c", command]
        result = await _run_process(
            exec_args,
            timeout=self.default_timeout if timeout is None else timeout,
            stdin_data=stdin_data,
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
_PRIME_SWE_IMAGE_PREFIX = "prime/primeintellect/"


def local_image_reference(image: str) -> str:
    """Translate Prime sandbox image IDs back to their public OCI source.

    PrimeIntellect's verified SWE-rebench subset preserves upstream task data
    but rewrites ``docker.io/swerebenchv2/...`` image names for Prime's sandbox
    registry. Docker and Enroot need the original OCI reference. Reversing that
    documented prefix rewrite also preserves compatibility with existing SQSH
    cache keys.
    """
    if image.startswith(_PRIME_SWE_IMAGE_PREFIX):
        return "docker.io/swerebenchv2/" + image.removeprefix(_PRIME_SWE_IMAGE_PREFIX)
    return image


def _enroot_import_uri(image: str) -> str:
    image = local_image_reference(image)
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


def _enroot_import_environment() -> dict[str, str]:
    """Return an import environment safe for read-only layer directories.

    Enroot 3.5 extracts each OCI layer with GNU tar. Some task images contain
    interleaved archive members which revisit a directory after its read-only
    mode has already been restored. Delaying all directory metadata restoration
    until the end of extraction prevents those later members from failing with
    ``Permission denied``.
    """
    env = dict(os.environ)
    tar_options = env.get("TAR_OPTIONS", "").split()
    if "--delay-directory-restore" not in tar_options:
        tar_options.append("--delay-directory-restore")
    env["TAR_OPTIONS"] = " ".join(tar_options)
    return env


class EnrootSandbox:
    """A persistent writable task sandbox backed by host Enroot."""

    def __init__(self, image: str) -> None:
        self.image = local_image_reference(image)
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
        self.enroot_control_dir: Path | None = None
        self.workdir: str | None = None
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
                attempts = max(1, int(os.getenv("SWE_ENROOT_IMPORT_ATTEMPTS", "3")))
                retry_delay = max(
                    0.0,
                    float(
                        os.getenv(
                            "SWE_ENROOT_IMPORT_RETRY_DELAY_SECONDS",
                            "2",
                        )
                    ),
                )
                timeout = float(
                    os.getenv(
                        "SWE_ENROOT_IMPORT_TIMEOUT_SECONDS",
                        os.getenv(
                            "SWE_SANDBOX_CREATE_TIMEOUT_SECONDS",
                            "1800",
                        ),
                    )
                )
                errors: list[str] = []
                for attempt in range(1, attempts + 1):
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
                            timeout=timeout,
                            check=False,
                            env=(
                                self._enroot_environment(_enroot_import_environment())
                                if self.session_tmp_dir is not None
                                else _enroot_import_environment()
                            ),
                        )
                    except subprocess.TimeoutExpired as error:
                        errors.append(
                            f"attempt {attempt}/{attempts} timed out after "
                            f"{timeout:g}s: {error.stdout or ''}".strip()
                        )
                    else:
                        if (
                            result.returncode == 0
                            and tmp_path.exists()
                            and tmp_path.stat().st_size > 0
                        ):
                            os.replace(tmp_path, image_path)
                            return image_path
                        detail = result.stdout.strip()
                        if result.returncode == 0:
                            detail = (
                                detail + "\n" if detail else ""
                            ) + "enroot import produced no image data"
                        errors.append(
                            f"attempt {attempt}/{attempts} exited "
                            f"{result.returncode}: {detail}"
                        )

                    if attempt < attempts and retry_delay:
                        time.sleep(retry_delay)

                raise RuntimeError(
                    f"enroot import failed for {self.image} after "
                    f"{attempts} attempt(s):\n" + "\n".join(errors)
                )
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

    def _try_reap_adopted_namespace_init(
        self,
        pid: int,
        start_time: str,
    ) -> bool:
        """Reap the exact recorded init if it is our adopted zombie."""
        details = _process_state_parent_and_start_time(pid)
        if details is None or details[2] != start_time:
            return True
        state, parent_pid, _ = details
        if state != "Z" or parent_pid != os.getpid():
            return False

        # A zombie cannot recycle its PID before waitpid. The single procfs
        # read above proves both the recorded start time and that this server
        # is now its parent, so this cannot consume another asyncio child.
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return not _process_identity_matches(pid, start_time)
        if waited_pid == pid:
            return True
        return not _process_identity_matches(pid, start_time)

    async def _stop_pid_namespace(self) -> None:
        process = self._pid_namespace_process
        anchor_pid = self._pid_namespace_anchor_pid
        anchor_identity = self._namespace_anchor_identity()
        teardown_identity: tuple[int, str] | None = None
        signal_namespace_init = False
        stop_timeout = float(
            os.getenv("SWE_ENROOT_NAMESPACE_STOP_TIMEOUT_SECONDS", "10")
        )
        stop_deadline = asyncio.get_running_loop().time() + stop_timeout

        if process is not None and process.returncode is None:
            if anchor_pid is not None and anchor_identity is not None:
                if anchor_pid in _direct_child_pids(process.pid):
                    signal_namespace_init = True
                else:
                    # The owner can exit before asyncio publishes returncode.
                    # Never signal a reparented live process; exact-identity
                    # verification below will reap it once waitable.
                    self._try_reap_adopted_namespace_init(
                        anchor_pid, anchor_identity[1]
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
                            signal_namespace_init = True
                            break
                    if asyncio.get_running_loop().time() >= stop_deadline:
                        raise RuntimeError(
                            "Timed out while waiting for the Enroot sandbox PID "
                            "namespace owner to create its init during cleanup"
                        )
                    await asyncio.sleep(0.01)

            if teardown_identity is not None and signal_namespace_init:
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

        if anchor_pid is not None and self._pid_namespace_anchor_start_time is not None:
            # The owner can die between the direct-child check and wait(). Its
            # Pdeath-killed init is then an exact zombie adopted by outer PID 1.
            self._try_reap_adopted_namespace_init(
                anchor_pid, self._pid_namespace_anchor_start_time
            )

        identities_to_verify: list[tuple[int, str]] = []
        if teardown_identity is not None:
            identities_to_verify.append(teardown_identity)
        if anchor_pid is not None and self._pid_namespace_anchor_start_time is not None:
            identities_to_verify.append(
                (anchor_pid, self._pid_namespace_anchor_start_time)
            )

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            # Pdeath delivery can lag behind reaping the killed owner. Retry
            # only the exact recorded identities until an adopted child
            # reaches waitable zombie state.
            for pid, start_time in identities_to_verify:
                self._try_reap_adopted_namespace_init(pid, start_time)
            remaining_identities = [
                (pid, start_time)
                for pid, start_time in identities_to_verify
                if _process_identity_matches(pid, start_time)
            ]
            if not remaining_identities:
                break
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
        if self.enroot_control_dir is None:
            raise RuntimeError("Enroot sandbox control dir has not been created")
        rc_path = self.enroot_control_dir / "rc"
        args = [
            "enroot",
            "start",
            "--rw",
            "--rc",
            str(rc_path),
        ]
        if self.root_remap:
            args.append("--root")

        mounts = os.getenv("SWE_ENROOT_MOUNTS", "")
        for mount in mounts.split(";"):
            mount = mount.strip()
            if mount:
                args += ["--mount", mount]

        # Keep rc.local empty as defense in depth. The selected SWE corpus was
        # audited to contain comments-only immutable rc.local files; OCI
        # wrappers and entrypoints for those images live in the copied rc.
        rc_local_path = self.session_tmp_dir / ".openreward-enroot-rc.local"
        args += ["--mount", f"{rc_local_path}:/etc/rc.local"]
        args += ["--mount", f"{self.session_tmp_dir}:/tmp"]
        # Enroot's standard configuration bind-mounts the host home directory
        # into every container. Parallel SWE sessions would therefore make
        # `git config --global` race on the same /root/.gitconfig.lock. Keep
        # Git's global config in the already session-private /tmp mount and
        # export the override for every command, including later agent tools.
        if self.workdir is not None:
            quoted_workdir = shlex.quote(self.workdir)
            command = (
                f"mkdir -p -- {quoted_workdir} 2>/dev/null && "
                f"cd {quoted_workdir} && {command}"
            )
        isolated_command = (
            "export GIT_CONFIG_GLOBAL=/tmp/.gitconfig-openreward; " + command
        )
        args += [self.name, "/bin/sh", "-c", isolated_command]
        return args

    def _enroot_environment(
        self,
        base: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if self.enroot_control_dir is None:
            raise RuntimeError("Enroot sandbox control dir has not been created")
        runtime_path = self.enroot_control_dir / "runtime"
        runtime_path.mkdir(mode=0o700, exist_ok=True)
        environment = dict(os.environ if base is None else base)
        environment["ENROOT_MOUNT_HOME"] = ""
        environment["ENROOT_RUNTIME_PATH"] = str(runtime_path)
        return environment

    @staticmethod
    def _workdir_from_enroot_rc(script: str) -> str:
        """Extract the OCI WORKDIR from Enroot's generated command script."""

        workdirs: list[str] = []
        suffix = ["&&", "unset", "OLDPWD", "||", "exit", "1"]
        for line in script.splitlines():
            try:
                words = shlex.split(line, posix=True)
            except ValueError:
                continue
            if len(words) >= 2 and words[0] == "cd" and words[2:] == suffix:
                workdirs.append(words[1])
        if len(workdirs) != 1:
            raise RuntimeError(
                "Could not determine OCI workdir from Enroot command script"
            )
        workdir = workdirs[0]
        if not workdir.startswith("/") or "\x00" in workdir:
            raise RuntimeError(
                f"Invalid OCI workdir in Enroot command script: {workdir!r}"
            )
        return workdir

    @staticmethod
    def _sanitize_enroot_rc(
        script: bytes,
        *,
        cargo_env_exists: bool | None,
    ) -> bytes:
        """Remove one exact, broken Cargo source command and nothing else."""

        if b"\x00" in script:
            raise RuntimeError("Invalid NUL byte in immutable Enroot command script")
        try:
            script.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Immutable Enroot command script is not valid UTF-8"
            ) from exc

        if _CARGO_ENV_PATH not in script:
            return script
        if cargo_env_exists is None:
            raise RuntimeError("Cargo environment path was not inspected")

        source_indexes: list[int] = []
        unknown_references: list[bytes] = []
        # Split only on LF, the physical-line delimiter consumed by POSIX sh.
        # ``bytes.splitlines()`` also splits on CR, VT, and FF and could expose
        # a false command inside an otherwise unsupported shell line.
        lines: list[bytes] = []
        line_start = 0
        while line_start < len(script):
            newline = script.find(b"\n", line_start)
            if newline < 0:
                lines.append(script[line_start:])
                break
            lines.append(script[line_start : newline + 1])
            line_start = newline + 1
        for index, line in enumerate(lines):
            # A CR before LF is shell input, not part of the delimiter.
            body = line[:-1] if line.endswith(b"\n") else line
            if _CARGO_ENV_PATH not in body:
                continue
            if _CARGO_SOURCE_LINE.fullmatch(body):
                source_indexes.append(index)
            else:
                unknown_references.append(body)

        if unknown_references:
            raise RuntimeError(
                "Unsupported Cargo environment reference in immutable Enroot "
                "command script"
            )
        if len(source_indexes) != 1:
            raise RuntimeError(
                "Expected exactly one Cargo environment source command in "
                "immutable Enroot command script"
            )
        if cargo_env_exists:
            return script
        # The observed node-local contamination is line one. Restrict repair
        # to that exact shape so a source-looking heredoc body, continuation,
        # or later command can never be deleted based on lexical appearance.
        if source_indexes[0] != 0:
            raise RuntimeError(
                "Cargo environment source command is not the first physical line"
            )
        del lines[source_indexes[0]]
        return b"".join(lines)

    @staticmethod
    def _cargo_env_exists_sync(
        unsquashfs: str,
        image_path: Path,
        *,
        timeout: float,
    ) -> bool:
        result = subprocess.run(
            [unsquashfs, "-cat", str(image_path), "usr/local/cargo/env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0 and not result.stderr:
            return True
        if (
            result.returncode == 2
            and not result.stdout
            and result.stderr in _CARGO_ENV_MISSING_ERRORS
        ):
            return False
        diagnostic = (
            (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        )
        raise RuntimeError(
            "Could not inspect Cargo environment in cached Enroot image: " + diagnostic
        )

    def _image_runtime_config_sync(self, image_path: Path) -> tuple[str, bytes]:
        """Copy trusted immutable startup bytes and extract OCI WORKDIR."""

        timeout = float(os.getenv("SWE_ENROOT_INSPECT_TIMEOUT_SECONDS", "120"))
        unsquashfs = shutil.which("unsquashfs")
        if unsquashfs is None:
            raise RuntimeError(
                "SWE_SANDBOX_RUNTIME=enroot requires unsquashfs to inspect OCI WORKDIR"
            )
        result = subprocess.run(
            [unsquashfs, "-cat", str(image_path), "etc/rc"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0 or result.stderr:
            diagnostic = (
                (result.stderr or result.stdout)
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise RuntimeError(
                "Could not inspect cached Enroot image command script: " + diagnostic
            )
        cargo_env_exists = None
        if _CARGO_ENV_PATH in result.stdout:
            cargo_env_exists = self._cargo_env_exists_sync(
                unsquashfs,
                image_path,
                timeout=timeout,
            )
        command_script = self._sanitize_enroot_rc(
            result.stdout,
            cargo_env_exists=cargo_env_exists,
        )
        try:
            script_text = command_script.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Immutable Enroot command script is not valid UTF-8"
            ) from exc
        return self._workdir_from_enroot_rc(script_text), command_script

    async def start(self) -> None:
        if shutil.which("enroot") is None:
            raise RuntimeError("SWE_SANDBOX_RUNTIME=enroot requires the enroot CLI")
        if self.started or self._container_created:
            raise RuntimeError("Enroot sandbox has already been started")

        try:
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
            self.enroot_control_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{self.name}-enroot-control-",
                    dir=tmp_root,
                )
            )
            self.enroot_control_dir.chmod(0o700)
            self._enroot_environment()
            image_path = await asyncio.to_thread(self._ensure_image_sync)
            # Inspect immutable image metadata on the host. Never execute the
            # node-local writable rootfs's /etc/rc: in addition to containing
            # OCI ENTRYPOINT logic, that runtime file may be stale or damaged.
            self.workdir, command_script = await asyncio.to_thread(
                self._image_runtime_config_sync,
                image_path,
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
            rc_path = self.enroot_control_dir / "rc"
            # Use startup bytes copied from the immutable SQSH, not the
            # node-local writable rootfs. This preserves OCI wrappers and
            # entrypoints while removing only one narrowly recognized source
            # command when its Cargo environment file is provably absent.
            rc_path.write_bytes(command_script)
            rc_path.chmod(0o700)
            # Mark removal necessary before spawning create: cancellation can
            # arrive after Enroot creates the named rootfs but before the
            # subprocess result reaches this coroutine.
            self._container_created = True
            result = await _run_process(
                ["enroot", "create", "--name", self.name, str(image_path)],
                timeout=float(os.getenv("SWE_SANDBOX_CREATE_TIMEOUT_SECONDS", "1800")),
                env=self._enroot_environment(),
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
        stdin_data: bytes | None = None,
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
                    env=self._enroot_environment(),
                    stdin_data=stdin_data,
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
                    env=self._enroot_environment(),
                )
            except Exception as error:
                self.started = False
                raise RuntimeError(f"Enroot sandbox removal failed: {error}") from error
            if result.return_code != 0:
                self.started = False
                raise RuntimeError(
                    "Enroot sandbox removal failed: " + result.output.strip()
                )

        if self.session_tmp_dir is not None:
            shutil.rmtree(self.session_tmp_dir, ignore_errors=True)
            self.session_tmp_dir = None
        if self.enroot_control_dir is not None:
            shutil.rmtree(self.enroot_control_dir, ignore_errors=True)
            self.enroot_control_dir = None
        self.workdir = None
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
