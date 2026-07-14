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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol
from uuid import uuid4


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
    """Run a host command and terminate its process group on timeout."""
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
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace")
        if output and not output.endswith("\n"):
            output += "\n"
        output += f"Command timed out after {timeout:g} seconds"
        return LocalRunResult(output=output, return_code=124)


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
            "-lc",
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
            ["docker", "exec", self.name, "/bin/sh", "-lc", command],
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
                    "\nSandbox restart after timeout failed: "
                    + restart.output.strip()
                )
        return result

    async def stop(self) -> None:
        await _run_process(["docker", "rm", "-f", self.name], timeout=60)
        self.started = False


def _enroot_import_uri(image: str) -> str:
    if image.startswith(("docker://", "dockerd://", "podman://")):
        return image

    image = image.removeprefix("https://").removeprefix("http://")
    first, separator, rest = image.partition("/")
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
        self.started = False

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

    def _start_args(self, command: str) -> list[str]:
        args = ["enroot", "start", "--rw"]
        if self.root_remap:
            args.append("--root")

        mounts = os.getenv("SWE_ENROOT_MOUNTS", "")
        for mount in mounts.split(";"):
            mount = mount.strip()
            if mount:
                args += ["--mount", mount]

        args += [self.name, "/bin/sh", "-lc", command]
        return args

    async def start(self) -> None:
        if shutil.which("enroot") is None:
            raise RuntimeError("SWE_SANDBOX_RUNTIME=enroot requires the enroot CLI")

        image_path = await asyncio.to_thread(self._ensure_image_sync)
        result = await _run_process(
            ["enroot", "create", "--name", self.name, str(image_path)],
            timeout=float(os.getenv("SWE_SANDBOX_CREATE_TIMEOUT_SECONDS", "1800")),
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"enroot create failed for {self.image}: {result.output.strip()}"
            )
        self.started = True

    async def run(
        self,
        command: str,
        timeout: float | None = None,
    ) -> LocalRunResult:
        if not self.started:
            raise RuntimeError("Enroot sandbox has not been started")
        return await _run_process(
            self._start_args(command),
            timeout=self.default_timeout if timeout is None else timeout,
        )

    async def stop(self) -> None:
        await _run_process(["enroot", "remove", "-f", self.name], timeout=120)
        self.started = False


def create_local_sandbox(runtime: str, image: str) -> SandboxBackend:
    if runtime == "docker":
        return DockerSandbox(image)
    if runtime == "enroot":
        return EnrootSandbox(image)
    raise ValueError(
        f"Unsupported local sandbox runtime {runtime!r}; "
        "expected 'docker' or 'enroot'"
    )
