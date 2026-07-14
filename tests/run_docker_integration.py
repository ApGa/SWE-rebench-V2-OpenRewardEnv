"""End-to-end smoke test for the local Docker sandbox runtime."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from openreward import OpenReward
from openreward.api.environments.types import TextBlock


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smoke_task"
TASK_IMAGE = "swe-rebench-v2-smoke-task:local"
ENVIRONMENT_NAME = "nebius/SWE-rebench-V2"

TEST_PATCH = """\
diff --git a/test_app.py b/test_app.py
new file mode 100644
--- /dev/null
+++ b/test_app.py
@@ -0,0 +1,5 @@
+from app import VALUE
+
+
+def test_value():
+    assert VALUE == "fixed"
"""


def _run(*args: str) -> str:
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(proc: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Environment server exited early:\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("Environment server did not become ready within 30 seconds")


def _write_dataset(data_dir: Path, base_commit: str) -> None:
    row = {
        "instance_id": "local-smoke-0",
        "repo": "local/smoke",
        "base_commit": base_commit,
        "test_patch": TEST_PATCH,
        "problem_statement": 'Change VALUE from "buggy" to "fixed".',
        "image_name": TASK_IMAGE,
        "language": "python",
        "FAIL_TO_PASS": ["test_app.py::test_value"],
        "PASS_TO_PASS": [],
        "install_config": {
            "test_cmd": (
                "if grep -q 'VALUE = \"fixed\"' app.py; "
                "then echo 'PASSED test_app.py::test_value'; "
                "else echo 'FAILED test_app.py::test_value'; exit 1; fi"
            ),
            "log_parser": "parse_log_pytest",
            "install": "",
            "base_image_name": "",
        },
    }
    pq.write_table(pa.Table.from_pylist([row]), data_dir / "data.parquet")


def _docker_socket() -> Path:
    docker_host = os.getenv("DOCKER_HOST", "")
    if docker_host.startswith("unix://"):
        return Path(docker_host.removeprefix("unix://"))
    candidates = [
        Path.home() / ".docker" / "run" / "docker.sock",
        Path("/var/run/docker.sock"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate the Docker daemon socket")


def main(server_image: str | None = None) -> None:
    _run("docker", "build", "-t", TASK_IMAGE, str(FIXTURE))
    base_commit = _run(
        "docker", "run", "--rm", TASK_IMAGE, "git", "rev-parse", "HEAD"
    )

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _write_dataset(data_dir, base_commit)
        port = _free_port()
        url = f"http://127.0.0.1:{port}"

        server_container: str | None = None
        if server_image:
            server_container = f"swe-rebench-server-smoke-{os.getpid()}"
            server_args = [
                "docker",
                "run",
                "--rm",
                "--name",
                server_container,
                "-p",
                f"{port}:8080",
                "-v",
                f"{data_dir}:/orwd_data:ro",
                "-v",
                f"{_docker_socket()}:/var/run/docker.sock",
                "-e",
                "SWE_SANDBOX_RUNTIME=docker",
                server_image,
            ]
            server = subprocess.Popen(
                server_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        else:
            env = os.environ.copy()
            env.update(
                {
                    "DATA_DIR": str(data_dir),
                    "SWE_SANDBOX_RUNTIME": "docker",
                    "OPENREWARD_PORT": str(port),
                    "OPENREWARD_API_URL": url,
                    "OPENREWARD_SESSION_URL": url,
                }
            )
            server = subprocess.Popen(
                [sys.executable, str(ROOT / "server.py")],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        try:
            _wait_for_server(server, port)
            os.environ["OPENREWARD_API_URL"] = url
            os.environ["OPENREWARD_SESSION_URL"] = url
            client = OpenReward(base_url=url)
            environment = client.environments.get(
                name=ENVIRONMENT_NAME,
                base_url=url,
            )

            with environment.session(split="train", index=0) as session:
                result = session.call_tool("submit_answer", {})
                assert result.reward == 0.0, result

            with environment.session(split="train", index=0) as session:
                edit = session.call_tool(
                    "str_replace",
                    {
                        "path": "/workspace/app.py",
                        "old_str": 'VALUE = "buggy"',
                        "new_str": 'VALUE = "fixed"',
                        "description": "Apply the smoke-test fix",
                    },
                )
                assert isinstance(edit.blocks[0], TextBlock), edit
                assert "Exit code: 0" in edit.blocks[0].text, edit
                result = session.call_tool("submit_answer", {})
                assert result.reward == 1.0, result

            print("Local Docker environment smoke test passed (reward 0 -> 1)")
        finally:
            if server_container is not None:
                subprocess.run(
                    ["docker", "rm", "-f", server_container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                server.wait(timeout=10)
            elif server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-image",
        help="Run the environment server from this Docker image",
    )
    args = parser.parse_args()
    main(server_image=args.server_image)
