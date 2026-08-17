import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from build_index import build_index
from dataset_store import TaskDataset
from sandbox_backends import (
    EnrootSandbox,
    LocalRunResult,
    _enroot_import_environment,
    _enroot_import_uri,
    local_image_reference,
)


async def _inline_to_thread(function, /, *args, **kwargs):
    """Keep subprocess-mocked unit tests independent of executor shutdown."""

    return function(*args, **kwargs)


_SWE_RC_CLASS_FIXTURES = {
    "generic": b"""# io.buildah.version 1.38.1
# org.opencontainers.image.ref.name ubuntu
# org.opencontainers.image.version 22.04

mkdir -p "/cloud-provider-azure" 2> /dev/null
cd "/cloud-provider-azure" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec  "$@"
else
    exec  '/bin/bash'
fi
""",
    "python": b"""# io.buildah.version 1.38.1

mkdir -p "/PlasmaPy" 2> /dev/null
cd "/PlasmaPy" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec  "$@"
else
    exec  'python3'
fi
""",
    "php": b"""# io.buildah.version 1.38.1

mkdir -p "/phpstan-disallowed-calls" 2> /dev/null
cd "/phpstan-disallowed-calls" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec 'docker-php-entrypoint' "$@"
else
    exec 'docker-php-entrypoint' 'php' '-a'
fi
""",
    "bash": b"""# io.buildah.version 1.38.1
# org.opencontainers.image.source https://github.com/rust-lang/docker-rust

mkdir -p "/salsa" 2> /dev/null
cd "/salsa" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec  "$@"
else
    exec  'bash'
fi
""",
    "cacert-jshell": b"""# io.buildah.version 1.38.1
# org.opencontainers.image.ref.name ubuntu
# org.opencontainers.image.version 24.04

mkdir -p "/laboratory" 2> /dev/null
cd "/laboratory" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec '/__cacert_entrypoint.sh' "$@"
else
    exec '/__cacert_entrypoint.sh' 'jshell'
fi
""",
    "cacert-sbt": b"""# io.buildah.version 1.40.1
# org.opencontainers.image.ref.name ubuntu
# org.opencontainers.image.version 24.04

mkdir -p "/scalameta" 2> /dev/null
cd "/scalameta" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec '/__cacert_entrypoint.sh' "$@"
else
    exec '/__cacert_entrypoint.sh' 'sbt'
fi
""",
    "maven": b"""# io.buildah.version 1.38.1
# org.opencontainers.image.description Apache Maven is a software project management and comprehension tool. Based on the concept of a project object model (POM), Maven can manage a project's build, reporting and documentation from a central piece of information.
# org.opencontainers.image.ref.name ubuntu
# org.opencontainers.image.source https://github.com/carlossg/docker-maven
# org.opencontainers.image.title Apache Maven
# org.opencontainers.image.url https://github.com/carlossg/docker-maven
# org.opencontainers.image.version 24.04

mkdir -p "/kafka-topology-builder" 2> /dev/null
cd "/kafka-topology-builder" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec '/usr/local/bin/mvn-entrypoint.sh' "$@"
else
    exec '/usr/local/bin/mvn-entrypoint.sh' 'mvn'
fi
""",
    "R": b"""# io.buildah.version 1.38.1
# org.opencontainers.image.authors Carl Boettiger <cboettig@ropensci.org>
# org.opencontainers.image.base.name docker.io/library/ubuntu:jammy
# org.opencontainers.image.description Reproducible builds to fixed version of R
# org.opencontainers.image.licenses GPL-2.0-or-later
# org.opencontainers.image.ref.name ubuntu
# org.opencontainers.image.revision 2393f40bb1366538e21d1137803b3c1dfec4d2e1
# org.opencontainers.image.source https://github.com/rocker-org/rocker-versioned2
# org.opencontainers.image.title rocker/r-ver
# org.opencontainers.image.vendor Rocker Project
# org.opencontainers.image.version R-4.4.1

mkdir -p "/scoringutils" 2> /dev/null
cd "/scoringutils" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec  "$@"
else
    exec  'R'
fi
""",
}


def _task(instance_id: str, test_cmd: str | list[str]) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "base_commit": "abc123",
        "patch": f"gold patch for {instance_id}",
        "test_patch": "",
        "problem_statement": "Fix the bug",
        "image_name": "docker.io/library/alpine:3.20",
        "language": "python",
        "FAIL_TO_PASS": ["test_bug"],
        "PASS_TO_PASS": ["test_existing"],
        "install_config": json.dumps(
            {
                "test_cmd": test_cmd,
                "log_parser": "parse_pytest",
                "install": "",
                "base_image_name": "",
            }
        ),
    }


class TaskDatasetTest(unittest.TestCase):
    def test_reads_shards_lazily_and_applies_valid_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            rows = [_task("task-0", "pytest"), _task("task-1", "")]
            shards_dir = data_dir / "data"
            shards_dir.mkdir()
            pq.write_table(
                pa.Table.from_pylist(rows),
                shards_dir / "part-0.parquet",
                row_group_size=1,
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        _task(
                            "task-2",
                            ["go test ./...", "go test ./pkg/..."],
                        ),
                        _task("task-3", []),
                    ]
                ),
                shards_dir / "part-1.parquet",
                row_group_size=1,
            )

            index = build_index(data_dir)
            self.assertEqual(index["valid_indices"], [0, 2])
            index_path = data_dir / "task_index.json"
            index_path.write_text(json.dumps(index))

            dataset = TaskDataset(data_dir)
            self.assertEqual(dataset.num_tasks, 2)
            self.assertEqual(dataset.get_task(0)["instance_id"], "task-0")
            self.assertEqual(dataset.get_task(1)["instance_id"], "task-2")
            self.assertEqual(dataset.raw_index(1), 2)
            self.assertEqual(
                dataset.get_row(
                    1,
                    columns=["instance_id", "patch"],
                ),
                {
                    "instance_id": "task-2",
                    "patch": "gold patch for task-2",
                },
            )
            with self.assertRaisesRegex(ValueError, "columns must not be empty"):
                dataset.get_row(0, columns=[])

    def test_environment_tools_declare_task_execution_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pq.write_table(
                pa.Table.from_pylist([_task("task-0", "pytest")]),
                data_dir / "data.parquet",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "DATA_DIR": str(data_dir),
                    "TASK_INDEX": str(data_dir / "missing-index.json"),
                },
            ):
                sys.modules.pop("server", None)
                server = importlib.import_module("server")

            tools = {tool.name: tool for tool in server.SWERebenchV2.list_tools().tools}
            key = server.TOOL_ROUTING_SCHEMA_KEY

            bash_schema = tools["bash"].input_schema
            self.assertIsNotNone(bash_schema)
            bash_routing = cast(
                dict[str, Any],
                bash_schema[key],  # type: ignore[index]
            )
            self.assertEqual(bash_routing["version"], 1)
            self.assertEqual(bash_routing["execution_domain"], "task")
            self.assertEqual(bash_routing["invocation"], {"kind": "direct"})
            self.assertEqual(
                set(bash_routing["capabilities"]),
                {
                    "filesystem.read",
                    "filesystem.write",
                    "system.execute",
                    "python.execute",
                },
            )
            expected_capabilities = {
                "view": ["filesystem.read"],
                "str_replace": ["filesystem.read", "filesystem.write"],
                "create_file": ["filesystem.write"],
                "submit_answer": ["task.submit"],
            }
            for tool_name, expected in expected_capabilities.items():
                schema = tools[tool_name].input_schema
                self.assertIsNotNone(schema)
                routing = cast(
                    dict[str, Any],
                    schema[key],  # type: ignore[index]
                )
                self.assertEqual(routing["capabilities"], expected)

    def test_reports_out_of_range_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pq.write_table(
                pa.Table.from_pylist([_task("task-0", "pytest")]),
                data_dir / "data.parquet",
            )
            dataset = TaskDataset(data_dir)
            with self.assertRaises(IndexError):
                dataset.get_task(1)


class SandboxHelpersTest(unittest.TestCase):
    def test_prime_verified_image_maps_to_upstream_oci_reference(self) -> None:
        prime_image = "prime/primeintellect/elastic-synthetics:316-f52f0bf"
        upstream_image = "docker.io/swerebenchv2/elastic-synthetics:316-f52f0bf"
        self.assertEqual(local_image_reference(prime_image), upstream_image)
        self.assertEqual(
            _enroot_import_uri(prime_image),
            "docker://swerebenchv2/elastic-synthetics:316-f52f0bf",
        )
        self.assertEqual(
            EnrootSandbox(prime_image).image,
            upstream_image,
        )

    def test_enroot_registry_uri_conversion(self) -> None:
        self.assertEqual(
            _enroot_import_uri("docker.io/swerebenchv2/repo:tag"),
            "docker://swerebenchv2/repo:tag",
        )
        self.assertEqual(
            _enroot_import_uri("registry-1.docker.io/library/ubuntu:22.04"),
            "docker://library/ubuntu:22.04",
        )
        self.assertEqual(
            _enroot_import_uri("docker://docker.io#swerebenchv2/repo:tag"),
            "docker://swerebenchv2/repo:tag",
        )
        self.assertEqual(
            _enroot_import_uri("docker://docker.io/swerebenchv2/repo:tag"),
            "docker://swerebenchv2/repo:tag",
        )
        self.assertEqual(
            _enroot_import_uri("ubuntu:22.04"),
            "docker://ubuntu:22.04",
        )
        self.assertEqual(
            _enroot_import_uri("ghcr.io/team/image:tag"),
            "docker://ghcr.io#team/image:tag",
        )
        self.assertEqual(
            _enroot_import_uri("localhost:5000/team/image:tag"),
            "docker://localhost:5000#team/image:tag",
        )
        self.assertEqual(
            _enroot_import_uri("docker://ghcr.io#team/image:tag"),
            "docker://ghcr.io#team/image:tag",
        )

    def test_local_result_supports_hosted_result_call_pattern(self) -> None:
        result = LocalRunResult("ok", 0)
        output, return_code = result
        self.assertEqual((output, return_code), ("ok", 0))
        self.assertEqual(result.exit_code, 0)

    @unittest.skipUnless(shutil.which("tar"), "requires GNU tar")
    def test_enroot_import_delays_read_only_directory_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            readonly = source / "readonly"
            output = root / "output"
            readonly.mkdir(parents=True)
            output.mkdir()
            (readonly / "first").write_text("first")
            (readonly / "last").write_text("last")
            (source / "interleaved").write_text("interleaved")
            readonly.chmod(0o555)
            archive = root / "layer.tar"

            try:
                subprocess.run(
                    [
                        "tar",
                        "-C",
                        str(source),
                        "--no-recursion",
                        "-cf",
                        str(archive),
                        "readonly",
                        "readonly/first",
                        "interleaved",
                        "readonly/last",
                    ],
                    check=True,
                )
                subprocess.run(
                    ["tar", "-C", str(output), "-pxf", str(archive)],
                    check=True,
                    env=_enroot_import_environment(),
                )
                self.assertEqual(
                    (output / "readonly" / "last").read_text(),
                    "last",
                )
                self.assertEqual(
                    (output / "readonly").stat().st_mode & 0o777,
                    0o555,
                )
            finally:
                # TemporaryDirectory cannot remove children of a read-only
                # directory even when the current user owns that directory.
                readonly.chmod(0o755)
                extracted = output / "readonly"
                if extracted.exists():
                    extracted.chmod(0o755)

    def test_enroot_import_retries_and_sets_tar_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls"
            enroot = bin_dir / "enroot"
            enroot.write_text(
                """#!/bin/sh
set -eu
test "$1" = import
case " ${TAR_OPTIONS-} " in
  *" --delay-directory-restore "*) ;;
  *) printf 'missing delayed directory restore'; exit 4 ;;
esac
count=0
test ! -f "$CALLS_FILE" || count=$(cat "$CALLS_FILE")
count=$((count + 1))
printf '%s' "$count" > "$CALLS_FILE"
if [ "$count" -lt 2 ]; then
  printf 'transient registry failure'
  exit 5
fi
shift
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then
    printf 'fake sqsh' > "$2"
    exit 0
  fi
  shift
done
exit 2
"""
            )
            enroot.chmod(0o755)
            unsquashfs = bin_dir / "unsquashfs"
            unsquashfs.write_text(
                """#!/bin/sh
set -eu
[ "$1" = "-cat" ] || exit 2
case "$3" in
  etc/rc)
    printf '%s\n' \\
      '. /usr/local/cargo/env' \\
      'mkdir -p "/workspace" 2> /dev/null' \\
      'cd "/workspace" && unset OLDPWD || exit 1' \\
      'exec stale-entrypoint "$@"'
    ;;
  usr/local/cargo/env)
    printf 'cat: no matches for /usr/local/cargo/env\n' >&2
    exit 2
    ;;
  *) exit 2 ;;
esac
"""
            )
            unsquashfs.chmod(0o755)

            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "CALLS_FILE": str(calls),
                "SWE_ENROOT_IMAGE_CACHE": str(root / "images"),
                "SWE_ENROOT_IMPORT_ATTEMPTS": "2",
                "SWE_ENROOT_IMPORT_RETRY_DELAY_SECONDS": "0",
                "TAR_OPTIONS": "--warning=no-unknown-keyword",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                sandbox = EnrootSandbox("docker.io/example/task:latest")
                image_path = sandbox._ensure_image_sync()

            self.assertEqual(calls.read_text(), "2")
            self.assertEqual(image_path.read_text(), "fake sqsh")

    def test_enroot_backend_lifecycle_with_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            enroot = bin_dir / "enroot"
            enroot.write_text(
                """#!/bin/sh
set -eu
case "$1" in
  import)
    shift
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output" ]; then
        printf 'fake sqsh' > "$2"
        exit 0
      fi
      shift
    done
    exit 2
    ;;
  create|remove)
    exit 0
    ;;
  start)
    if [ "${ENROOT_MOUNT_HOME+x}" != x ] || [ -n "$ENROOT_MOUNT_HOME" ]; then
      printf 'unsafe ENROOT_MOUNT_HOME=%s' "${ENROOT_MOUNT_HOME-unset}"
      exit 3
    fi
    all_args="$*"
    found_rc=0
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--rc" ]; then
        [ -f "$2" ] || exit 4
        ! grep -q '/usr/local/cargo/env' "$2" || exit 5
        grep -Fq 'exec stale-entrypoint "$@"' "$2" || exit 5
        found_rc=1
        shift 2
        continue
      fi
      shift
    done
    [ "$found_rc" -eq 1 ] || {
      # Reproduce the stale OCI entrypoint this regression guards against.
      printf '/etc/rc: 1: .: cannot open /usr/local/cargo/env: No such file'
      exit 6
    }
    case "$all_args" in
      *"cat /etc/rc"*)
        printf '%s\n' \
          '. /usr/local/cargo/env' \
          'mkdir -p "/workspace" 2> /dev/null' \
          'cd "/workspace" && unset OLDPWD || exit 1' \
          'exec stale-entrypoint "$@"'
        exit 0
        ;;
    esac
    printf 'fake command output'
    exit 0
    ;;
esac
exit 2
"""
            )
            enroot.chmod(0o755)
            unsquashfs = bin_dir / "unsquashfs"
            unsquashfs.write_text(
                """#!/bin/sh
set -eu
[ "$1" = "-cat" ] || exit 2
case "$3" in
  etc/rc)
    printf '%s\n' \\
      '. /usr/local/cargo/env' \\
      'mkdir -p "/workspace" 2> /dev/null' \\
      'cd "/workspace" && unset OLDPWD || exit 1' \\
      'exec stale-entrypoint "$@"'
    ;;
  usr/local/cargo/env)
    printf 'cat: no matches for /usr/local/cargo/env\n' >&2
    exit 2
    ;;
  *) exit 2 ;;
esac
"""
            )
            unsquashfs.chmod(0o755)

            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "ENROOT_MOUNT_HOME": str(root / "host-home"),
                "SWE_ENROOT_IMAGE_CACHE": str(root / "images"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                sandbox = EnrootSandbox("docker.io/example/task:latest")
                expected_rc = b"""mkdir -p "/workspace" 2> /dev/null
cd "/workspace" && unset OLDPWD || exit 1
exec stale-entrypoint "$@"
"""

                async def exercise() -> None:
                    await sandbox.start()
                    session_tmp = sandbox.session_tmp_dir
                    self.assertIsNotNone(session_tmp)
                    assert session_tmp is not None
                    self.assertTrue(session_tmp.is_dir())
                    rc_local_path = session_tmp / ".openreward-enroot-rc.local"
                    self.assertEqual(rc_local_path.read_text(), "")
                    self.assertEqual(
                        rc_local_path.stat().st_mode & 0o777,
                        0o600,
                    )
                    control_dir = sandbox.enroot_control_dir
                    self.assertIsNotNone(control_dir)
                    assert control_dir is not None
                    self.assertNotEqual(control_dir, session_tmp)
                    self.assertNotIn(session_tmp, control_dir.parents)
                    rc_path = control_dir / "rc"
                    self.assertEqual(rc_path.read_bytes(), expected_rc)
                    self.assertEqual(rc_path.stat().st_mode & 0o777, 0o700)
                    runtime_path = control_dir / "runtime"
                    self.assertTrue(runtime_path.is_dir())
                    # The container sees session_tmp as /tmp; a task-created
                    # lookalike there cannot modify the unmounted control rc.
                    (session_tmp / "rc").write_text("task mutation")
                    self.assertEqual(rc_path.read_bytes(), expected_rc)
                    self.assertEqual(sandbox.workdir, "/workspace")
                    start_args = sandbox._start_args("echo hello")
                    self.assertIn("--rc", start_args)
                    self.assertEqual(
                        start_args[start_args.index("--rc") + 1], str(rc_path)
                    )
                    self.assertNotIn("-lc", start_args)
                    self.assertEqual(start_args[-3:-1], ["/bin/sh", "-c"])
                    self.assertIn(
                        f"{rc_local_path}:/etc/rc.local",
                        start_args,
                    )
                    self.assertIn(
                        f"{session_tmp}:/tmp",
                        start_args,
                    )
                    mount_sources = [
                        start_args[index + 1].split(":", 1)[0]
                        for index, value in enumerate(start_args)
                        if value == "--mount"
                    ]
                    self.assertNotIn(str(control_dir), mount_sources)
                    self.assertTrue(
                        all(
                            not Path(source).is_relative_to(control_dir)
                            for source in mount_sources
                        )
                    )
                    self.assertEqual(
                        start_args[-1],
                        "export GIT_CONFIG_GLOBAL=/tmp/.gitconfig-openreward; "
                        "mkdir -p -- /workspace 2>/dev/null && "
                        "cd /workspace && echo hello",
                    )
                    result = await sandbox.run("echo hello")
                    self.assertEqual(result.return_code, 0)
                    self.assertEqual(result.output, "fake command output")
                    await sandbox.stop()
                    self.assertIsNone(sandbox.session_tmp_dir)
                    self.assertIsNone(sandbox.enroot_control_dir)
                    self.assertFalse(session_tmp.exists())
                    self.assertFalse(control_dir.exists())

                with mock.patch(
                    "sandbox_backends.asyncio.to_thread",
                    side_effect=_inline_to_thread,
                ):
                    asyncio.run(exercise())
                self.assertEqual(
                    len(list((root / "images").glob("*.sqsh"))),
                    1,
                )

    def test_enroot_workdir_parser_ignores_stale_entrypoint(self) -> None:
        script = """\
. /usr/local/cargo/env
mkdir -p "/path with spaces" 2> /dev/null
cd "/path with spaces" && unset OLDPWD || exit 1
exec stale-entrypoint "$@"
"""
        self.assertEqual(
            EnrootSandbox._workdir_from_enroot_rc(script),
            "/path with spaces",
        )

    def test_selected_swe_rc_classes_are_preserved_byte_for_byte(self) -> None:
        # The corresponding immutable rc.local files in the selected SWE-100
        # corpus were audited as comments-only. The existing empty rc.local
        # mount is therefore equivalent for these eight observed rc classes;
        # this assertion covers /etc/rc itself, including every wrapper,
        # entrypoint, and trailing newline.
        self.assertEqual(len(_SWE_RC_CLASS_FIXTURES), 8)
        for name, script in _SWE_RC_CLASS_FIXTURES.items():
            with self.subTest(name=name):
                self.assertEqual(
                    EnrootSandbox._sanitize_enroot_rc(
                        script,
                        cargo_env_exists=None,
                    ),
                    script,
                )
                EnrootSandbox._workdir_from_enroot_rc(script.decode())

    def test_missing_cargo_source_is_removed_without_other_byte_changes(self) -> None:
        script = (
            b"\t.  /usr/local/cargo/env \t\n"
            b"# preserved header\r\n"
            b'mkdir -p "/workspace" 2> /dev/null\r\n'
            b'cd "/workspace" && unset OLDPWD || exit 1\r\n'
            b'exec preserved-entrypoint "$@"'
        )
        expected = (
            b"# preserved header\r\n"
            b'mkdir -p "/workspace" 2> /dev/null\r\n'
            b'cd "/workspace" && unset OLDPWD || exit 1\r\n'
            b'exec preserved-entrypoint "$@"'
        )
        self.assertEqual(
            EnrootSandbox._sanitize_enroot_rc(
                script,
                cargo_env_exists=False,
            ),
            expected,
        )
        self.assertEqual(
            EnrootSandbox._sanitize_enroot_rc(
                b"source /usr/local/cargo/env",
                cargo_env_exists=False,
            ),
            b"",
        )

    def test_existing_cargo_source_is_preserved_byte_for_byte(self) -> None:
        script = (
            b"source /usr/local/cargo/env\n"
            b'mkdir -p "/workspace" 2> /dev/null\n'
            b'cd "/workspace" && unset OLDPWD || exit 1\n'
            b'exec preserved-entrypoint "$@"\n'
        )
        self.assertEqual(
            EnrootSandbox._sanitize_enroot_rc(
                script,
                cargo_env_exists=True,
            ),
            script,
        )

    def test_cargo_source_sanitizer_rejects_ambiguous_inputs(self) -> None:
        invalid_scripts = [
            b'. "/usr/local/cargo/env"\n',
            b". /usr/local/cargo/env || true\n",
            b"echo /usr/local/cargo/env\n",
            b". /usr/local/cargo/env\n. /usr/local/cargo/env\n",
            b"# /usr/local/cargo/env\n",
            b". /usr/local/cargo/env\r\n",
            b". /usr/local/cargo/env\r",
            b".\v/usr/local/cargo/env\n",
            b".\f/usr/local/cargo/env\n",
            b".\r/usr/local/cargo/env\n",
            b"echo\r. /usr/local/cargo/env\n",
            b"echo\v. /usr/local/cargo/env\n",
            b"echo\f. /usr/local/cargo/env\n",
            b"# metadata\n. /usr/local/cargo/env\n",
            b"echo \\\n. /usr/local/cargo/env\n",
            b"cat <<EOF\n. /usr/local/cargo/env\nEOF\n",
        ]
        for script in invalid_scripts:
            with self.subTest(script=script):
                with self.assertRaises(RuntimeError):
                    EnrootSandbox._sanitize_enroot_rc(
                        script,
                        cargo_env_exists=False,
                    )
        with self.assertRaisesRegex(RuntimeError, "valid UTF-8"):
            EnrootSandbox._sanitize_enroot_rc(
                b"# \xff\n",
                cargo_env_exists=None,
            )

    def test_cargo_presence_probe_is_fail_closed(self) -> None:
        image = Path("immutable.sqsh")
        cases = [
            (0, b"cargo bytes", b"", True),
            (
                2,
                b"",
                b"cat: no matches for /usr/local/cargo/env\n",
                False,
            ),
            (
                2,
                b"",
                b"cat: no matches for /usr/local/cargo\n",
                False,
            ),
        ]
        for returncode, stdout, stderr, expected in cases:
            with self.subTest(returncode=returncode):
                completed = subprocess.CompletedProcess(
                    args=["unsquashfs"],
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
                with mock.patch(
                    "sandbox_backends.subprocess.run",
                    return_value=completed,
                ):
                    self.assertEqual(
                        EnrootSandbox._cargo_env_exists_sync(
                            "unsquashfs",
                            image,
                            timeout=1,
                        ),
                        expected,
                    )
        completed = subprocess.CompletedProcess(
            args=["unsquashfs"],
            returncode=1,
            stdout=b"",
            stderr=b"corrupt image\n",
        )
        with mock.patch(
            "sandbox_backends.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(RuntimeError, "corrupt image"):
                EnrootSandbox._cargo_env_exists_sync(
                    "unsquashfs",
                    image,
                    timeout=1,
                )

    def test_sanitized_rc_executes_preserved_entrypoint_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            marker = root / "wrapper-ran"
            wrapper = bin_dir / "preserved-entrypoint"
            wrapper.write_text(
                '#!/bin/sh\nprintf wrapped > "$WRAPPER_MARKER"\nexec "$@"\n'
            )
            wrapper.chmod(0o755)
            workdir = root / "workdir"
            script = (
                b". /usr/local/cargo/env\n"
                + f'mkdir -p "{workdir}" 2> /dev/null\n'.encode()
                + f'cd "{workdir}" && unset OLDPWD || exit 1\n'.encode()
                + b'exec preserved-entrypoint "$@"\n'
            )
            rc_path = root / "rc"
            rc_path.write_bytes(
                EnrootSandbox._sanitize_enroot_rc(
                    script,
                    cargo_env_exists=False,
                )
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                    "WRAPPER_MARKER": str(marker),
                }
            )
            result = subprocess.run(
                [str(shutil.which("sh")), str(rc_path), "sh", "-c", "printf payload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "payload")
            self.assertEqual(marker.read_text(), "wrapped")
            self.assertTrue(workdir.is_dir())

    def test_enroot_workdir_parser_fails_closed(self) -> None:
        invalid_scripts = [
            "exec stale-entrypoint\n",
            (
                'cd "/one" && unset OLDPWD || exit 1\n'
                'cd "/two" && unset OLDPWD || exit 1\n'
            ),
            'cd "relative" && unset OLDPWD || exit 1\n',
            'cd "/workspace" && unset OLDPWD || exit 1; echo trailing\n',
            'cd "unterminated && unset OLDPWD || exit 1\n',
        ]
        for script in invalid_scripts:
            with self.subTest(script=script):
                with self.assertRaises(RuntimeError):
                    EnrootSandbox._workdir_from_enroot_rc(script)

    def test_enroot_workdir_inspection_requires_unsquashfs(self) -> None:
        sandbox = EnrootSandbox("docker.io/example/task:latest")
        with mock.patch(
            "sandbox_backends.shutil.which",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires unsquashfs"):
                sandbox._image_runtime_config_sync(Path("unused.sqsh"))

    def test_concurrent_sandboxes_use_distinct_unmounted_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandboxes = [
                EnrootSandbox("docker.io/example/one:latest"),
                EnrootSandbox("docker.io/example/two:latest"),
            ]
            for index, sandbox in enumerate(sandboxes):
                sandbox.session_tmp_dir = root / f"mounted-{index}"
                sandbox.session_tmp_dir.mkdir()
                sandbox.enroot_control_dir = root / f"control-{index}"
                sandbox.enroot_control_dir.mkdir(mode=0o700)

            # Both live sandboxes prepare their control state before either is
            # torn down; no executor thread is needed to prove path isolation.
            environments = [sandbox._enroot_environment() for sandbox in sandboxes]
            runtime_paths = [
                Path(environment["ENROOT_RUNTIME_PATH"]) for environment in environments
            ]
            self.assertEqual(len(set(runtime_paths)), 2)
            for sandbox, runtime_path in zip(sandboxes, runtime_paths):
                assert sandbox.enroot_control_dir is not None
                assert sandbox.session_tmp_dir is not None
                self.assertTrue(runtime_path.is_relative_to(sandbox.enroot_control_dir))
                self.assertFalse(runtime_path.is_relative_to(sandbox.session_tmp_dir))

    def test_inspection_failure_cleans_unmounted_control_dirs(self) -> None:
        sandbox = EnrootSandbox("docker.io/example/task:inspect-failure")

        async def exercise() -> None:
            with (
                mock.patch(
                    "sandbox_backends.shutil.which",
                    return_value="/usr/bin/enroot",
                ),
                mock.patch.object(
                    sandbox,
                    "_ensure_image_sync",
                    return_value=Path("cached.sqsh"),
                ),
                mock.patch.object(
                    sandbox,
                    "_image_runtime_config_sync",
                    side_effect=RuntimeError("inspect failed"),
                ),
                mock.patch(
                    "sandbox_backends.asyncio.to_thread",
                    side_effect=_inline_to_thread,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "inspect failed"):
                    await sandbox.start()

        asyncio.run(exercise())
        self.assertIsNone(sandbox.session_tmp_dir)
        self.assertIsNone(sandbox.enroot_control_dir)
        self.assertIsNone(sandbox.namespace_control_dir)
        self.assertFalse(sandbox._container_created)

    def test_missing_image_workdir_is_created_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = EnrootSandbox("docker.io/example/task:latest")
            sandbox.session_tmp_dir = root / "mounted-tmp"
            sandbox.session_tmp_dir.mkdir()
            sandbox.enroot_control_dir = root / "control"
            sandbox.enroot_control_dir.mkdir(mode=0o700)
            (sandbox.enroot_control_dir / "rc").write_text('#!/bin/sh\nexec "$@"\n')
            missing = root / "missing workdir"
            sandbox.workdir = str(missing)
            command = sandbox._start_args("printf swe-workdir-ok")[-1]
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "swe-workdir-ok")
            self.assertTrue(missing.is_dir())


if __name__ == "__main__":
    unittest.main()
