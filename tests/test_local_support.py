import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from build_index import build_index
from dataset_store import TaskDataset
from sandbox_backends import EnrootSandbox, LocalRunResult, _enroot_import_uri


def _task(instance_id: str, test_cmd: str) -> dict:
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "base_commit": "abc123",
        "test_patch": "",
        "problem_statement": "Fix the bug",
        "image_name": "docker.io/library/alpine:3.20",
        "language": "python",
        "FAIL_TO_PASS": ["test_bug"],
        "PASS_TO_PASS": ["test_existing"],
        "install_config": {
            "test_cmd": test_cmd,
            "log_parser": "parse_pytest",
            "install": "",
            "base_image_name": "",
        },
    }


class TaskDatasetTest(unittest.TestCase):
    def test_reads_shards_lazily_and_applies_valid_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            rows = [_task("task-0", "pytest"), _task("task-1", "")]
            pq.write_table(
                pa.Table.from_pylist(rows),
                data_dir / "part-0.parquet",
                row_group_size=1,
            )
            pq.write_table(
                pa.Table.from_pylist([_task("task-2", "go test ./...")]),
                data_dir / "part-1.parquet",
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
    def test_enroot_registry_uri_conversion(self) -> None:
        self.assertEqual(
            _enroot_import_uri("docker.io/swerebenchv2/repo:tag"),
            "docker://docker.io#swerebenchv2/repo:tag",
        )
        self.assertEqual(
            _enroot_import_uri("ubuntu:22.04"),
            "docker://ubuntu:22.04",
        )
        self.assertEqual(
            _enroot_import_uri("docker://registry.example#team/image:tag"),
            "docker://registry.example#team/image:tag",
        )

    def test_local_result_supports_hosted_result_call_pattern(self) -> None:
        result = LocalRunResult("ok", 0)
        output, return_code = result
        self.assertEqual((output, return_code), ("ok", 0))
        self.assertEqual(result.exit_code, 0)

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
    printf 'fake command output'
    exit 0
    ;;
esac
exit 2
"""
            )
            enroot.chmod(0o755)

            env = {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "SWE_ENROOT_IMAGE_CACHE": str(root / "images"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                sandbox = EnrootSandbox("docker.io/example/task:latest")

                async def exercise() -> None:
                    await sandbox.start()
                    result = await sandbox.run("echo hello")
                    self.assertEqual(result.return_code, 0)
                    self.assertEqual(result.output, "fake command output")
                    await sandbox.stop()

                asyncio.run(exercise())
                self.assertEqual(
                    len(list((root / "images").glob("*.sqsh"))),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
