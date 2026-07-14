"""Lazy access to SWE-rebench-V2 parquet shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


TASK_COLUMNS = [
    "instance_id",
    "repo",
    "base_commit",
    "test_patch",
    "problem_statement",
    "image_name",
    "language",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "install_config",
]


@dataclass(frozen=True)
class _ParquetShard:
    path: Path
    offset: int
    num_rows: int


class TaskDataset:
    """Read individual rows without loading the full ~2 GB table into memory."""

    def __init__(
        self,
        data_dir: Path,
        *,
        index_path: Path | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.paths = self._discover_paths(data_dir)
        self.shards: list[_ParquetShard] = []

        offset = 0
        for path in self.paths:
            num_rows = pq.ParquetFile(path).metadata.num_rows
            self.shards.append(
                _ParquetShard(path=path, offset=offset, num_rows=num_rows)
            )
            offset += num_rows
        self.total_rows = offset

        if index_path is None:
            index_path = data_dir / "task_index.json"
        self.valid_indices = self._load_valid_indices(index_path)

    @staticmethod
    def _discover_paths(data_dir: Path) -> list[Path]:
        single_file = data_dir / "data.parquet"
        if single_file.exists():
            return [single_file]

        paths = sorted(data_dir.rglob("*.parquet"))
        if not paths:
            fallback = Path("data") / "data.parquet"
            if fallback.exists():
                return [fallback]
            raise FileNotFoundError(
                f"No parquet files found under {data_dir}. Download "
                "nebius/SWE-rebench-V2 and mount it at DATA_DIR."
            )
        return paths

    def _load_valid_indices(self, index_path: Path) -> list[int] | None:
        if not index_path.exists():
            return None
        payload = json.loads(index_path.read_text())
        indices = payload.get("valid_indices")
        if not isinstance(indices, list):
            raise ValueError(
                f"{index_path} does not contain a valid_indices list"
            )
        result = [int(index) for index in indices]
        if result and (min(result) < 0 or max(result) >= self.total_rows):
            raise ValueError(
                f"{index_path} contains indices outside dataset row range "
                f"0..{self.total_rows - 1}"
            )
        return result

    @property
    def num_tasks(self) -> int:
        if self.valid_indices is not None:
            return len(self.valid_indices)
        return self.total_rows

    def _raw_index(self, task_index: int) -> int:
        if task_index < 0 or task_index >= self.num_tasks:
            raise IndexError(
                f"Task index {task_index} out of range "
                f"(0..{self.num_tasks - 1})"
            )
        if self.valid_indices is None:
            return task_index
        return self.valid_indices[task_index]

    def _locate_shard(self, raw_index: int) -> tuple[_ParquetShard, int]:
        for shard in self.shards:
            if raw_index < shard.offset + shard.num_rows:
                return shard, raw_index - shard.offset
        raise IndexError(f"Raw task index {raw_index} is outside the dataset")

    def get_task(self, task_index: int) -> dict[str, Any]:
        raw_index = self._raw_index(task_index)
        shard, file_index = self._locate_shard(raw_index)
        parquet_file = pq.ParquetFile(shard.path)

        row_group_offset = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_group_rows = parquet_file.metadata.row_group(
                row_group_index
            ).num_rows
            if file_index < row_group_offset + row_group_rows:
                row_offset = file_index - row_group_offset
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=TASK_COLUMNS,
                )
                row = {
                    column: table.column(column)[row_offset].as_py()
                    for column in TASK_COLUMNS
                }
                return self._normalize_row(row)
            row_group_offset += row_group_rows

        raise IndexError(
            f"Task index {task_index} could not be located in {shard.path}"
        )

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            value = row.get(key)
            if isinstance(value, str):
                row[key] = json.loads(value)
            elif value is None:
                row[key] = []

        install_config = row.get("install_config")
        if isinstance(install_config, str):
            row["install_config"] = json.loads(install_config)
        return row
