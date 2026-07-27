import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_store import TaskDataset
from preflight import (
    _existing_results,
    assess_phase,
    select_indices,
    stratified_sample,
    validate_metadata,
)
from sandbox_backends import EnrootSandbox


class SelectIndicesTest(unittest.TestCase):
    def test_resume_counts_latest_existing_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"index": 2, "valid": False}),
                        "not-json",
                        json.dumps({"index": 3, "valid": False}),
                        json.dumps({"index": 2, "valid": True}),
                    ]
                )
            )

            self.assertEqual(
                _existing_results(path),
                {2: True, 3: False},
            )

    def test_selects_explicit_ranges_and_shards(self) -> None:
        self.assertEqual(
            select_indices(
                20,
                index_spec="1,3-8,12",
                shard_id=1,
                num_shards=3,
                max_tasks=None,
            ),
            [1, 4, 7],
        )

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(IndexError):
            select_indices(
                3,
                index_spec="3",
                shard_id=0,
                num_shards=1,
                max_tasks=None,
            )

    def test_stratifies_and_can_require_cached_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"language": "python", "image_name": "image-python-0"},
                {"language": "python", "image_name": "image-python-1"},
                {"language": "go", "image_name": "image-go-0"},
                {"language": "go", "image_name": "image-go-1"},
            ]
            pq.write_table(pa.Table.from_pylist(rows), root / "data.parquet")
            (root / "task_index.json").write_text(
                json.dumps({"valid_indices": [0, 1, 2, 3]})
            )
            dataset = TaskDataset(root)
            image_cache = root / "images"
            image_cache.mkdir()
            for image in ("image-python-1", "image-go-0"):
                sandbox = EnrootSandbox(image)
                sandbox.image_cache = image_cache
                sandbox._cached_image_path().write_text("cached")

            selected = stratified_sample(
                dataset,
                [0, 1, 2, 3],
                per_language=1,
                seed=0,
                cached_image_dir=image_cache,
            )
            self.assertEqual(selected, [1, 2])


class AssessPhaseTest(unittest.TestCase):
    def test_base_requires_observed_f2p_failure_and_p2p_pass(self) -> None:
        result = assess_phase(
            "base",
            {
                "new_test": "FAILED",
                "old_test": "PASSED",
            },
            ["new_test"],
            ["old_test"],
            1,
        )
        self.assertTrue(result["valid"])

        # Expected statuses are authoritative. Some framework commands return
        # zero even when their parser reports the task's expected failure.
        zero_exit = assess_phase(
            "base",
            {
                "new_test": "FAILED",
                "old_test": "PASSED",
            },
            ["new_test"],
            ["old_test"],
            0,
        )
        self.assertTrue(zero_exit["valid"])

        partial_f2p = assess_phase(
            "base",
            {
                "new_test": "FAILED",
                "also_new": "PASSED",
                "old_test": "PASSED",
            },
            ["new_test", "also_new"],
            ["old_test"],
            1,
        )
        self.assertTrue(partial_f2p["valid"])
        self.assertEqual(
            partial_f2p["observed_fail_to_pass_failure_count"],
            1,
        )

        no_edit_pass = assess_phase(
            "base",
            {
                "new_test": "PASSED",
                "old_test": "PASSED",
            },
            ["new_test"],
            ["old_test"],
            0,
        )
        self.assertFalse(no_edit_pass["valid"])
        self.assertTrue(no_edit_pass["no_fail_to_pass_failure"])

        missing = assess_phase(
            "base",
            {"old_test": "PASSED"},
            ["new_test"],
            ["old_test"],
            1,
        )
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["missing"], ["new_test"])

    def test_gold_requires_expected_tests_and_ordinary_exit(self) -> None:
        passing = {
            "new_test": "PASSED",
            "old_test": "PASSED",
        }
        self.assertTrue(
            assess_phase(
                "gold",
                passing,
                ["new_test"],
                ["old_test"],
                1,
            )["valid"]
        )
        self.assertFalse(
            assess_phase(
                "gold",
                passing,
                ["new_test"],
                ["old_test"],
                2,
            )["valid"]
        )

    def test_timing_alias_shared_by_groups_is_validated_without_contradiction(
        self,
    ) -> None:
        fail_to_pass = ["case [2.40 ms]"]
        pass_to_pass = ["case [0.10 ms]", "existing [0.20 ms]"]
        base = assess_phase(
            "base",
            {
                "case": "FAILED",
                "existing": "PASSED",
            },
            fail_to_pass,
            pass_to_pass,
            1,
        )
        self.assertTrue(base["valid"])
        self.assertEqual(base["shared_expected"], ["case"])

        gold = assess_phase(
            "gold",
            {
                "case": "PASSED",
                "existing": "PASSED",
            },
            fail_to_pass,
            pass_to_pass,
            0,
        )
        self.assertTrue(gold["valid"])


class MetadataValidationTest(unittest.TestCase):
    @staticmethod
    def _row(
        fail_to_pass: list[str],
        pass_to_pass: list[str],
    ) -> dict:
        return {
            "instance_id": "phpactor__phpactor-2290",
            "base_commit": "abc",
            "image_name": "image",
            "patch": "patch",
            "test_patch": "test patch",
            "language": "php",
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "install_config": {
                "test_cmd": "phpunit",
                "log_parser": "parse_log_phpunit",
            },
        }

    def test_timing_alias_is_diagnostic_not_invalid(self) -> None:
        result = validate_metadata(
            0,
            0,
            self._row(
                ["Chain Resolver [0.11 ms]"],
                ["Chain Resolver [0.07 ms]"],
            ),
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["normalized_cross_group_alias_count"], 1)

    def test_verbatim_cross_group_overlap_is_invalid(self) -> None:
        result = validate_metadata(
            0,
            0,
            self._row(["same"], ["same"]),
        )
        self.assertFalse(result["valid"])
        self.assertIn("occur verbatim", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
