import unittest

from log_parsers import parse_log_pytest
from scoring import score_test_results


class RewardScoringTest(unittest.TestCase):
    def test_binary_reward_requires_every_expected_test(self) -> None:
        expected = ["tests/test_app.py::test_bug", "tests/test_app.py::test_edge"]
        parsed = {
            "tests/test_app.py::test_bug": "PASSED",
            "tests/test_app.py::test_edge": "PASSED",
            "tests/test_app.py::test_existing": "PASSED",
        }
        score = score_test_results(
            parsed,
            expected,
            ["tests/test_app.py::test_existing"],
        )
        self.assertEqual(score.reward, 1.0)
        self.assertEqual(score.fail_to_pass_passed, 2)
        self.assertEqual(score.pass_to_pass_passed, 1)

        del parsed["tests/test_app.py::test_edge"]
        score = score_test_results(
            parsed,
            expected,
            ["tests/test_app.py::test_existing"],
        )
        self.assertEqual(score.reward, 0.0)
        self.assertEqual(score.fail_to_pass_passed, 1)

    def test_partial_reward_weights_groups_equally(self) -> None:
        parsed = {
            "f2p-1": "PASSED",
            "f2p-2": "FAILED",
            "p2p-1": "PASSED",
            "p2p-2": "PASSED",
            "p2p-3": "PASSED",
            "p2p-4": "FAILED",
        }
        score = score_test_results(
            parsed,
            ["f2p-1", "f2p-2"],
            ["p2p-1", "p2p-2", "p2p-3", "p2p-4"],
            reward_mode="partial",
        )
        self.assertEqual(score.reward, 0.625)

    def test_partial_reward_ignores_an_empty_group(self) -> None:
        score = score_test_results(
            {"f2p-1": "PASSED", "f2p-2": "FAILED"},
            ["f2p-1", "f2p-2"],
            [],
            reward_mode="fractional",
        )
        self.assertEqual(score.reward, 0.5)

    def test_pytest_parser_output_scores_and_missing_ids_fail(self) -> None:
        parsed = parse_log_pytest(
            "PASSED tests/test_app.py::test_bug\n"
            "FAILED tests/test_app.py::test_existing - AssertionError\n"
        )
        score = score_test_results(
            parsed,
            ["tests/test_app.py::test_bug"],
            ["tests/test_app.py::test_existing", "tests/test_app.py::not_reported"],
        )
        self.assertEqual(score.fail_to_pass_passed, 1)
        self.assertEqual(score.pass_to_pass_passed, 0)
        self.assertEqual(score.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
