"""Pure reward calculation for SWE-rebench test results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from log_parsers import TestStatus


@dataclass(frozen=True)
class TestScore:
    """Counts and reward produced from parsed test statuses."""

    fail_to_pass_passed: int
    pass_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_total: int
    reward: float


def score_test_results(
    test_results: Mapping[str, str],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    *,
    reward_mode: str = "binary",
) -> TestScore:
    """Score parsed statuses with binary or equally weighted partial reward."""
    passed_status = TestStatus.PASSED.value
    fail_to_pass_passed = sum(
        test_results.get(test_id) == passed_status
        for test_id in fail_to_pass
    )
    pass_to_pass_passed = sum(
        test_results.get(test_id) == passed_status
        for test_id in pass_to_pass
    )

    all_required_passed = (
        fail_to_pass_passed == len(fail_to_pass)
        and pass_to_pass_passed == len(pass_to_pass)
    )
    binary_reward = 1.0 if all_required_passed else 0.0

    if reward_mode.strip().lower() in {"partial", "fractional"}:
        group_scores: list[float] = []
        if fail_to_pass:
            group_scores.append(fail_to_pass_passed / len(fail_to_pass))
        if pass_to_pass:
            group_scores.append(pass_to_pass_passed / len(pass_to_pass))
        reward = (
            sum(group_scores) / len(group_scores)
            if group_scores
            else binary_reward
        )
    else:
        reward = binary_reward

    return TestScore(
        fail_to_pass_passed=fail_to_pass_passed,
        pass_to_pass_passed=pass_to_pass_passed,
        fail_to_pass_total=len(fail_to_pass),
        pass_to_pass_total=len(pass_to_pass),
        reward=reward,
    )
