"""Pure reward calculation for SWE-rebench test results."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from log_parsers import TestStatus, ansi_escape


# Some test runners embed durations in identifiers. These patterns match the
# official SWE-rebench V2 evaluator.
_TIMING_NORMALIZE_RES = [
    re.compile(
        r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$",
        re.IGNORECASE,
    ),
]


def normalize_test_name(name: str) -> str:
    """Strip ANSI escapes and nondeterministic timing text from a test ID."""
    name = ansi_escape(name)
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


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
    normalized_results = {
        normalize_test_name(test_id): status
        for test_id, status in test_results.items()
    }
    fail_to_pass_passed = sum(
        normalized_results.get(normalize_test_name(test_id)) == passed_status
        for test_id in fail_to_pass
    )
    pass_to_pass_passed = sum(
        normalized_results.get(normalize_test_name(test_id)) == passed_status
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
