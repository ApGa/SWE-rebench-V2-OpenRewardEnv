import subprocess
import unittest

from task_commands import (
    _split_top_level_and,
    build_test_command_script,
    normalize_test_commands,
)


class TestCommandTest(unittest.TestCase):
    def test_accepts_string_and_list_commands(self) -> None:
        self.assertEqual(normalize_test_commands("pytest -q"), ["pytest -q"])
        self.assertEqual(
            normalize_test_commands(["make unit", "make integration"]),
            ["make unit", "make integration"],
        )
        self.assertEqual(
            normalize_test_commands(["make unit", "make integration"]),
            ["make unit", "make integration"],
        )

    def test_rejects_missing_or_empty_commands(self) -> None:
        for value in (None, "", [], ["", "  "]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_test_commands(value)

    def test_splits_only_top_level_and_operators(self) -> None:
        self.assertEqual(
            _split_top_level_and(
                "cd repo && run-tests && find . -name '*.xml' -exec cat {} +"
            ),
            [
                "cd repo",
                "run-tests",
                "find . -name '*.xml' -exec cat {} +",
            ],
        )
        self.assertEqual(
            _split_top_level_and(
                "sh -c 'setup && run-tests' && (cd child && run-tests) "
                '&& printf "%s" "report && details"'
            ),
            [
                "sh -c 'setup && run-tests'",
                "(cd child && run-tests)",
                'printf "%s" "report && details"',
            ],
        )

    def test_runs_collectors_and_later_commands_after_failure(self) -> None:
        script = build_test_command_script(
            [
                "printf 'suite-one\\n' && false && printf 'report-one\\n'",
                "printf 'suite-two\\n'",
            ]
        )
        result = subprocess.run(
            ["/bin/sh", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("suite-one", result.stdout)
        self.assertIn("report-one", result.stdout)
        self.assertIn("suite-two", result.stdout)
        self.assertIn(
            "[OPENREWARD_TEST_COMMAND_END index=0 stage=1 exit_code=1]",
            result.stdout,
        )

    def test_preserves_first_nonzero_status_and_shell_state(self) -> None:
        script = build_test_command_script(
            "export OPENREWARD_TEST_VALUE=preserved "
            "&& /bin/sh -c 'exit 7' "
            "&& printf '%s\\n' \"$OPENREWARD_TEST_VALUE\" "
            "&& /bin/sh -c 'exit 9'"
        )
        result = subprocess.run(
            ["/bin/sh", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 7)
        self.assertIn("preserved", result.stdout)
        self.assertIn(
            "[OPENREWARD_TEST_COMMAND_END index=0 stage=3 exit_code=9]",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
