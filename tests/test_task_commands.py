import unittest

from task_commands import normalize_test_commands, build_test_command_script


class TestCommandTest(unittest.TestCase):
    def test_accepts_string_and_list_commands(self) -> None:
        self.assertEqual(normalize_test_commands("pytest -q"), ["pytest -q"])
        self.assertEqual(
            normalize_test_commands(["make unit", "make integration"]),
            ["make unit", "make integration"],
        )
        self.assertEqual(
            build_test_command_script(["make unit", "make integration"]),
            "set -e\nmake unit\nmake integration",
        )

    def test_rejects_missing_or_empty_commands(self) -> None:
        for value in (None, "", [], ["", "  "]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_test_commands(value)


if __name__ == "__main__":
    unittest.main()
