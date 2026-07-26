import unittest

from log_parsers import TestStatus, parse_log_js_4


class JavaScriptParserTest(unittest.TestCase):
    def test_jest_multiplication_x_failure_symbol(self) -> None:
        results = parse_log_js_4(
            """
              ✓ passes with a check mark (3 ms)
              ✕ fails with Jest's multiplication x (12 ms)
            """
        )

        self.assertEqual(
            results,
            {
                "passes with a check mark": TestStatus.PASSED.value,
                "fails with Jest's multiplication x": TestStatus.FAILED.value,
            },
        )


if __name__ == "__main__":
    unittest.main()
