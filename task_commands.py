"""Normalize the string-or-list test command schema used by SWE-rebench."""

from __future__ import annotations


def _split_top_level_and(command: str) -> list[str]:
    """Split a shell command at top-level ``&&`` operators.

    SWE-rebench commands commonly use ``test && cat test-report.xml``.  The
    test command is expected to return nonzero before a fix, but the report is
    still needed by the log parser.  Splitting only top-level operators keeps
    quoted commands and grouped/subshell expressions intact while allowing the
    generated runner to execute report collectors and later test suites.
    """
    segments: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0
    index = 0

    while index < len(command):
        char = command[index]

        if escaped:
            escaped = False
            index += 1
            continue

        if char == "\\":
            escaped = True
            index += 1
            continue

        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue

        if char in ("'", '"', "`"):
            quote = char
            index += 1
            continue

        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif (
            char == "&"
            and index + 1 < len(command)
            and command[index + 1] == "&"
            and not paren_depth
            and not brace_depth
            and not bracket_depth
        ):
            segment = command[start:index].strip()
            if not segment:
                return [command]
            segments.append(segment)
            index += 2
            start = index
            continue

        index += 1

    final_segment = command[start:].strip()
    if not final_segment:
        return [command]
    segments.append(final_segment)
    return segments


def normalize_test_commands(value: object) -> list[str]:
    """Return nonempty commands using the official evaluator's coercion."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("test_cmd must be a string or list of strings")

    commands = [
        command
        for command in value
        if isinstance(command, str) and command.strip()
    ]
    if not commands:
        raise ValueError("test_cmd must contain at least one nonempty command")
    return commands


def build_test_command_script(value: object) -> str:
    """Build a test script that preserves failures without failing fast.

    Every configured command and top-level ``&&`` stage runs even when an
    earlier test fails.  This is important for multi-suite commands and for
    commands whose final stage prints an XML/TRX report.  The first nonzero
    status remains the script's exit status, so a successful report collector
    cannot mask the test failure that preceded it.
    """
    lines = [
        "set +e",
        "__openreward_test_status=0",
    ]

    for command_index, command in enumerate(normalize_test_commands(value)):
        stages = _split_top_level_and(command)
        for stage_index, stage in enumerate(stages):
            lines.extend(
                [
                    (
                        "printf '%s\\n' "
                        f"'[OPENREWARD_TEST_COMMAND_START "
                        f"index={command_index} stage={stage_index}]'"
                    ),
                    stage,
                    "__openreward_stage_status=$?",
                    (
                        "printf '%s\\n' "
                        f"\"[OPENREWARD_TEST_COMMAND_END "
                        f"index={command_index} stage={stage_index} "
                        "exit_code=${__openreward_stage_status}]\""
                    ),
                    (
                        'if [ "$__openreward_stage_status" -ne 0 ] '
                        '&& [ "$__openreward_test_status" -eq 0 ]; then'
                    ),
                    "  __openreward_test_status=$__openreward_stage_status",
                    "fi",
                ]
            )

    lines.append('exit "$__openreward_test_status"')
    return "\n".join(lines)
