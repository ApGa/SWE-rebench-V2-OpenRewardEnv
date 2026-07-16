"""Normalize the string-or-list test command schema used by SWE-rebench."""

from __future__ import annotations


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
    """Build the fail-fast shell script used for submission evaluation."""
    return "set -e\n" + "\n".join(normalize_test_commands(value))
