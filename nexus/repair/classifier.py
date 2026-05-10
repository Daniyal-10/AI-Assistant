"""
nexus/repair/classifier.py
──────────────────────────
Error classification and fix strategy instructions.
Extracted from nexus/ai/prompts.py — logic is identical.
"""


def classify_error(stderr: str, stdout: str) -> str:
    """
    Classify the failure type from stderr/stdout into a fix strategy category.
    Returns a string tag used to inject targeted instructions into the fix prompt.
    """
    combined = (stderr + stdout).lower()

    if "modulenotfounderror" in combined or "no module named" in combined:
        return "MODULE_NOT_FOUND"
    if "importerror" in combined:
        return "IMPORT_ERROR"
    if "syntaxerror" in combined or "was never closed" in combined:
        return "SYNTAX_ERROR"
    if "assertionerror" in combined or "assert" in combined:
        return "ASSERTION_ERROR"
    if "filenotfounderror" in combined or "no such file" in combined:
        return "FILE_NOT_FOUND"
    if (
        "socket" in combined
        or "getaddrinfo" in combined
        or "connectionrefused" in combined
        or "network" in combined
        or "urlopen error" in combined
    ):
        return "NETWORK_IN_TEST"
    if "typeerror" in combined:
        return "TYPE_ERROR"
    if "nameerror" in combined:
        return "NAME_ERROR"
    return "UNKNOWN"


_FIX_STRATEGY_INSTRUCTIONS = {
    "MODULE_NOT_FOUND": (
        "STRATEGY: MODULE_NOT_FOUND\n"
        "The import is failing because a third-party module is not installed.\n"
        "ACTION: Rewrite the code to use Python stdlib equivalents:\n"
        "  - pandas -> csv, pathlib, collections\n"
        "  - requests -> urllib.request\n"
        "  - numpy -> math, statistics\n"
        "Only add the module to requirements.txt if stdlib CANNOT replace it."
    ),
    "IMPORT_ERROR": (
        "STRATEGY: IMPORT_ERROR\n"
        "A local module import is failing.\n"
        "ACTION: Fix the import path. Check that the module file exists "
        "in the workspace and that the import statement matches the filename exactly.\n"
        "Do NOT change module logic — only fix the import statement."
    ),
    "SYNTAX_ERROR": (
        "STRATEGY: SYNTAX_ERROR\n"
        "The file has a syntax error and cannot be parsed.\n"
        "ACTION: Fix ONLY the syntax error on the reported line. "
        "Do not refactor or restructure the file."
    ),
    "ASSERTION_ERROR": (
        "STRATEGY: ASSERTION_ERROR\n"
        "A test assertion is failing — either the logic is wrong or the test expectation is wrong.\n"
        "ACTION:\n"
        "  1. Read the STDOUT carefully — it shows the actual vs expected values.\n"
        "  2. If the source logic is correct, fix the TEST assertion to match reality.\n"
        "  3. If the source logic is wrong, fix the SOURCE CODE.\n"
        "  4. Do NOT mock away the logic being tested.\n"
        "  EXAMPLE: If test asserts factorial(5) == 600 but actual is 120, "
        "fix the test to assert == 120."
    ),
    "FILE_NOT_FOUND": (
        "STRATEGY: FILE_NOT_FOUND\n"
        "A required file is missing from the workspace.\n"
        "ACTION: Generate the missing file with appropriate sample content. "
        "Include it in fixed_files."
    ),
    "NETWORK_IN_TEST": (
        "STRATEGY: NETWORK_IN_TEST\n"
        "A test is making a real network call which is forbidden in the sandbox.\n"
        "ACTION: Fix the TEST file to mock the network call using unittest.mock.patch.\n"
        "Do NOT change the implementation logic.\n"
        "Use this pattern:\n"
        "  with patch('module.urllib.request.urlopen') as mock_url:\n"
        "      mock_url.return_value.__enter__ = lambda s: s\n"
        "      mock_url.return_value.read.return_value = b'expected_response'\n"
        "      result = function_under_test()\n"
        "      assert result == expected_value"
    ),
    "TYPE_ERROR": (
        "STRATEGY: TYPE_ERROR\n"
        "A function received the wrong type.\n"
        "ACTION: Fix the type mismatch. Check the function signature and "
        "the call site. Add type conversion if needed."
    ),
    "NAME_ERROR": (
        "STRATEGY: NAME_ERROR\n"
        "A variable or function name is referenced before definition.\n"
        "ACTION: Define the variable or import the function before its first use."
    ),
    "UNKNOWN": (
        "STRATEGY: UNKNOWN\n"
        "Read STDERR carefully and apply the most targeted fix possible.\n"
        "Do NOT change code that is unrelated to the error."
    ),
}


def get_strategy_instruction(error_category: str) -> str:
    """Return the fix strategy instruction string for a given error category."""
    return _FIX_STRATEGY_INSTRUCTIONS.get(
        error_category,
        _FIX_STRATEGY_INSTRUCTIONS["UNKNOWN"],
    )
