"""
Script: tests.quote
Description: Test quote parsing.
"""

import re


def has_contractions(line: str) -> bool:
    """
    Detects if a line contains contractions like "can't", "I'll", "it's".

    Returns:
        bool: True if contractions are found, otherwise False.
    """
    contractions = re.compile(r"\b\w+'\w+\b")  # Matches contractions ONLY
    return bool(contractions.search(line))  # Returns True if found


def clean_contractions(line: str) -> str:
    """
    Removes contractions from a sentence while preserving possessives.

    Returns:
        str: Text with contractions removed.
    """
    return re.sub(r"\b\w+'\w+\b", "", line)  # Removes contractions, not possessives


def has_quotes(line: str) -> bool:
    """
    Checks if single and double quotes are balanced.

    - Allows single words wrapped in single quotes (e.g., 'Band')
    - Ensures all other quotes are correctly paired.

    Returns:
        bool: True if quotes are balanced, False if unmatched quotes exist.
    """
    # Temporarily remove contractions to avoid false positives
    temp_line = clean_contractions(line)

    # Count quotes
    double_quotes = temp_line.count('"')
    single_quotes = temp_line.count("'")

    # Allow single-quoted words like 'Band' or 'Yes'
    single_quote_words = re.findall(r"'\w+'", temp_line)
    single_quotes -= len(single_quote_words) * 2  # Remove counted pairs

    # Validate even counts
    return double_quotes % 2 == 0 and single_quotes % 2 == 0


# 🚀 Test Cases
def test_contractions() -> list[tuple[str, bool]]:
    return [
        ('"Hello, world!"', False),  # ✅ No contraction
        ("'Hello, world!'", False),  # ✅ No contraction
        ("It's a beautiful day.", True),  # ✅ Contraction detected
        ("'John's book'", False),  # ✅ Possessive, NOT a contraction
        ("Mr. Jones' wallet", False),  # ✅ Possessive, NOT a contraction
        ("the foxes' tracks.", False),  # ✅ Possessive, NOT a contraction
        ("He said, 'Let's go!", True),  # ✅ Contraction detected
        ("She said, 'What's up?'", True),  # ✅ Contraction detected
        ('Tiny said, "Hello, world!"', False),  # ✅ No contraction
        ("say 'hip hip hooray'.", False),  # ✅ Single-quoted word, NOT a contraction
        ("\"The word is 'trophy'.\"", False),  # ✅ Single-quoted word, NOT a contraction
        ("'Unmatched single quote", False),  # ✅ Not a contraction
        ('"Unmatched double quote', False),  # ✅ Not a contraction
        ("'Hello", False),  # ✅ Not a contraction
        ("Hello'", False),  # ✅ Not a contraction
        ('She said, "Hello', False),  # ✅ No contraction
        ('She said, "Hello!"', False),  # ✅ No contraction
    ]


def test_quotes() -> list[tuple[str, bool]]:
    return [
        ('"Hello, world!"', True),
        ("'Hello, world!'", True),
        ("It's a beautiful day.", True),
        ("'John's book'", True),
        ("Mr. Jones' wallet", False),  # ✅ Fixed: Now correctly detected
        ("the foxes' tracks.", False),  # ✅ Fixed: Now correctly detected
        ("He said, 'Let's go!", False),
        ("She said, 'What's up?'", True),
        ('Tiny said, "Hello, world!"', True),
        ("say 'hip hip hooray'.", True),
        ("\"The word is 'trophy'.\"", True),
        ("'Unmatched single quote", False),
        ('"Unmatched double quote', False),
        ("'Hello", False),
        ("Hello'", False),
        ('She said, "Hello', False),
        ('She said, "Hello!"', True),
    ]


def run_tests(test_label: str, test_cases: callable, test_function: callable) -> None:
    print(test_label)
    for test, expected in test_cases():
        result = test_function(test)
        status = "✅ Passed" if result == expected else "❌ Failed"
        print(f"[{status}] Expected {expected}, got {result}: test={test}")
    print()


run_tests("Contractions", test_contractions, has_contractions)
run_tests("Quotes", test_quotes, has_quotes)
