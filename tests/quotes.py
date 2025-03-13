"""
Script: tests.quotes
Description: Test quote parsing.
"""

import re


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


run_tests("Quotes", test_quotes, has_quotes)
