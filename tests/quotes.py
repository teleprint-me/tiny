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
    contractions = re.compile(r"\b\w+'\w+\b")  # Matches contractions
    return bool(contractions.search(line))  # Returns True if found


def has_quotes(line: str) -> bool:
    """
    Checks if single and double quotes are balanced.

    - Allows single words wrapped in single quotes (e.g., 'Band')
    - Ensures all other quotes are correctly paired.

    Returns:
        bool: True if quotes are balanced, False if unmatched quotes exist.
    """
    # Count quotes
    double_quotes = line.count('"')
    single_quotes = line.count("'")

    # Allow single-quoted words like 'Band' or 'Yes'
    single_quote_words = re.findall(r"'\w+'", line)
    single_quotes -= len(single_quote_words) * 2  # Remove counted pairs

    # Validate even counts
    return double_quotes % 2 == 0 and single_quotes % 2 == 0


def validate_quotes(line: str) -> bool:
    """
    Validates whether a line contains properly matched quotes.

    - Allows common contractions.
    - Handles single-quoted words.
    - Ensures quotes are properly paired.

    Returns:
        bool: True if the text is valid, False if unmatched quotes are detected.
    """
    # Temporarily remove contractions to avoid false positives
    temp_line = re.sub(r"\b\w+'\w+\b", "", line)  # Removes contractions safely

    return has_quotes(temp_line)


# 🚀 Test Cases
def test_cases() -> list[tuple[str, bool]]:
    return [
        ('"Hello, world!"', True),  # ✅ Balanced double quotes
        ("'Hello, world!'", True),  # ✅ Balanced single quotes
        ("It's a beautiful day.", True),  # ✅ Valid contraction
        ("'John's book'", True),  # ✅ Correct possessive case
        ("Mr. Jones' wallet", True),  # ✅ Possessive apostrophe
        ("the foxes' tracks.", True),  # ✅ Plural possessive
        ("He said, 'Let's go!", False),  # ❌ Unmatched single quote
        ("She said, 'What's up?'", True),  # ✅ Contraction is valid
        ('Tiny said, "Hello, world!"', True),  # ✅ Balanced
        ("say 'hip hip hooray'.", True),  # ✅ Single-quoted phrase is fine
        ("\"The word is 'trophy'.\"", True),  # ✅ Nested single within double
        ("'Unmatched single quote", False),  # ❌ Missing closing
        ('"Unmatched double quote', False),  # ❌ Missing closing
        ("'Hello", False),  # ❌ Unmatched single
        ("Hello'", False),  # ❌ Unmatched single
        ('She said, "Hello', False),  # ❌ Unmatched double
        ('She said, "Hello!"', True),  # ✅ Matched properly
    ]


# Run Tests
for test, expected in test_cases():
    result = validate_quotes(test)
    status = "✅" if result == expected else "❌"
    print(f"Expected {expected}, got {result} -> {status} | {test}")
