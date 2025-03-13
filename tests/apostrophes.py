"""
Script: tests.apostrophes
Description: Test apostrophe parsing.
"""

import re


def has_apostrophes(line: str) -> bool:
    """
    Detects if a line contains contractions like "can't", "I'll", "it's",
    and possessives like "John's", "James'", but not incorrect cases.

    Returns:
        bool: True if contractions or possessives are found, otherwise False.
    """
    patterns = [
        r"\b\w+'\w+\b",  # ex: can't, I'll, it's
        r"|\b\w+'s\b",  # ex: John's, cat's
        r"|\b\w+'(?=\s|[.,!?])",  # ex: James', foxes', followed by space or punctuation
    ]
    apostrophes = re.compile("".join(patterns))
    return bool(apostrophes.search(line))  # Returns True if found


# 🚀 Test Cases
def test_apostrophes() -> list[tuple[str, bool]]:
    return [
        ('"Hello, world!"', False),  # ✅ No contraction
        ("'Hello, world!'", False),  # ✅ No contraction
        ("It's a beautiful day.", True),  # ✅ Contraction detected
        ("'John's book'", True),  # ✅ Possessive, NOT a contraction
        ("Mr. Jones' wallet", True),  # ✅ Possessive, NOT a contraction
        ("the foxes' tracks.", True),  # ✅ Possessive, NOT a contraction
        ("He said, 'Let's go!", True),  # ✅ Contraction detected
        ("She said, 'What's up?'", True),  # ✅ Contraction detected
        ('Tiny said, "Hello, world!"', False),  # ✅ No contraction
        ("say 'hip hip hooray'.", False),  # ✅ Not a contraction or possessive
        ("\"The word is 'trophy'.\"", False),  # ✅ Not a contraction or possessive
        ("'Unmatched single quote", False),  # ✅ Not a contraction
        ('"Unmatched double quote', False),  # ✅ Not a contraction
        ("'Hello", False),  # ✅ Not a contraction
        ("Hello'", False),  # ✅ Not a contraction
        ('She said, "Hello', False),  # ✅ No contraction
        ('She said, "Hello!"', False),  # ✅ No contraction
        ("'tis the season", True),  # ✅ Single-letter contraction should match
        ("'em for the win", True),  # ✅ Should match ('em = them)
    ]


def run_tests(test_label: str, test_cases: callable, test_function: callable) -> None:
    print(test_label)
    for test, expected in test_cases():
        result = test_function(test)
        status = "✅ Passed" if result == expected else "❌ Failed"
        print(f"[{status}] Expected {expected}, got {result}: test={test}")


run_tests("Apostrophes", test_apostrophes, has_apostrophes)
