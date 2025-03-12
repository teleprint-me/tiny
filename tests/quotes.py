"""
Script: tests.quote
Description: Test quote parsing.
"""

import re

# Matches common contractions (e.g., "don't", "I'll", "it's")
CONTRACTIONS = re.compile(r"\b(?:'t|'s|'re|'m|'ve|'d|'ll|'am|'em)\b", re.IGNORECASE)


def validate_enclosed_quotes(sequence: str) -> bool:
    """
    Checks if a sequence has properly enclosed double and single quotes.

    - Ignores apostrophes used in contractions and possessives.
    - Ensures all opening quotes have closing counterparts.
    - Handles single quotes used for emphasis within double quotes.

    Returns:
        True  → Quotes are properly paired.
        False → Unmatched/missing quote detected.
    """
    quote_flag = False  # Tracks double quotes
    single_quote_flag = False  # Tracks single quotes

    i = 0
    while i < len(sequence):
        char = sequence[i]

        # Handle double quotes
        if char == '"':
            quote_flag = not quote_flag  # Toggle open/close state

        # Handle single quotes (only if it's not part of a contraction or possessive)
        elif char == "'":
            # If within a double-quoted section, treat as valid
            if quote_flag:
                pass  # Ignore single quotes inside double quotes
            elif i > 0 and sequence[i - 1].isalpha():
                match = CONTRACTIONS.match(sequence[i:])
                if match:
                    i += len(match.group()) - 1  # Skip over contraction
                # If it's part of a possessive noun (e.g., John's)
                elif i + 1 < len(sequence) and sequence[i + 1].isalpha():
                    pass  # Ignore possessive apostrophe
                # If it's a possessive apostrophe at the end of a word (e.g., "Jones'")
                elif i + 1 < len(sequence) and sequence[i - 1].isalpha():
                    pass  # Ignore possessives
                else:
                    single_quote_flag = not single_quote_flag  # Toggle open/close state
            else:
                single_quote_flag = not single_quote_flag  # Toggle open/close state

        i += 1

    # True = All quotes are matched; False = Unmatched quote(s) detected.
    return not quote_flag and not single_quote_flag


# 🚀 Test Cases
test_cases = [
    ('"Hello, world!"', True),
    ("'Hello, world!'", True),
    ("It's a beautiful day.", True),
    ("'John's book'", True),
    ("Mr. Jones' wallet", True),
    ("the foxes' tracks.", True),
    ("He said, 'Let's go!", False),
    ("She said, 'What's up?'", True),
    ("say 'hip hip hooray'.", True),
    ("\"The word is 'trophy'.\"", True),  # Nested pair
    ("'Unmatched single quote", False),
    ('"Unmatched double quote', False),
    ("'Hello", False),
    ("Hello'", False),
    ('She said, "Hello', False),
    ('She said, "Hello!"', True),
]

# Run Tests
for test, expected in test_cases:
    result = validate_enclosed_quotes(test)
    status = "✅" if result == expected else "❌"
    print(f"Expected {expected}, got {result} -> {status} | {test}")
