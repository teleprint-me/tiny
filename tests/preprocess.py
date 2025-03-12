"""
Script: tests.preprocess
Description: Test preprocessing raw text.
"""

import re

# Matches common contractions
CONTRACTIONS = re.compile(r"\b(?:'t|'s|'re|'m|'ve|'d|'ll|'am|'em)\b", re.IGNORECASE)
# Define English sentence-terminal symbols
TERMINALS = {".", "!", "?", "‼", "‽", "⁇", "⁈", "⁉"}


def preprocess_story_lines(story: str) -> list[str]:
    """
    Process a story into a list of clean, individual sentences.

    - Handles both single and double quotes properly.
    - Processes **character-by-character** for maximum flexibility.
    - Only splits sentences when **not inside a quote**.
    - Ensures **clean sentence breaks** without concatenation issues.

    Returns:
        list[str]: A properly segmented list of sentences.
    """
    sentences = []
    current_sentence = []
    quote_flag = False
    single_quote_flag = False

    for line in story.splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        for i, char in enumerate(line):
            current_sentence.append(char)

            # Handle double quotes
            if char == '"':
                quote_flag = not quote_flag  # Toggle open/close state

            # Handle single quotes (only if it's not part of a contraction)
            elif char == "'" and (i == 0 or not line[i - 1].isalpha()):
                single_quote_flag = not single_quote_flag

            # Only split sentences if we are NOT inside a quote
            elif char in TERMINALS and not (quote_flag or single_quote_flag):
                sentences.append("".join(current_sentence).strip())
                current_sentence = []

        # Ensure the sentence is **fully stored before moving on**
        if current_sentence:
            sentences.append("".join(current_sentence).strip())
            current_sentence = []

    return sentences


story = """But then, Jane saw a dog in a window. The dog said 'Hello!' and wagged its tail. Jane was so happy to have made a new friend. She waved to the dog and said 'What's your name?'
The dog barked, and Jane realised that it couldn't tell her its name. Instead, it showed Jane that feeling happy was more important than saying hello.
Jane smiled to herself and shrugged. She knew that from now on, she would never be too shy to say 'hello' and make new friends."""

processed_sentences = preprocess_story_lines(story)
for sentence in processed_sentences:
    print(f"- {sentence}")
