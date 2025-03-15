"""
Copyright © 2025 Austin Berrio
Script: tiny.data.stories
Description: Downloads and converts Tiny Stories to a simple tiny format.

This script downloads and converts Tiny Stories data into a simplified format suitable for
training a small-scale model. The goal is to obtain around 100 random samples, where each
sample is represented as an input-target pair. The original dataset is in plaintext, containing
short stories, usually 2 - 3 paragraphs in length, and use a simple vocabulary. The plaintext
must be converted to JSON and the dataset must contain sequences within a maximum length.
I'm thinking line by line might be okay per short story. Take each line in pairs. The first line
is the input and the second line is the target. Not sure yet. This is challenging.
"""

import json
import multiprocessing
import random
from pathlib import Path

from tiny.data.args import TinyDataArgs
from tiny.data.downloader import TinyDataDownloader


class TinyDataPath:
    """Data structure for managing file paths."""

    def __init__(self, dir_path: str):
        self.dir = Path(dir_path)
        self.dir.mkdir(exist_ok=True)

    @property
    def source_file(self) -> Path:
        """Path to the source file."""
        return self.dir / "stories.txt"

    @property
    def training(self) -> Path:
        """Path to the training dataset."""
        return self.dir / "stories_train.json"

    @property
    def validation(self) -> Path:
        """Path to the validation dataset."""
        self.valid_path = self.dir / "stories_valid.json"

    def url(self, label: str) -> str:
        """Source URL for Tiny Stories."""
        base = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
        return f"{base}/TinyStories-{label}.txt?download=true"

    def save(self, label: str, data: any) -> None:
        path = self.valid if label == "valid" else self.train
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def extract_stories(text: str) -> list[str]:
    """Extract individual stories from raw text using <|endoftext|> as delimiter."""

    stories = text.split("<|endoftext|>")
    num_workers = min(multiprocessing.cpu_count(), len(stories))  # Define after stories

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(str.strip, stories)  # Use map() since .strip() takes one arg

    return results


def clean_ascii(text: str) -> str:
    """
    Converts curly quotes and apostrophes to standard ASCII equivalents.
    This ensures clean, consistent formatting across all environments.
    """
    replacements = {
        "’": "'",  # Fancy apostrophe → ASCII apostrophe
        "‘": "'",  # Single open quote → ASCII apostrophe
        "”": '"',  # Fancy close double quote → ASCII double quote
        "“": '"',  # Fancy open double quote → ASCII double quote
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


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
    # Define English sentence-terminal symbols
    TERMINALS = {".", "!", "?", "‼", "‽", "⁇", "⁈", "⁉"}

    sentences = []
    current_sentence = []
    quote_flag = False
    single_quote_flag = False

    for line in story.splitlines():
        line = clean_ascii(line).strip()
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


def preprocess_stories(stories: list[str]) -> list[list[str]]:
    """Convert each story into a clean list of sentences using multiprocessing."""

    num_workers = min(multiprocessing.cpu_count(), len(stories))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(preprocess_story_lines, stories)

    return results


def generate_sentence_pairs(
    story_sentences: list[str], input_size: int = 2, target_size: int = 1
) -> list:
    """
    Convert preprocessed sentences into structured input-target pairs.

    - Explicitly joins **exactly** `input_size` sentences in input.
    - Ensures **exactly** `target_size` sentences in target.
    - Cleans text to **avoid unicode inconsistencies**.

    Args:
        story_sentences (list[str]): A preprocessed list of sentences.
        input_size (int): Number of sentences per input.
        target_size (int): Number of sentences per target.

    Returns:
        list: List of {"input": ..., "target": ...} pairs.
    """

    pairs = []
    start = len(story_sentences)
    step = input_size + target_size  # Prevents out-of-bounds errors
    for i in range(0, start, step):
        input_sentences = " ".join(story_sentences[i : i + input_size]).strip()
        target_sentences = " ".join(story_sentences[i + input_size : i + step]).strip()

        if input_sentences and target_sentences:
            pairs.append({"input": input_sentences, "target": target_sentences})

    return pairs


def main():
    """Main function to download, process, and save a subset of the dataset."""
    args = TinyDataArgs("Download and pre-process Tiny Stories.").parse_args()

    path = TinyDataPath(args.dir)
    downloader = TinyDataDownloader(args.dir, args.verbose)

    data = downloader.read_or_download(path.url, path.source, "text")

    print("Extracting stories...")
    stories = extract_stories(data)

    print(f"Processing {len(stories)} stories...")
    processed_stories = preprocess_stories(stories)

    print("Generating input-target pairs...")
    all_pairs = []
    for story_sentences in processed_stories:
        sentence_pairs = generate_sentence_pairs(story_sentences, args.input_size, args.target_size)
        all_pairs.extend(sentence_pairs)
    print(f"Generated {len(all_pairs)} samples.")

    if args.all_pairs or len(all_pairs) < args.samples:
        print(f"Warning: Using all {len(all_pairs)} pairs found.")
        sample = all_pairs
    else:
        print(f"Selecting {args.samples} random samples...")
        sample = random.sample(all_pairs, args.samples)

    print(f"Saving dataset to {destination_path}...")
    with open(destination_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print("Done! Dataset saved.")


if __name__ == "__main__":
    main()
