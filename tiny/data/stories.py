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


# NOTE: Just keep this simple for now. At least until the parser is fully ironed out.
# WARN: Due to the variety of datasets, this cannot be cleanly abstracted and generalized.
class TinyStoriesPath:
    """Manages TinyStories dataset paths and ensures dataset availability."""

    def __init__(
        self, dataset_name: str = "tinystories", root_dir: str = "data", verbose: bool = False
    ):
        """
        Args:
            dataset_name (str): The name of the dataset (default: 'tinystories').
            root_dir (str): The root data directory (default: 'data').
            verbose (bool): Enable verbose logging.
        """
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = TinyDataDownloader(root_dir=self.dataset_dir, verbose=verbose)

        # NOTE: File references are asymmetrical, e.g. TinyStoriesV2-GPT4-train.txt
        # NOTE: We are ONLY using the GPT-3.5 dataset for now.
        self.source_file = self.dataset_dir / "TinyStories-train.txt"  # Input file (train)
        self.training = self.dataset_dir / "TinyStories-train.json"  # Output file (train)
        self.validation = self.dataset_dir / "TinyStories-valid.json"  # Output file (valid)

    @property
    def source_url(self) -> str:
        """Returns the URL for downloading the TinyStories GPT-3.5 dataset."""
        base = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{file}?download=true"
        return base.format(file="TinyStories-train.txt")

    def read_or_download(self) -> str:
        """Downloads the dataset if missing, then reads it."""
        return self.downloader.read_or_download(self.source_url, str(self.source_file), "text")

    def save(self, data: list[dict[str, str]], split: float = 0.8) -> None:
        """
        Splits and saves the dataset into training and validation JSON files.

        Args:
            data (list): The processed dataset (list of dicts with "input"/"target").
            split (float): Percentage of data to use for training (default: 0.8).
        """
        split_idx = int(len(data) * split)  # Ensure integer index
        train_data = data[:split_idx]  # First `split%` of data for training
        valid_data = data[split_idx:]  # Remaining data for validation

        with open(self.training, "w", encoding="utf-8") as file:
            json.dump(train_data, file, indent=2, ensure_ascii=False)
        with open(self.validation, "w", encoding="utf-8") as file:
            json.dump(valid_data, file, indent=2, ensure_ascii=False)


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
    step = input_size + target_size  # Prevents out-of-bounds errors
    for i in range(start=0, stop=len(story_sentences), step=step):
        chunk = i + input_size  # slice sentences into chunks
        input_sentences = " ".join(story_sentences[i:chunk]).strip()
        target_sentences = " ".join(story_sentences[chunk : i + step]).strip()

        if input_sentences and target_sentences:
            pairs.append({"input": input_sentences, "target": target_sentences})

    return pairs


def main():
    """Main function to download, process, and save a subset of the dataset."""
    args = TinyDataArgs("Download and pre-process Tiny Stories.").parse_args()

    path = TinyStoriesPath(root_dir=args.dir, verbose=args.verbose)
    text = path.read_or_download()

    print("Extracting stories...")
    stories = extract_stories(text)

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
        data = all_pairs
    else:
        print(f"Selecting {args.samples} random samples...")
        data = random.sample(all_pairs, args.samples)

    print(f"Saving dataset to {path.dataset_dir}...")
    path.save(data, split=args.split)
    print("Done! Dataset saved.")


if __name__ == "__main__":
    main()
