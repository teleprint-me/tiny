"""
Script: tiny.data.gutenberg
Description: Download, pre-process, and convert Alice in Wonderland to JSON source-target pairs.

Gutenberg License and Terms are clear and all Gutenberg references must be stripped from the final
output. Any reference to Gutenberg is trademarked. The actual, original, work is in the Public Domain.
"""

from pathlib import Path

from tiny.data.args import TinyDataArgs
from tiny.data.downloader import TinyDataDownloader


class TinyDataPath:
    """Data structure for managing file paths."""

    def __init__(self, dir_path: str):
        self.dir = Path(dir_path)
        self.dir.mkdir(exist_ok=True)

    @property
    def source(self) -> Path:
        """Path to the source file."""
        return self.dir / "alice.txt"

    @property
    def train(self) -> Path:
        """Path to the training dataset."""
        return self.dir / "alice_train.json"

    @property
    def valid(self) -> Path:
        """Path to the validation dataset."""
        self.valid_path = self.dir / "alice_valid.json"

    def url(self) -> str:
        """Cached plaintext file for Alice in Wonderland."""
        return "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def extract_corpus(text: str) -> str:
    """Extracts Alice in Wonderland text, removing Project Gutenberg headers/footers."""
    start_marker = "START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "END OF THE PROJECT GUTENBERG EBOOK"

    lines = text.splitlines()
    start_idx = -1
    end_idx = -1

    # Locate the start of the book
    for i, line in enumerate(lines):
        if start_marker in line.strip():
            start_idx = i + 2  # Skip the start marker and tag
            break  # Stop after the first match

    # Locate the only fully uppercase "THE END"
    for i, line in enumerate(lines):
        if end_marker in line.strip():  # Exact match
            end_idx = i - 1  # Skip the end marker
            break  # Stop after the first uppercase occurrence

    # Trim the content
    if start_idx != -1 and end_idx != -1:
        return "\n".join(lines[start_idx:end_idx]).strip()

    return text  # Return unmodified if markers aren't found


def generate_pairs(sentences: list[str], input_size: int, target_size: int) -> list[dict]:
    """Creates source-target pairs from the text."""
    pairs = []
    step = input_size + target_size
    for i in range(0, len(sentences), step):
        segment = i + input_size
        source = " ".join(sentences[i:segment])
        target = " ".join(sentences[segment : i + step])
        pairs.append({"source": source, "target": target})
    return pairs


def main() -> None:
    args = TinyDataArgs("Download and preprocess Alice in Wonderland.").parse_args("alice")
    path = TinyDataPath(args.dir)
    text = TinyDataDownloader(path.url, path.source).read_or_download("text")
    corpus = extract_corpus(text)

    # Debug output (verify correctness before proceeding to Step 2)
    print("First 250 characters:")
    print(corpus[:250])  # Preview the first 500 characters


if __name__ == "__main__":
    main()
