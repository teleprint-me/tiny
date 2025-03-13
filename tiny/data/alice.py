"""
Script: tiny.data.alice
Description: Download, pre-process, and convert Alice in Wonderland to JSON source-target pairs.

Gutenberg License and Terms are clear and all Gutenberg references must be stripped from the final
output. Any reference to Gutenberg is trademarked. The actual, original, work is in the Public Domain.
"""

from pathlib import Path

from tiny.data.args import TinyDataArgs
from tiny.data.downloader import TinyDownloader


def get_source_url() -> str:
    """Cached plaintext file for Alice in Wonderland."""
    return "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def extract_corpus(text: str) -> str:
    """Extracts Alice in Wonderland text, removing Project Gutenberg headers/footers."""
    start_line = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
    end_line = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"

    lines = text.splitlines()
    start_idx = -1
    end_idx = -1

    # Locate the start of the book
    for i, line in enumerate(lines):
        if start_line == line.strip():
            start_idx = i + 2  # Skip the start marker and tag
            break  # Stop after the first match

    # Locate the only fully uppercase "THE END"
    for i, line in enumerate(lines):
        if end_line == line.strip():  # Exact match
            end_idx = i - 1  # Skip the end marker
            break  # Stop after the first uppercase occurrence

    # Trim the content
    if start_idx != -1 and end_idx != -1:
        return "\n".join(lines[start_idx:end_idx]).strip()

    return text  # Return unmodified if markers aren't found


def main() -> None:
    args = TinyDataArgs("Download and preprocess Alice in Wonderland.").parse_args()

    root_dir = Path(args.dir)
    root_dir.mkdir(exist_ok=True)
    source_path = root_dir / "alice.txt"
    target_path = root_dir / "alice.json"

    text = TinyDownloader(get_source_url(), source_path).read_or_download("text")
    corpus = extract_corpus(text)

    # Debug output (verify correctness before proceeding to Step 2)
    print(corpus)  # Preview the first 500 characters


if __name__ == "__main__":
    main()
