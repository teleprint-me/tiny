"""
Script: tiny.data.gutenberg
Description: Download, pre-process, and convert Alice in Wonderland to JSON source-target pairs.

Gutenberg License and Terms are clear and all Gutenberg references must be stripped from the final
output. Any reference to Gutenberg is trademarked. The actual, original, work is in the Public Domain.
"""

import argparse
import sys
import time
from pathlib import Path

import requests

library = [
    {
        "name": "The-Odyssey-Homer.txt",
        "url": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
    },
    {
        "name": "Alice-in-Wonderland-Lewis-Carroll.txt",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    },
    {
        "name": "The-Time-Machine-H-G-Wells.txt",
        "url": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
    },
    {
        "name": "The-Count-of-Monte-Cristo-Alexandre-Dumas.txt",
        "url": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    },
]


def download_file(url: str, file: Path):
    """Download a file from a URL to a local file, with simple progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded_size = 0
    last_print = -1  # So 0% will print

    print(f"[bytes] {total_size} [", end="")
    with open(file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                percent = int((downloaded_size / total_size) * 100) if total_size else 0
                if percent // 10 > last_print // 10:
                    print("*", end="")
                    sys.stdout.flush()
                    last_print = percent
    print(f"] [path] {file}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="data",
        help="Directory to write generated files to (default: 'data').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(args.dir)
    path.mkdir(exist_ok=True)

    corpus = []
    for book in library:
        name = book["name"]
        original = path / "original"
        original.mkdir(exist_ok=True)
        original_book = original / name

        # Skip books that already exist
        if not original_book.exists():
            time.sleep(0.35)
            url = book["url"]
            download_file(url, original_book)

        # strip gutenberg text from the book
        text = ""
        with open(original_book, "r", encoding="utf-8") as f:
            text = extract_corpus(f.read())

        if text:
            print(f"Extracted {len(text)}")
            # save the stripped book to disk
            stripped = path / "stripped"
            stripped_book = stripped / name
            with open(stripped_book, "w", encoding="utf-8") as f:
                f.write(text)

        # Collect and label text for pre-processing
        corpus.append({"name": name, "text": text})


if __name__ == "__main__":
    main()
