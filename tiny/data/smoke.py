"""
Script: tiny.data.smoke
Description: Runs a smoke test on a generated dataset to detect discrepancies and determine max sequence length.

Discrepancies include:
- Extremely short sequences (≤ 3 characters).
- Sequences beginning/ending with a quote or apostrophe.
- Sequences starting with a terminal symbol.
- Unusual tokenization issues.

It also computes the **maximum sequence length** across all input-target pairs.
"""

import argparse
import json
import logging
from typing import Dict, List

TERMINALS = {".", "!", "?", "‼", "‽", "⁇", "⁈", "⁉"}


def load_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_pairs(dataset: List[Dict[str, str]]) -> List[str]:
    """
    Check dataset for common formatting discrepancies.

    Returns:
        List of warnings (if any).
    """
    warnings = []
    for idx, pair in enumerate(dataset):
        input_text = pair["input"].strip()
        target_text = pair["target"].strip()

        if len(input_text) <= 3 or len(target_text) <= 3:
            warnings.append(f"Pair {idx}: Extremely short sequence detected.")

        if input_text[0] in TERMINALS:
            warnings.append(f"Pair {idx}: Input starts with a terminal symbol ({input_text[0]}).")

        if input_text[-1] in {'"', "'"} or target_text[-1] in {'"', "'"}:
            warnings.append(f"Pair {idx}: Sequence ends with a quote/apostrophe.")

    return warnings


def calc_max_seq(dataset: List[Dict[str, str]], tokenizer) -> int:
    """
    Calculate the maximum sequence length in dataset.
    - Ensures max length is power-of-2 aligned.
    """
    logging.info("Calculating the maximum sequence length.")
    max_seq = 0

    for pair in dataset:
        source = tokenizer.encode(pair["input"], add_bos=True)
        target = tokenizer.encode(pair["target"], add_eos=True)
        max_seq = max(len(source) + len(target), max_seq)

    # Log the real maximum sequence length
    logging.info(f"Calculated Maximum sequence length is {max_seq}")

    # Round up to nearest power of 2
    max_seq = 2 ** ((max_seq - 1).bit_length())

    # Log the recommended maximum sequence length
    logging.info(f"Recommended maximum sequence length set to {max_seq}")
    return max_seq


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a smoke test on a dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Loading dataset...")
    dataset = load_dataset(args.dataset)

    logging.info("Validating dataset...")
    issues = validate_pairs(dataset)
    if issues:
        logging.warning(f"{len(issues)} issues found:")
        for issue in issues:
            logging.warning(issue)
    else:
        logging.info("No issues found in dataset!")

    # Dummy tokenizer for now (replace with real tokenizer)
    class DummyTokenizer:
        def encode(self, text, add_bos=False, add_eos=False):
            return text.split()  # Simple token split for testing

    tokenizer = DummyTokenizer()
    max_seq = calc_max_seq(dataset, tokenizer)

    logging.info(f"Final Maximum Sequence Length: {max_seq}")


if __name__ == "__main__":
    main()
