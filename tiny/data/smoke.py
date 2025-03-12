"""
Script: tiny.data.smoke
Description: This script performs on a smoke test on a generated dataset to determine if it contains any descrepencies.

Descrepencies include outliers in any input-target pair within the generated dataset.

Descrepencies may include, but may not be limited to:
- A sequence containing a single character.
- A sequence that is extremely short, e.g. 3 characters or less.
- A sequence beginning or ending with a quote or apostrophe.
- A sequence beginning with a terminal symbol.
- etc.

This script will also attempt to determine the maximum sequence length out of all the input-target pairs contained
within the dataset and output the result.
"""

import json


def load_dataset(filepath: str) -> list[dict[str, str]]:
    with open(filepath, "r") as file:
        return json.load(file)


def calc_max_seq(self) -> None:
    self.logger.info("Calculating the maximum sequence length.")
    # Calculate the longest pair
    for pair in self.dataset:
        # input is the beginning of the sequence
        source = self.tokenizer.encode(pair["input"], add_bos=self.config.add_bos)
        # target is the predicted output which continues on from the input sequence
        target = self.tokenizer.encode(pair["target"], add_eos=self.config.add_eos)
        # calculate the max length of any given sequence
        self.config.max_seq = max(len(source) + len(target), self.config.max_seq)

    # Max seq len must be evenly divisible
    if self.config.max_seq % 2 != 0:
        self.config.max_seq += 1  # Should probably be some power of 2, but I'm being lazy

    self.logger.info(f"Maximum sequence length set to {self.config.max_seq}")


def parse_args() -> argparse.Namespace:
    pass


if __name__ == "__main__":
    main()
