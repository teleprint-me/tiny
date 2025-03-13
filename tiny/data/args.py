"""
Copyright © 2025 Austin Berrio
Module: tiny.data.args
Description: This modules manages common CLI parameters for managing datasets.
"""

from argparse import ArgumentParser, Namespace


class TinyDataArgs:
    def __init__(self, description: str = "Tiny CLI Tool"):
        self.parser = ArgumentParser(description=description)

    def parse_args(self) -> Namespace:
        self.parser.add_argument(
            "--samples",
            type=int,
            default=100,
            help="Number of samples to select (default: 100).",
        )
        self.parser.add_argument(
            "--all",
            action="store_true",
            help="Select all samples. Overrides --samples when True (Default: False).",
        )
        self.parser.add_argument(
            "--split",
            type=float,
            default=0.8,
            help="Percentage for splitting between training and validation (Default: 0.8).",
        )
        self.parser.add_argument(
            "--input-size",
            type=int,
            default=2,
            help="Joins up to `input_size` sentences in input (Default: 2).",
        )
        self.parser.add_argument(
            "--target-size",
            type=int,
            default=1,
            help="Joins up to `target_size` sentences in target (Default: 1).",
        )
        self.parser.add_argument(
            "--dir",
            default="data",
            help="Directory to write generated files to (default: 'data').",
        )
        return self.parser.parse_args()
