"""
Copyright © 2025 Austin Berrio
Module: tiny.generator
Description: A simple generator for the Tiny Transformer model.
"""

from tiny.config import TinyConfig
from tiny.model import TinyTransformer
from tiny.state import TinyState
from tiny.tokenizer import TinyTokenizer


class TinyGenerator:
    def __init__(self, config: TinyConfig):
        self.state = TinyState(config)
        self.state.load_model()
        self.logger = config.logger(self.__class__.__name__, config.verbose)

    # === 🔥 Convenience Properties === #
    @property
    def config(self) -> TinyConfig:
        return self.state.config

    @property
    def tokenizer(self) -> TinyTokenizer:
        return self.state.tokenizer

    @property
    def model(self) -> TinyTransformer:
        return self.state.model

    @property
    def device(self) -> TinyConfig:
        return self.config.device


if __name__ == "__main__":
    from tiny.args import TinyArgs

    args = TinyArgs("Tiny Generator CLI").parse_args("generator")
    config = TinyConfig()
    generator = TinyGenerator(config)
