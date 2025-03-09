"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.
"""

from string import printable

from tiny.config import TinyConfig


class TinyTokenizer:
    def __init__(self, config: TinyConfig):
        self.pad: str = config.pad or "<pad>"
        self.bos: str = config.bos or "<bos>"
        self.eos: str = config.eos or "<eos>"
        self.unk: str = config.unk or "<unk>"

    @property
    def special(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]

    @property
    def vocab(self) -> list[str]:
        return self.special + list(printable)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def stoi(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.vocab)}

    @property
    def itos(self) -> dict[int, str]:
        return {i: s for s, i in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab.index(self.pad)

    @property
    def bos_id(self) -> int:
        return self.vocab.index(self.bos)

    @property
    def eos_id(self) -> int:
        return self.vocab.index(self.eos)

    @property
    def unk_id(self) -> int:
        return self.vocab.index(self.unk)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = [self.stoi.get(c, self.unk_id) for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[t] for t in tokens if t > 3)


# Usage example
if __name__ == "__main__":
    config = TinyConfig()
    tokenizer = TinyTokenizer(config)
    encoded = tokenizer.encode("Hello, world!", add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
