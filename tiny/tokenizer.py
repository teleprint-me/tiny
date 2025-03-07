"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.
"""

from dataclasses import dataclass
from string import printable


@dataclass(frozen=True)
class TinySpecial:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

    @property
    def tokens(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]

    @property
    def pad_id(self) -> int:
        return self.tokens.index(self.pad)

    @property
    def bos_id(self) -> int:
        return self.tokens.index(self.bos)

    @property
    def eos_id(self) -> int:
        return self.tokens.index(self.eos)

    @property
    def unk_id(self) -> int:
        return self.tokens.index(self.unk)


class TinyTokenizer:
    def __init__(self):
        self.special = TinySpecial()
        self.chars = self.special.tokens + list(printable)
        self.stoi = {s: i for i, s in enumerate(self.chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.stoi.get(c, self.stoi[self.special.unk]) for c in text]
        if add_bos:
            tokens = [self.stoi[self.special.bos]] + tokens
        if add_eos:
            tokens = tokens + [self.stoi[self.special.eos]]
        return tokens

    def decode(self, tokens: list[int]):
        return "".join(self.itos[t] for t in tokens if t > 3)


# Usage example
if __name__ == "__main__":
    tokenizer = TinyTokenizer()
    encoded = tokenizer.encode("Hello, world!")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
