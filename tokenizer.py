"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.
"""

from dataclasses import dataclass
from string import printable


@dataclass(frozen=True)
class Special:
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


class Tokenizer:
    def __init__(self, max_seq_len: int = 20):
        self.max_seq_len = max_seq_len
        self.special = Special()
        self.chars = self.special.tokens + list(printable)
        self.stoi = {s: i for i, s in enumerate(self.chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text: str):
        tokens = [self.stoi.get(c, self.stoi[self.special.unk]) for c in text]
        tokens = [self.stoi[self.special.bos]] + tokens + [self.stoi[self.special.eos]]
        return tokens + [self.stoi[self.special.pad]] * (self.max_seq_len - len(tokens))

    def decode(self, tokens: list[int]):
        return "".join(self.itos[t] for t in tokens if t > 3)


# Usage example
if __name__ == "__main__":
    tokenizer = Tokenizer()
    encoded = tokenizer.encode("Hello, world!")
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
