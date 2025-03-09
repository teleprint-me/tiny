"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.

NOTE: list() is O(n) and dict() is O(1).
The initial unicode code point computation is costly, but is mitigated by precomputation
and caching the results. A list() can be leveraged to generate chunks of code points to
parallelize the initial mapping. Once the indices and code points are precomputed, the
dict() loses its ordering, but the mapping should implicitly preserve the expected indices.
"""

import functools
import json
import os
import unicodedata

from tiny.config import TinyConfig


class TinyVocab:
    MAPPING_FILE = "data/vocab.json"

    def __init__(self, config: TinyConfig):
        self.pad: str = config.pad or "<pad>"
        self.bos: str = config.bos or "<bos>"
        self.eos: str = config.eos or "<eos>"
        self.unk: str = config.unk or "<unk>"

    @functools.lru_cache
    def special(self) -> list[str]:
        """Returns the list of special tokens."""
        return [self.pad, self.bos, self.eos, self.unk]

    @functools.lru_cache
    def mapping(self) -> list[str]:
        """Returns the full vocabulary, including special tokens."""
        return self._load_or_generate()

    @functools.lru_cache
    def stoi(self) -> dict[str, int]:
        """Returns string-to-index mapping."""
        return {s: i for i, s in enumerate(self.mapping())}

    @functools.lru_cache
    def itos(self) -> dict[int, str]:
        """Returns index-to-string mapping."""
        return {i: s for i, s in enumerate(self.mapping())}

    def _load(self) -> list[str]:
        with open(self.MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, mapping: list[str]) -> None:
        # Save to JSON
        os.makedirs(os.path.dirname(self.MAPPING_FILE), exist_ok=True)
        with open(self.MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    def _load_or_generate(self) -> list[str]:
        """Loads vocabulary if it exists, otherwise generates it."""
        try:
            return self._load()
        except FileNotFoundError:
            return self._generate()

    def _generate(self) -> list[str]:
        """Generate a Unicode character-to-index mapping."""
        # NOTE: This is O(n) because the computation is linear and single threaded.
        mapping = self.special()[:]  # Create a shallow copy

        # Unicode range up to 0x10FFFF
        for codepoint in range(0x110000):
            char = chr(codepoint)
            category = unicodedata.category(char)

            # Exclude control characters, surrogates, unassigned characters
            if not (0xD800 <= codepoint <= 0xDFFF) and not category.startswith("C"):
                mapping.append(char)

        self._save(mapping)

        return mapping


class TinyTokenizer:
    def __init__(self, config: TinyConfig):
        # Precompute the models vocabulary
        self._vocab = TinyVocab(config)

    @property
    def vocab(self) -> list[str]:
        return self._vocab.mapping()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab.mapping())

    @property
    def stoi(self) -> dict[str, int]:
        return self._vocab.stoi()

    @property
    def itos(self) -> dict[int, str]:
        return self._vocab.itos()

    @property
    def pad_id(self) -> int:
        return self.stoi[self._vocab.pad]

    @property
    def bos_id(self) -> int:
        return self.stoi[self._vocab.bos]

    @property
    def eos_id(self) -> int:
        return self.stoi[self._vocab.eos]

    @property
    def unk_id(self) -> int:
        return self.stoi[self._vocab.unk]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encodes a string into a list of token indices."""
        tokens = [self.stoi.get(c, self.unk_id) for c in text]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs into a string."""
        return "".join(self.itos[t] if t < len(self.itos) else self.unk for t in tokens)


# Usage example
if __name__ == "__main__":
    config = TinyConfig()
    tokenizer = TinyTokenizer(config)
    print(f"Vocab Size: {tokenizer.vocab_size}")

    text = "Hello, world!"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"Text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    text = "こんにちは世界！"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"Text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
