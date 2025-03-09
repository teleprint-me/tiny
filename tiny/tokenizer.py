"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.
"""

import json
import os
import unicodedata

from tiny.config import TinyConfig

# Define the JSON file to store the mapping


class Unicode:
    MAPPING_FILE = "data/unicode.json"

    def generate_mapping(self):
        """Generate a Unicode character-to-index mapping with inverse mapping."""
        mapping = {}
        inverse_mapping = {}

        # Iterate over all valid Unicode characters
        index = 0
        for codepoint in range(0x110000):  # Unicode range up to 0x10FFFF
            char = chr(codepoint)
            category = unicodedata.category(char)

            # Exclude invalid characters (control chars, surrogates, unassigned)
            if not (0xD800 <= codepoint <= 0xDFFF) and not category.startswith("C"):
                mapping[char] = index
                inverse_mapping[index] = char
                index += 1

        # Save as JSON
        data = {"forward": mapping, "inverse": inverse_mapping}
        with open(self.MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return mapping, inverse_mapping

    def load_mapping(self):
        """Load the Unicode mapping if it exists, otherwise generate it."""
        os.makedirs(os.path.dirname(self.MAPPING_FILE), exist_ok=True)
        if os.path.exists(self.MAPPING_FILE):
            with open(self.MAPPING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["forward"], {int(k): v for k, v in data["inverse"].items()}
        return self.generate_mapping()


class TinyTokenizer:
    def __init__(self, config: TinyConfig):
        self.pad: str = config.pad or "<pad>"
        self.bos: str = config.bos or "<bos>"
        self.eos: str = config.eos or "<eos>"
        self.unk: str = config.unk or "<unk>"
        unicode = Unicode()
        self._stoi, self._itos = unicode.generate_mapping()
        self._vocab = [v for v in self._itos.values()]

    @property
    def special(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]

    @property
    def vocab(self) -> list[str]:
        return self.special + self._vocab

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
        """Decodes a list of token IDs into a string."""
        return "".join(self.itos.get(t, self.unk) for t in tokens if t in self.itos)


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
