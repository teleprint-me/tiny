"""
Module: tiny.tokenizer
Description: A super simple character level tokenizer.
"""

import json
import os
import unicodedata

from tiny.config import TinyConfig


class Unicode:
    MAPPING_FILE = "data/unicode.json"

    def __init__(self):
        self.char_to_index, self.index_to_char = self.load_mapping()

    def generate_mapping(self):
        """Generate a Unicode character-to-index mapping with inverse mapping."""
        char_to_index = {}
        index_to_char = []

        index = 0
        for codepoint in range(0x110000):  # Unicode range up to 0x10FFFF
            char = chr(codepoint)
            category = unicodedata.category(char)

            # Exclude invalid characters (control chars, surrogates, unassigned)
            if not (0xD800 <= codepoint <= 0xDFFF) and not category.startswith("C"):
                char_to_index[char] = index
                index_to_char.append(char)  # Use list for faster lookups
                index += 1

        # Save as JSON
        data = {"forward": char_to_index, "inverse": index_to_char}
        os.makedirs(os.path.dirname(self.MAPPING_FILE), exist_ok=True)
        with open(self.MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return char_to_index, index_to_char

    def load_mapping(self):
        """Load the Unicode mapping if it exists, otherwise generate it."""
        if os.path.exists(self.MAPPING_FILE):
            with open(self.MAPPING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["forward"], data["inverse"]
        return self.generate_mapping()


class TinyTokenizer:
    def __init__(self, config):
        self.pad: str = config.pad or "<pad>"
        self.bos: str = config.bos or "<bos>"
        self.eos: str = config.eos or "<eos>"
        self.unk: str = config.unk or "<unk>"

        # Load Unicode mappings
        unicode = Unicode()
        self._stoi = unicode.char_to_index
        self._itos = unicode.index_to_char

        # Vocabulary with special tokens at the beginning
        self._vocab = self.special + self._itos

        # Precompute stoi and itos mappings
        self._stoi = {s: i for i, s in enumerate(self._vocab)}
        self._itos = self._vocab  # Keep list-based lookup

    @property
    def special(self):
        return [self.pad, self.bos, self.eos, self.unk]

    @property
    def vocab(self):
        return self._vocab  # Already precomputed

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def stoi(self):
        return self._stoi  # Use precomputed dict

    @property
    def itos(self):
        return self._itos  # Use precomputed list

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk]

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
