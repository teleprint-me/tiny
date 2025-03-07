class CharTokenizer:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")
        self.chars = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"] + self.chars
        self.stoi = {s: i for i, s in enumerate(self.chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text, max_len=20):
        tokens = [self.stoi.get(c, self.stoi["[UNK]"]) for c in text.lower()]
        tokens = [self.stoi["[SOS]"]] + tokens + [self.stoi["[EOS]"]]
        return tokens + [self.stoi["[PAD]"]] * (max_len - len(tokens))

    def decode(self, tokens):
        return "".join(self.itos[t] for t in tokens if t > 3)


tokenizer = CharTokenizer()
print(tokenizer.encode("hello"))
print(tokenizer.decode(tokenizer.encode("hello")))
