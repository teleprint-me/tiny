"""
Copyright © 2025 Austin Berrio
Module: tiny.generator
Description: A simple generator for the Tiny Transformer model.
"""

import torch
import torch.nn.functional as F

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

    def generate(self, prompt: str):
        """Generates text from the given prompt with token streaming."""

        # Encode input sequence
        input_ids = (
            torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
        )

        # Generate tokens iteratively
        for _ in range(self.config.max_tokens):
            logits = self.model(input_ids)  # Forward pass
            logits = logits[:, -1, :]  # Take last token logits
            probs = F.softmax(logits / self.config.temperature, dim=-1)

            # Sample next token (Top-K or Greedy)
            if self.config.greedy:
                _, next_token = torch.topk(probs, k=1, dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = next_token.item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=self.device)], dim=1
            )

            # Stop if we hit the EOS token
            if next_token == self.tokenizer.eos_id:
                break

            yield self.tokenizer.decode([next_token])  # Stream token immediately


if __name__ == "__main__":
    from tiny.args import TinyArgs

    args = TinyArgs("Tiny Generator CLI").parse_args("generator")

    config = TinyConfig(
        # General
        verbose=args.verbose,
        # Device
        seed=args.seed,
        dname=args.dname,
        dtype=torch.float32,
        # Tokenizer
        vocab_path=args.vocab_path,
        add_bos=args.add_bos,  # Frozen: To be decided
        add_eos=args.add_eos,  # Frozen: To be decided
        # Model
        model_path=args.model_path,
        # Generator
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
    )

    generator = TinyGenerator(config)
    for token in generator.generate(args.prompt):
        print(token, end="", flush=True)
    print()
