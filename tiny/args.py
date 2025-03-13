"""
Copyright © 2025 Austin Berrio
Module: tiny.args
Description: This modules manages the CLI parameters for TingConfig.
"""

from argparse import ArgumentParser, Namespace


# TODO: Rename to TinyModelArgs for clarity
class TinyArgs:
    def __init__(self, description: str = "Tiny CLI Tool"):
        self.parser = ArgumentParser(description=description)

    def parse_args(self, mode: str) -> Namespace:
        """
        Parses command-line arguments for a given mode ('trainer' or 'generator').

        Args:
            mode (str): The mode of execution ('trainer' or 'generator').

        Returns:
            argparse.Namespace: Parsed arguments.
        """

        # Shared arguments
        self.add_general_args()
        self.add_device_args()
        self.add_tokenizer_args()
        self.add_model_args()

        if mode == "trainer":
            self.add_dataset_args()  # Training input-target pairs
            self.add_trainer_args()  # Training hyperparameters
            self.add_optimizer_args()  # Optimizer settings
            self.add_criterion_args()  # Loss function

        elif mode == "generator":
            self.add_generator_args()  # Input for inference

        else:
            raise ValueError("Invalid mode. Use 'trainer' or 'generator'.")

        return self.parser.parse_args()

    def add_general_args(self) -> None:
        """Common arguments shared across training and inference."""

        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose mode",
        )

    def add_device_args(self) -> None:
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        self.parser.add_argument(
            "--dname",
            choices=["cpu", "cuda", "mps"],
            default="cpu",
            help="Device type.",
        )

    def add_tokenizer_args(self) -> None:
        self.parser.add_argument(
            "--vocab-path",
            required=True,
            help="Path to tokenizer models vocabulary.",
        )
        self.parser.add_argument(
            "--pad-token",
            default="<pad>",
            help="Set the pad token (Default: <pad>)",
        )
        self.parser.add_argument(
            "--bos-token",
            default="<bos>",
            help="Set the beginning-of-sequence token (Default: <bos>)",
        )
        self.parser.add_argument(
            "--eos-token",
            default="<eos>",
            help="Set the end-of-sequence token (Default: <eos>)",
        )
        self.parser.add_argument(
            "--unk-token",
            default="<unk>",
            help="Set the unkown token (Default: <unk>).",
        )
        self.parser.add_argument(
            "--add-bos",
            action="store_false",
            help="Enable beggining-of-sequence token (Default: True).",
        )
        self.parser.add_argument(
            "--add-eos",
            action="store_false",
            help="Enable end-of-sequence token (Default: True).",
        )

    def add_model_args(self) -> None:
        """Model architecture and configuration."""

        self.parser.add_argument(
            "--model-path",
            required=True,
            help="Path to save or load the model.",
        )
        self.parser.add_argument(
            "--max-seq",
            type=int,
            default=256,
            help="Maximum sequence length (Default: 256).",
        )
        self.parser.add_argument(
            "--d-model",
            type=int,
            default=128,
            help="Embedding dimension size (Default: 128).",
        )
        self.parser.add_argument(
            "--num-heads",
            type=int,
            default=16,
            help="Number of attention heads (Default: 16).",
        )
        self.parser.add_argument(
            "--eps",
            type=float,
            default=1e-6,
            help="Epsilon for numerical stability (Default: 1e-6).",
        )
        self.parser.add_argument(
            "--ff-mult",
            type=float,
            default=3.0,
            help="Feed-forward network expansion factor (Default: 3.0).",
        )
        self.parser.add_argument(
            "--num-layers",
            type=int,
            default=4,
            help="Number of transformer layers (Default: 4).",
        )

    def add_dataset_args(self) -> None:
        """Arguments related to dataset loading and preprocessing."""

        self.parser.add_argument(
            "--dataset-path",
            required=True,
            help="Path to dataset file (.json).",
        )
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for training (Default: 8).",
        )
        self.parser.add_argument(
            "--shuffle",
            action="store_true",
            help="Shuffle the dataset (Default: False).",
        )

    def add_trainer_args(self) -> None:
        """Training hyperparameters."""

        self.parser.add_argument(
            "--num-epochs",
            type=int,
            default=10,
            help="Number of training epochs (Default: 10).",
        )
        self.parser.add_argument(
            "--save-every",
            type=int,
            default=10,
            help="Save model every N epochs (Default: 10).",
        )
        self.parser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Gradient accumulation steps (Default: 1).",
        )

    def add_optimizer_args(self) -> None:
        """Optimizer settings for training."""

        # General optimizer settings
        self.parser.add_argument(
            "--recurse",
            action="store_false",
            help="Optimizer will yield model parameters recursively (Default: True).",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="Optimizer learning rate (Default: 1e-3).",
        )
        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=1e-2,
            help="Weight decay (Default: 1e-2).",
        )
        self.parser.add_argument(
            "--amsgrad",
            action="store_true",
            help="Use AMSGrad variant of Adam (Default: False).",
        )

    def add_criterion_args(self) -> None:
        """Loss function selection."""

        self.parser.add_argument(
            "--reduction",
            choices=["mean", "sum", "none"],
            default="mean",
            help="Reduction method for loss calculation (Default: mean).",
        )

    def add_generator_args(self) -> None:
        """Arguments required for inference (text generation)."""

        self.parser.add_argument(
            "--prompt",
            required=True,
            help="Input text prompt for generation.",
        )
        self.parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Maximum number of tokens to generate (Default: 128).",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=0.5,
            help="Sampling temperature (Default: 0.5). Lower is more deterministic.",
        )
        self.parser.add_argument(
            "--top-k",
            type=int,
            default=50,
            help="Top-K sampling size (Default: 50). Lower values make generation more focused.",
        )
        self.parser.add_argument(
            "--top-p",
            type=float,
            default=0.9,
            help="Top-P (nucleus) sampling probability (Default: 0.9). Controls diversity.",
        )
        self.parser.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.2,
            help="Penalty for repeated tokens (Default: 1.2). Higher values discourage repetition.",
        )
        self.parser.add_argument(
            "--greedy",
            action="store_true",
            help="Enable greedy decoding instead of sampling (Default: False).",
        )
