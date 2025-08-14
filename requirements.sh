#!/usr/bin/env bash

# Function to display usage information
usage() {
    echo "Usage: $0 <cuda|rocm|cpu>"
    echo "  cuda: Install PyTorch with CUDA support"
    echo "  rocm: Install PyTorch with ROCM support"
    echo "  cpu: Install PyTorch for CPU support (default)"
    exit 1
}

# Check for valid argument or default to CPU
if [ -z "$1" ]; then
    echo "No environment specified, defaulting to CPU."
    ENVIRONMENT="cpu"
else
    case "$1" in
        "cuda")
            ENVIRONMENT="cuda"
            ;;
        "rocm")
            ENVIRONMENT="rocm"
            ;;
        *)
            usage
            ;;
    esac
fi

# Create virtual environment if it doesn't exist
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    echo "Virtual environment created and activated."
else
    echo "Virtual environment already exists."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --upgrade

# Install PyTorch based on the specified environment
case "$ENVIRONMENT" in
    "cuda")
        echo "Installing PyTorch with CUDA support..."
        pip install torch
        ;;
    "rocm")
        echo "Installing PyTorch with ROCM support..."
        pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
        ;;
    *)
        echo "Installing PyTorch for CPU support..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

echo "Setup completed."
deactivate
