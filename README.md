# Tiny

Tiny is a super simple Transformer implementation.

The Transformer model is actually rather simple in implementation. The
complexity arises from the surrounding tooling and pipeline.

Tiny is designed to simplify that pipeline down to its core fundementals.

## Setup

### 1. Clone the repository

```sh
git clone https://github.com/teleprint-me/tiny.git
cd tiny
```

### 2. Set up a virtual environment

```sh
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

#### Install requirements

```sh
pip install -U pip
pip install -r requirements.txt --upgrade
```

#### Install PyTorch

- **CUDA (NVIDIA GPUs)**

```sh
pip install torch
```

_PyTorch defaults to CUDA if available._

- **ROCm (AMD GPUs)**

```sh
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

- **CPU only**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Automated Setup

A convenience script (`requirements.sh`) is provided:

```sh
chmod +x requirements.sh
./requirements.sh <cuda|rocm|cpu>
```

_(Defaults to CPU if unspecified.)_

## Usage

### Download

Download the raw text:

```sh
python -m tiny.download --dir data
```

### Tokenizer Training

Train the tokenizer on the raw text.

```sh
python -m tiny.tokenizer --merges 5000 --corpus data/stripped --save model/tokenizer.json
```

_Expects plaintext input._

### Build the Dataset

TODO

### Pre-training

Train the model from scratch:

```sh
python -m tiny.trainer --dname cuda \
    --vocab-path model/vocab.json \
    --model-path model/tiny.pth \
    --dataset-path data/hotpot.json \
    --save-every 1
```

_In progress._
