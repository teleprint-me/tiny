# Tiny

Tiny is a super simple Transformer implementation for debugging [Mini](https://github.com/teleprint-me/mini.git).

The Transformer model is actually rather simple in implementation. The complexity arises from the surrounding tooling and pipeline.

Tiny is designed to simplify that pipeline down to its core fundementals.

## **Installation & Setup**

### **1. Clone the repository**

```sh
git clone https://github.com/teleprint-me/mini.git
cd mini
```

### **2. Setup a virtual environment**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

### **3. Install dependencies**

#### **Install PyTorch**

- **CPU**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- **CUDA**

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

- **ROCm**

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2.4
```

#### **Install Requirements**

```sh
pip install -r requirements.txt
```

## **Usage**

### **Dataset Preparation**

Download the hotpot dataset:

```sh
python -m tiny.hotpot --dataset dev --samples 100 --output data/hotpot.json
```

_Samples are selected at random._

### **Pre-training**

Train a model from scratch on a dataset:

```sh
python -m tiny.trainer --dname cuda --vocab-path model/vocab.json --model-path model/tiny.pth --dataset-path data/hotpot.json --save-every 1
```
