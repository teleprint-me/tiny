#!/usr/bin/env bash

if [ ! -d .venv ]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
fi

pip install -r requirements.txt --upgrade

case $1 in
    "cuda")
        pip install torch
        ;;
    "rocm")
        pip install torch --index-url https://download.pytorch.org/whl/rocm
        ;;
    *)
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
esac
