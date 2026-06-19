#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${DEST_DIR:-$SCRIPT_DIR/data/ibm_aml_dataset/raw}"
DEST_FILE="$DEST_DIR/HI-Small_Trans.csv.zip"
KAGGLE_DATASET="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
HF_MIRROR_URL="https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip"

mkdir -p "$DEST_DIR"

if [ -f "$DEST_FILE" ]; then
    echo "Already exists: $DEST_FILE"
    ls -lh "$DEST_FILE"
    exit 0
fi

if command -v kaggle >/dev/null 2>&1; then
    echo "Downloading official Kaggle file: $KAGGLE_DATASET / HI-Small_Trans.csv.zip"
    kaggle datasets download \
        -d "$KAGGLE_DATASET" \
        -f HI-Small_Trans.csv.zip \
        -p "$DEST_DIR"
else
    echo "kaggle command not found; using Hugging Face mirror for HI-Small_Trans.csv.zip"
    echo "Mirror: $HF_MIRROR_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "$DEST_FILE" "$HF_MIRROR_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$DEST_FILE" "$HF_MIRROR_URL"
    else
        echo "ERROR: need kaggle, curl, or wget to download the dataset." >&2
        exit 2
    fi
fi

if [ ! -f "$DEST_FILE" ]; then
    echo "ERROR: download did not create $DEST_FILE" >&2
    exit 3
fi

ls -lh "$DEST_FILE"
