#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1

streamlit run FINAL.py --server.address=127.0.0.1 --server.port=8501
