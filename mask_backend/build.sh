#!/usr/bin/env bash
# build.sh  — Render build script
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Training ML model from df.csv..."
python train_model.py --csv df.csv

echo "==> Build complete ✅"
