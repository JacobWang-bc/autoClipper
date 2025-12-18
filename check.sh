#!/bin/bash
# Linux/Mac script for code checking

echo "============================================================"
echo "Start code checking..."
echo "============================================================"

echo ""
echo "[1/3] Ruff code quality checking..."
ruff check funclip
RUFF_CHECK=$?

if [ $RUFF_CHECK -ne 0 ]; then
    echo ""
    echo "Found errors! Run 'ruff check --fix funclip' to automatically fix some problems"
fi

echo ""
echo "[2/3] Ruff code format checking..."
ruff format --check funclip
RUFF_FORMAT=$?

if [ $RUFF_FORMAT -ne 0 ]; then
    echo ""
    echo "Format needs to be adjusted! Run 'ruff format funclip' to automatically format"
fi

echo ""
echo "[3/3] Pyright type checking..."
if command -v pyright &> /dev/null; then
    pyright funclip
else
        echo "Pyright not installed, skipping type checking"
    echo "Install command: pip install pyright or npm install -g pyright"
fi

echo ""
echo "============================================================"
echo "Checking completed!"
echo "============================================================"

