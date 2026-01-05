#!/bin/bash

set -e  # Exit on error

echo "=== Starting ML Training Pipeline ==="
echo ""

echo "Step 1: Generating datasets..."
python3 code/gen_data.py
echo "✓ Datasets generated"
echo ""

echo "Step 2: Training Part 1 (Math GPT)..."
python3 code/train_part1.py
echo "✓ Part 1 training complete"
echo ""

echo "Step 3: Training Part 2 (Boolean GPT)..."
python3 code/train_part2.py
echo "✓ Part 2 training complete"
echo ""

echo "Step 4: Running demonstrations..."
python3 main.py
echo "✓ Demonstrations complete"
echo ""

echo "=== Pipeline Complete ==="