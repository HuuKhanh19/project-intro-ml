#!/bin/bash

# Script to run a single experiment
# Usage: ./scripts/experiments/run_experiment.sh exp01_densenet121_weighted_ce

set -e  # Exit on error

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_id>"
    echo "Example: $0 exp01_densenet121_weighted_ce"
    echo ""
    echo "Available experiments:"
    echo "  exp01_densenet121_weighted_ce"
    echo "  exp02_densenet121_focal"
    echo "  exp03_efficientnet_b0_weighted_ce"
    echo "  exp04_efficientnet_b0_focal"
    echo "  exp05_lenet_weighted_ce"
    echo "  exp06_lenet_focal"
    echo "  exp07_mlp_weighted_ce"
    echo "  exp08_mlp_focal"
    exit 1
fi

EXPERIMENT_ID=$1

echo "======================================================================"
echo "Running Experiment: $EXPERIMENT_ID"
echo "======================================================================"
echo ""

# Train model
echo ">>> TRAINING <<<"
python scripts/experiments/train.py --experiment $EXPERIMENT_ID

echo ""
echo ">>> EVALUATION <<<"

# Evaluate on test set
CHECKPOINT_PATH="checkpoints/${EXPERIMENT_ID}/checkpoint_best.pth"

if [ -f "$CHECKPOINT_PATH" ]; then
    python scripts/experiments/evaluate.py \
        --checkpoint $CHECKPOINT_PATH \
        --split test
    
    echo ""
    echo "======================================================================"
    echo "✓ Experiment completed: $EXPERIMENT_ID"
    echo "======================================================================"
    echo "Results:"
    echo "  Checkpoints: checkpoints/${EXPERIMENT_ID}/"
    echo "  Evaluation: checkpoints/${EXPERIMENT_ID}/evaluation/"
    echo "  TensorBoard: tensorboard --logdir checkpoints/${EXPERIMENT_ID}/logs"
    echo ""
else
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi