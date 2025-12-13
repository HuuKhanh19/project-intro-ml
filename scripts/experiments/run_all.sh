#!/bin/bash

# Script to run all experiments in order
# Runs 8 experiments: 4 models × 2 loss functions

set -e  # Exit on error

echo "======================================================================"
echo "RUNNING ALL EXPERIMENTS"
echo "4 Models × 2 Loss Functions = 8 Experiments"
echo "======================================================================"
echo ""

# List of experiments in priority order
EXPERIMENTS=(
    "exp01_densenet121_weighted_ce"
    "exp02_densenet121_focal"
    "exp03_efficientnet_b0_weighted_ce"
    "exp04_efficientnet_b0_focal"
    "exp05_lenet_weighted_ce"
    "exp06_lenet_focal"
    "exp07_mlp_weighted_ce"
    "exp08_mlp_focal"
)

TOTAL=${#EXPERIMENTS[@]}
COUNTER=1

# Run each experiment
for exp_id in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[$COUNTER/$TOTAL] Running: $exp_id"
    echo "======================================================================"
    echo ""
    
    # Run experiment
    ./scripts/experiments/run_experiment.sh $exp_id
    
    echo ""
    echo "✓ Completed experiment $COUNTER/$TOTAL: $exp_id"
    echo ""
    
    COUNTER=$((COUNTER + 1))
    
    # Sleep to cool down GPU (optional)
    sleep 5
done

echo ""
echo "======================================================================"
echo "✓ ALL EXPERIMENTS COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in: checkpoints/"
echo ""
echo "Compare all results:"
echo "  python scripts/experiments/compare_results.py"
echo ""
echo "View all TensorBoard logs:"
echo "  tensorboard --logdir checkpoints/"
echo ""