#!/bin/bash
# Wrapper to run any Python script in the entrope conda environment.
# Handles the GLIBCXX version mismatch on this system.
#
# Usage:
#   bash scripts/run.sh train_entropy_model.py [args...]
#   bash scripts/run.sh run_longExp.py [args...]
#
# Or source this file in other scripts:
#   source scripts/run.sh   (sets PYTHON and LD_PRELOAD vars)

CONDA_ENV="/home/AD/sachith/.conda/envs/entrope"
export LD_PRELOAD="${CONDA_ENV}/lib/libstdc++.so.6"
export PYTHON="${CONDA_ENV}/bin/python"

if [ $# -gt 0 ]; then
    exec $PYTHON "$@"
fi
