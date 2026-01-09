#!/bin/bash
#SBATCH --cpus-per-gpu=4 --gpus=a30:1 -t 240:00:00 --nodelist=gpu02

module purge
conda activate fmperf-env
pip install -r requirements.txt

# Run your experiment
echo "Start experiment"
python -u MIT_experiment.py
