#!/bin/bash
#SBATCH --cpus-per-gpu=4 --gpus=a30:1 -t 240:00:00

module purge
conda activate fmperf-env
pip install -r requirements.txt

# Run request generation
echo "Start request generation"
python -u generate_requests.py 1 100 --prompts-file oasst_roots_en_max1000_tokens.jsonl
