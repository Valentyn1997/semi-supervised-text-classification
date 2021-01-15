#!/bin/bash
#SBATCH --output=./generate_augmentations1.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
python3 ./runnables/generate_augmentations.py