#!/bin/bash
#SBATCH --output=./generate_augmentations1.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1


# !!! Run the command not in Slurm cluster first - in order to load pre-trained weights from transformers library
# Otherwise could produce deadlock.

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
python3 ./runnables/generate_augmentations.py