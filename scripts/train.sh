#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --gres=gpu:4

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
#module load cuda42/toolkit
ssh -N -f -L localhost:5000:localhost:5000 ubuntu@10.195.1.127
python3 ./runnables/train_ssl.py +setting=ssl data.path='data/REVIEWS-clean/in-topic' optimizer.lr=1e-5 exp.task_name=SL3 optimizer.auto_lr_find=False exp.logging=True exp.gpus="-1" exp.early_stopping_patience=10 data.load_from_cache=True

