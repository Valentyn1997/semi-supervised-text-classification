#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --gres=gpu:4

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
#module load cuda42/toolkit
ssh -N -f -L localhost:5000:localhost:5000 ubuntu@10.195.1.127
python3 ./runnables/train.py -m +setting=ssl data.path='data/REVIEWS-clean/in-topic' exp.task_name=SSL3 exp.logging=True exp.gpus="-1" data.load_from_cache=True model.threshold=0.95,0.75,0.65 model.lambda_u=0.1,0.5,1.0 optimizer.lr=1e-5,2e-5 optimizer.weight_decay=0.0,1e-5

