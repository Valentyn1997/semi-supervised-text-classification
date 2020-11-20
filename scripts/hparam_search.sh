#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5001:localhost:5000 ubuntu@10.195.1.127
python3 ./runnables/train.py -m +setting=ssl data.path='data/REVIEWS-clean/in-topic' exp.task_name=SSL3 exp.logging=True exp.gpus="-1" data.write_to_cache=False data.load_from_cache=True model.threshold=0.99,0.95,0.9 model.lambda_u=0.1,0.01,0.5 optimizer.lr=1e-05,2e-05 exp.check_exisisting_hash=True

