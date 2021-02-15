#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5100:localhost:5000 ubuntu@10.195.1.127
#python3 ./runnables/train.py -m +setting=ssl data.path='data/REVIEWS-clean/in-topic' exp.task_name=SSL3 exp.logging=True exp.gpus="-1" data.write_to_cache=False data.load_from_cache=True model.threshold=0.99 model.lambda_u=0.1 optimizer.lr=1e-05 optimizer.weight_decay=0.0 exp.early_stopping_patience=500
python3 ./runnables/train.py -m +setting=supervised data.path='data/REVIEWS-clean/in-topic' optimizer.lr=1e-5,1e-6 exp.task_name=SL3 exp.gpus='-1' data.num_labelled_train=600,300 optimizer.auto_lr_find=False exp.logging=True exp.early_stopping_patience=500 exp.max_epochs=500 data.augment=True data.balance_labelled=True,False