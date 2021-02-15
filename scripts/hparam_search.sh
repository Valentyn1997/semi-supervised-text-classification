#!/bin/bash
#SBATCH --output=./hparam_search.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1


# !!! Run the command not in Slurm cluster first - in order to load pre-trained weights from transformers library
# Otherwise could produce deadlock.
cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5100:localhost:5000 ubuntu@10.195.1.127
python3 ./runnables/train.py -m +setting=ssl data.path='data/IMDB-clean' exp.task_name=SSL data.labels_list=[neg,pos] exp.logging=True exp.gpus="-1" data.write_to_cache=True data.load_from_cache=False model.threshold=0.9 model.lambda_u=0.01 optimizer.lr=1e-7 exp.check_exisisting_hash=True exp.early_stopping_patience=5000 data.max_seq_length=512 model.max_ul_batch_size_per_gpu=200 model.choose_only_wrongly_predicted_branches=True model.from_scratch=False exp.tsa=False model.tsa_as_threshold=False exp.max_epochs=5000
#python3 ./runnables/train.py -m +setting=supervised data.path='data/IMDB-clean' optimizer.lr=1e-7 exp.task_name=SL data.labels_list=[neg,pos] exp.gpus='-1' optimizer.auto_lr_find=False exp.logging=True exp.max_epochs=1000 data.max_seq_length=512 exp.early_stopping_patience=1000 data.augment=True exp.check_exisisting_hash=True