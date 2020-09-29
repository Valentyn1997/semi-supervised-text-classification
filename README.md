semi-supervised-stance-detection
==============================

Semi-supervised learning for argument mining. Project is based on 
- [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep-learning models
- [Hydra](https://hydra.cc/docs/intro/) - command line arguments managment
- [MlFlow](https://mlflow.org/) - experiments tracking

## Installation
1. Make sure, you have Python 3.7
2. Create a virtual environment:
```console
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
2. `pip3 install -r requirements.txt`

## MlFlow Server
One can either run their own mlflow server:

`mlflow server --default-artifact-root='/home/ubuntu/semi-supervised-stance-detection/mlruns/'`

or connect to an existing one (LRZ server: 10.195.1.127):

`ssh -N -f -L localhost:5000:localhost:5000 <user>@10.195.1.127`

## Data structure
While running scripts, one should indicate the path to dataset. There are two possible scenarios:

1. **In-topic scenario**. Train/test/validation split is done randomly, without considering the topics. `data.test_id` run argument should be `None` and the files should be structured in the following way:
                      
        ├── data          
        │   └── <dataset-name>                      <- Dataset name
        |   │   └── in-topic                        <- In-topic setting / this should be passed to data.path argument
        |   |       ├── train.tcv                   <- Train labelled data
        |   |       ├── augmentations_labelled      <- Train labelled data augmentations
        |   |       |   ├── SynonymAug.tsv          
        |   |       |   ├── WordEmbsAug.tsv         
        |   |       |   └── ...
        |   |       ├── unlabelled.tsv              <- Train unlabelled data    
        |   |       ├── augmentations_unlabelled    <- Train unlabelled data augmentations
        |   |       |   ├── SynonymAug.tsv          
        |   |       |   ├── WordEmbsAug.tsv         
        |   |       |   └── ...
        |   |       ├── test.tcv                    <- Test data
        |   |       └── val.tcv                     <- Val data
        ...

    To generate augmentations look to [Offline augmentations](#offline-augmentations) section.

2. **Cross-topic scenario**. For now, possible only for fully-supervised scenario. One should indicate the topic in `data.test_id` run argument, which would be used as a test subset. Train/val splits are done randomly. The dataset structure is following:

        ├── data          
        │   └── <dataset-name>                      <- Dataset name
        |   │   └── cross-topic                     <- Cross-topic setting / this should be passed to data.path argument
        |   |       └── complete.tcv                <- All data together
        ...
    
## Running scripts
### Offline augmentations
To generate offline augmentations for fully-supervised/semi-supervised settings:

`PYTHONPATH=. python3 runnables/generate_augmentations.py`

### Running experiments
All the configurations are in the .yaml format and could be found in the `config/` folder.

Fully-supervised experiments (`config/config.yaml` and `config/setting/supervised.yaml`):

`PYTHONPATH=. python3 runnables/train.py -m +setting=supervised data.path='data/REVIEWS-clean/in-topic' optimizer.learning_rate=1e-5,1e-6 exp.task_name=SL3 exp.gpus='[0]' optimizer.auto_lr_find=False exp.logging=True exp.early_stopping_patience=10`

`PYTHONPATH=. python3 runnables/train.py -m +setting=supervised data.path='data/UKP-clean/in-topic' optimizer.learning_rate=1e-5,1e-6 exp.task_name=SL2 exp.logging=True`

Semi-supervised setting (`config/config.yaml` and `config/setting/ssl.yaml`):

`PYTHONPATH=. python3 runnables/train.py +setting=ssl data.path='data/REVIEWS-clean/in-topic' optimizer.learning_rate=1e-5 exp.task_name=SSL3 exp.logging=True exp.gpus="2" data.load_from_cache=True`

### Experiments with Slurm cluster 
Look to `scripts/train.sh`:

`sbatch train.sh`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
