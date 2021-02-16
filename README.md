semi-supervised-text-classification
==============================

Semi-supervised text classification based on BERT backbone. The project adapts FixMatch algorithm (https://arxiv.org/abs/2001.07685) by introducing an adaptive weak/strong augmentations selection among 6 basic NLP augmentations (from [nlpaug](https://github.com/makcedward/nlpaug) library):
1. WordEmbsAug (top n similar word random substitutions)
2. BackTranslationAug (back translation)
3. AbstSummAug (abstractive summarization)
4. SynonymAug (random synonims substitution)
5. ContextualWordEmbsAug (contextual word embeddings random substitutions)
6. ContextualWordEmbsForSentenceAug (extra sentence generation)

Project is based on 
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
3. `pip3 install -r requirements.txt`

## MlFlow Server
One can either run their own mlflow server:

`mlflow server --default-artifact-root='/home/ubuntu/semi-supervised-stance-detection/mlruns/'`

or connect to an existing one (LRZ server: 10.195.1.127):

`ssh -N -f -L localhost:5000:localhost:5000 <user>@10.195.1.127`

## Data structure
While running scripts, one should indicate the path to dataset. There are two possible scenarios:

1. **In-topic scenario**. Train/test/validation split is done randomly, without considering the topics. `data.test_id` run argument should be `None` and the files should be structured in the following way:
                      
        ├── data          
        │   ├── <dataset-name>                      <- Dataset name
        |   |   ├── train.tcv                   <- Train labelled data
        |   |   ├── augmentations_labelled      <- Train labelled data augmentations
        |   |   |   ├── SynonymAug.tsv          
        |   |   |   ├── WordEmbsAug.tsv         
        |   |   |   └── ...
        |   |   ├── unlabelled.tsv              <- Train unlabelled data    
        |   |   ├── augmentations_unlabelled    <- Train unlabelled data augmentations
        |   |   |   ├── SynonymAug.tsv          
        |   |   |   ├── WordEmbsAug.tsv         
        |   |   |   └── ...
        |   |   ├── test.tcv                    <- Test data
        |   |   └── val.tcv                     <- Val data
        ...

    To generate augmentations look to [Offline augmentations](#offline-augmentations) section.
    
## Running scripts
### Offline augmentations
To generate offline augmentations for fully-supervised/semi-supervised settings:

`PYTHONPATH=. python3 runnables/generate_augmentations.py`

### Running experiments
All the configurations are in the .yaml format and could be found in the `config/` folder.

Fully-supervised experiments (`config/config.yaml` and `config/setting/supervised.yaml`):

```
PYTHONPATH=. python3 ./runnables/train.py -m +setting=supervised 
        data.path='data/IMDB-clean' 
        optimizer.lr=1e-6 
        exp.task_name=SL 
        data.labels_list=[neg,pos] 
        exp.gpus='-1' 
        exp.logging=True 
        exp.max_epochs=1000 
        data.max_seq_length=512 
        exp.early_stopping_patience=1000 
        data.augment=True
```

```
PYTHONPATH=. python3 ./runnables/train.py -m +setting=supervised 
        data.path='data/in-topic/REVIEWS-clean' 
        optimizer.lr=1e-6 
        exp.task_name=SL3 
        exp.gpus='-1' 
        exp.logging=True 
        exp.max_epochs=1000 
        data.max_seq_length=512 
        exp.early_stopping_patience=1000 
        data.augment=True
```


Semi-supervised setting (`config/config.yaml` and `config/setting/ssl.yaml`):

```
PYTHONPATH=. python3 ./runnables/train.py -m +setting=ssl 
        data.path='data/IMDB-clean' 
        exp.task_name=SSL 
        data.labels_list=[neg,pos] 
        exp.logging=True 
        exp.gpus="-1" 
        model.threshold=0.9 
        model.lambda_u=0.01 
        optimizer.lr=1e-6 
        exp.early_stopping_patience=5000 
        data.max_seq_length=512 
        model.max_ul_batch_size_per_gpu=200 
        model.choose_only_wrongly_predicted_branches=True 
        exp.tsa=False 
        exp.max_epochs=5000
```

```
PYTHONPATH=. python3 ./runnables/train.py -m +setting=ssl 
        data.path='data/in-topic/REVIEWS-clean' 
        exp.task_name=SSL3
        exp.logging=True 
        exp.gpus="-1" 
        model.threshold=0.9 
        model.lambda_u=0.01 
        optimizer.lr=1e-5 
        exp.early_stopping_patience=1000 
        data.max_seq_length=512 
        model.max_ul_batch_size_per_gpu=200 
        model.choose_only_wrongly_predicted_branches=True 
        exp.tsa=False 
        exp.max_epochs=1000
```

### Experiments with Slurm cluster 
Look to `scripts/train.sh` or `scripts/hparam_search.sh`:

`sbatch train.sh` or `sbatch hparam_search.sh`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
