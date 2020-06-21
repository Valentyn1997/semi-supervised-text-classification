import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import torch
import random
import numpy as np
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def set_seed(args: DictConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, prefix):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    return {
        prefix + '_acc': acc,
        prefix + '_f1_macro': f1_macro,
        prefix + '_f1_micro': f1_micro,
        prefix + '_acc_and_f1': (acc + f1_micro) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        'pearson': pearson_corr,
        'spearmanr': spearman_corr,
        'corr': (pearson_corr + spearman_corr) / 2,
    }


# def compute_metrics(task_name, preds, labels):
#     assert len(preds) == len(labels)
#     if task_name == "SD":
#         return acc_and_f1(preds, labels)
#     else:
#         raise KeyError(task_name)
