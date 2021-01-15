import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import hashlib
from copy import deepcopy
import random
import numpy as np
from omegaconf import DictConfig
import torch
import math


logger = logging.getLogger(__name__)


def set_seed(args: DictConfig):
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.exp.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean()


def acc_and_f1(preds, labels, prefix):
    acc = simple_accuracy(preds, labels)
    f1_macro = torch.tensor(f1_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="macro"),
                            device=preds.device)
    f1_micro = torch.tensor(f1_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="micro"),
                            device=preds.device)
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


def calculate_hash(args: DictConfig):
    args_copy = deepcopy(args)
    args_copy.data.pop('path')
    return hashlib.md5(str(args_copy).encode()).hexdigest()



class TrainingSignalAnnealing(object):
    def __init__(self, num_targets: int, total_steps: int):
        self.num_targets = num_targets
        self.total_steps = total_steps
        self.current_step = 1

    def step(self):
        self.current_step += 1

    @property
    def threshold(self):
        alpha_t = math.exp((self.current_step / self.total_steps - 1) * 5)
        return alpha_t * (1 - 1 / self.num_targets) + 1 / self.num_targets

