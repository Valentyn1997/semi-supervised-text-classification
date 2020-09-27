from torch.utils.data import Dataset
from src.data.processor import convert_examples_to_features
from typing import List, Dict, Tuple
from transformers.data.processors import InputFeatures
import torch
import math
from copy import deepcopy
import numpy as np
from sklearn import preprocessing
import pandas as pd
from transformers.data.processors import InputExample


class AugmentableTextClassificationDataset(Dataset):
    def __init__(self, data: List[Tuple[InputFeatures, List[Tuple[str, InputFeatures]]]], n_branches=2):
        self.data = data
        self.n_branches = n_branches
        self.aug_encoder = preprocessing.LabelEncoder()

        # Fitting aug_encoder
        _, example_aug = self.data[0]
        example_aug = np.array(example_aug)
        aug_types = np.unique(example_aug[:, 0])
        self.aug_encoder.fit(aug_types)

    def __getitem__(self, index):
        _, example_aug = self.data[index]
        example_aug = np.array(example_aug)
        aug_types = np.unique(example_aug[:, 0])

        assert self.n_branches <= len(aug_types)

        aug_sample = np.random.choice(aug_types, size=self.n_branches, replace=False)

        result = []
        for aug_name in aug_sample:
            example = np.random.choice(example_aug[:, 1][example_aug[:, 0] == aug_name])

            input_ids = torch.tensor(example.input_ids, dtype=torch.long)
            attention_mask = torch.tensor(example.attention_mask, dtype=torch.long)
            token_type_ids = torch.tensor(example.token_type_ids, dtype=torch.long)
            label = torch.tensor(example.label, dtype=torch.long)
            aug_name_label = torch.tensor(self.aug_encoder.transform([aug_name])[0], dtype=torch.int)
            result.append((input_ids, attention_mask, token_type_ids, label, aug_name_label))
        return tuple(result)

    def __len__(self):
        return len(self.data)


class FixMatchCompositeTrainDataset(Dataset):
    def __init__(self, l_dataset: AugmentableTextClassificationDataset, ul_dataset: AugmentableTextClassificationDataset,
                 mu: int, len_mode='min'):
        self.l_dataset = l_dataset
        self.ul_dataset = ul_dataset
        self.mu = mu

        self.l_indexes = []
        self.ul_indexes = []
        self.len = eval(len_mode)(len(self.l_dataset), len(self.ul_dataset) // self.mu)
        self.construct_indices()

    def __getitem__(self, i):
        return [self.l_dataset[ind] for ind in self.l_indexes[i]], [self.ul_dataset[ind] for ind in self.ul_indexes[i]]

    def construct_indices(self):
        l_indices = np.arange(0, len(self.l_dataset))
        ul_indices = np.arange(0, len(self.ul_dataset))

        l_n_repeats = math.ceil(self.len / len(l_indices))
        ul_n_repeats = math.ceil(self.len * self.mu / len(ul_indices))

        l_indices_repeated = np.tile(l_indices, l_n_repeats).reshape(l_n_repeats, l_indices.size)
        ul_indices_repeated = np.tile(ul_indices, ul_n_repeats).reshape(ul_n_repeats, ul_indices.size)

        self.l_indexes = np.array(list(map(np.random.permutation, l_indices_repeated))).flatten()[:self.len]
        self.ul_indexes = np.array(list(map(np.random.permutation, ul_indices_repeated))).flatten()[:self.len * self.mu]

        self.l_indexes = self.l_indexes.reshape(self.len, 1)
        self.ul_indexes = self.ul_indexes.reshape(self.len, self.mu)

    def __len__(self):
        # will be called every epoch
        self.construct_indices()
        return self.len
