from torch.utils.data import Dataset
from src.data.processor import convert_examples_to_features
from typing import List
import nlpaug.flow as naf
import torch
from copy import deepcopy

from transformers.data.processors import InputExample


class AugmentableTextClassificationDataset(Dataset):
    def __init__(self, data: List[InputExample], tokenizer, label_list, max_seq_length, model_type,
                 weak_transform: naf.Pipeline = None, strong_transform: naf.Pipeline = None):
        self.data = data
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.model_type = model_type

    def __getitem__(self, index):
        example = self.data[index]
        example_copy = deepcopy(example)

        # Transformation
        if self.weak_transform is not None:
            example.text_a = self.weak_transform.augment(example.text_a)

        # Text to features
        features = convert_examples_to_features(
            [example],
            self.tokenizer,
            label_list=self.label_list,
            max_length=self.max_seq_length,
            output_mode='classification',
            pad_on_left=bool(self.model_type in ["xlnet"]),
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
        )[0]

        # Features to tensors
        input_ids = torch.tensor(features.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(features.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(features.token_type_ids, dtype=torch.long)
        label = torch.tensor(features.label, dtype=torch.long)

        if self.strong_transform is not None:
            example_copy.text_a = self.strong_transform.augment(example_copy.text_a)
            strong_features = convert_examples_to_features(
                [example_copy],
                self.tokenizer,
                label_list=self.label_list,
                max_length=self.max_seq_length,
                output_mode='classification',
                pad_on_left=bool(self.model_type in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
            )[0]

            # Features to tensors
            s_input_ids = torch.tensor(strong_features.input_ids, dtype=torch.long)
            s_attention_mask = torch.tensor(strong_features.attention_mask, dtype=torch.long)
            s_token_type_ids = torch.tensor(strong_features.token_type_ids, dtype=torch.long)

            return input_ids, attention_mask, token_type_ids, s_input_ids, s_attention_mask, s_token_type_ids, label

        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.data)