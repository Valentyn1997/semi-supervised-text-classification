from omegaconf import DictConfig
import logging
import os
from transformers.data.processors import InputFeatures, DataProcessor
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

from src import OUTPUT_MODES

logger = logging.getLogger(__name__)


class SupervisedTwoLabelProcessor(DataProcessor):
    """Processor for the argument mining supervised data set."""

    def __init__(self, load_augmentations=True):
        super().__init__()
        self.mask = None  # Used for train/val split for cross-topic setting
        self.load_augmentations = load_augmentations

    def get_train_examples(self, args: DictConfig):  # train labelled examples
        if args.data.test_id is None:
            df = self.read_tsv(os.path.join(args.data.path, "train.tsv"))
            if self.load_augmentations:
                aug_dfs = self.read_tsvs(f'{args.data.path}/augmentations_labelled')
                aug_df = pd.concat((aug_df.assign(aug_name=aug_name) for aug_name, aug_df in aug_dfs.items()))
                return self._create_examples(df, aug_df)
            else:
                return self._create_examples(df)
        else:
            df = self.read_tsv(os.path.join(args.data.path, "complete.tsv"))
            df_train_set = df[df["topic"] != args.data.test_id]
            if self.mask is None:
                train_indices = np.random.choice(range(len(df_train_set)),
                                                 int((1 - args.data.validation_size) * len(df_train_set)),
                                                 replace=False)
                self.mask = np.zeros(len(df_train_set), dtype=int)
                self.mask[train_indices] = 1
            df_train_set = df_train_set[self.mask.astype(bool)]
            return self._create_examples(df_train_set)

    def get_unlab_examples(self, args: DictConfig):
        df = self.read_tsv(os.path.join(args.data.path, "unlabelled.tsv"))
        if self.load_augmentations:
            aug_dfs = self.read_tsvs(f'{args.data.path}/augmentations_unlabelled')
            aug_df = pd.concat((aug_df.assign(aug_name=aug_name) for aug_name, aug_df in aug_dfs.items()))
            return self._create_examples(df, aug_df)
        else:
            return self._create_examples(df)

    def get_test_examples(self, args: DictConfig):
        if args.data.test_id is None:
            df = self.read_tsv(os.path.join(args.data.path, "test.tsv"))
            return self._create_examples(df)
        else:
            df = self.read_tsv(os.path.join(args.data.path, "complete.tsv"))
            df_test_set = df[df["topic"] == args.data.test_id]
            return self._create_examples(df_test_set)

    def get_val_examples(self, args: DictConfig):
        if args.data.test_id is None:
            df = self.read_tsv(os.path.join(args.data.path, "val.tsv"))
            return self._create_examples(df)
        else:
            df = self.read_tsv(os.path.join(args.data.path, "complete.tsv"))
            df_train_set = df[df["topic"] != args.data.test_id]
            if self.mask is None:
                val_indices = np.random.choice(range(len(df_train_set)),
                                               int(args.data.validation_size * len(df_train_set)),
                                               replace=False)
                self.mask = np.ones(len(df_train_set), dtype=int)
                self.mask[val_indices] = 0
            df_train_set = df_train_set[~self.mask.astype(bool)]
            return self._create_examples(df_train_set)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["Argument_for", "Argument_against"]

    @staticmethod
    def read_tsv(input_file):
        df = pd.read_csv(input_file, sep="\t")
        df = df.replace(np.nan, '', regex=True)
        df["topic"] = df["topic"].apply(lambda x: x.replace(' ', '_'))
        return df

    @staticmethod
    def read_tsvs(input_dir):
        dfs = {}
        input_files = glob.glob(f'{input_dir}/*')
        for input_file in input_files:
            name = input_file.split('/')[-1].split('.')[0]
            dfs[name] = pd.read_csv(input_file, sep="\t")
            dfs[name] = dfs[name].replace(np.nan, '', regex=True)
        return dfs

    @staticmethod
    def _create_examples(df, aug_df=None):
        """Creates examples for the training and test sets."""
        df = df[df['annotation'] != "NoArgument"]
        if aug_df is not None:
            aug_df = aug_df[aug_df['id'].isin(df.id)]
        return df, aug_df

    # @staticmethod
    # def _create_example(row, aug_df=None):
    #     guid = row['id']
    #     text_a = row["sentence"]
    #     text_b = row["topic"]
    #     label = row["annotation"]
    #     if aug_df is not None:
    #         augmentations = aug_df[aug_df.id == guid]
    #         return dict(guid=guid, text_a=text_a, text_b=text_b, label=label, augmentations=augmentations)
    #     else:
    #         return dict(guid=guid, text_a=text_a, text_b=text_b, label=label)


class SupervisedThreeLabelProcessor(SupervisedTwoLabelProcessor):
    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["Argument_for", "Argument_against", "NoArgument"]

    @staticmethod
    def _create_examples(df, aug_df=None):
        """Creates examples for the training and test sets"""
        return df, aug_df


PROCESSORS = {
    "SL2": SupervisedTwoLabelProcessor,  # fully-supervised setting, 2 labels: "Argument_for", "Argument_against"
    "SL3": SupervisedThreeLabelProcessor,  # fully-supervised setting, 3 labels: "Argument_for", "Argument_against", "NoArgument"
    "SSL2": SupervisedTwoLabelProcessor,  # ssl setting, 2 labels: "Argument_for", "Argument_against" + unlabelled subset
    "SSL3": SupervisedThreeLabelProcessor  # ssl setting, 3 labels: "Argument_for", "Argument_against", "NoArgument" +
}


def convert_examples_to_features(examples: pd.DataFrame,
                                 tokenizer, max_length=512,
                                 task=None, label_list=None, output_mode=None,
                                 pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, examples_aug: pd.DataFrame = None) -> dict:
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet
        where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    if task is not None:
        processor = PROCESSORS[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = OUTPUT_MODES[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['UNL'] = -1

    features = {}

    # Reduced dataset
    # examples = examples.head(4000)
    # if examples_aug is not None:
    #     examples_aug = examples_aug[examples_aug.id.isin(examples.id)]

    for (_, example) in tqdm(examples.iterrows(), total=len(examples)):
        features[example['id']] = preprocess_example(example['sentence'], example['topic'], example['annotation'], tokenizer,
                                                     mask_padding_with_zero, max_length, pad_on_left,
                                                     pad_token, pad_token_segment_id, output_mode, label_map)
    extended_features = {}
    if examples_aug is not None:
        grouped_aug = examples_aug.groupby('id')
        for id, group in tqdm(grouped_aug, total=len(grouped_aug)):
            features_aug = []
            for (_, example_aug) in group.iterrows():
                label = examples[examples.id == id].iloc[0].annotation
                features_aug.append((example_aug['aug_name'],
                                    preprocess_example(example_aug['sentence'], '', label,
                                                       tokenizer, mask_padding_with_zero, max_length, pad_on_left,
                                                       pad_token, pad_token_segment_id, output_mode, label_map)))
            extended_features[id] = (features[id], features_aug)

        return extended_features

    else:
        return features


def preprocess_example(text_a, text_b, label, tokenizer, mask_padding_with_zero, max_length, pad_on_left, pad_token,
                       pad_token_segment_id, output_mode, label_map):
    inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length,
                                   truncation=True)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert (len(attention_mask) == max_length), "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert (len(token_type_ids) == max_length), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if output_mode == "classification":
        label = label_map[label]
    elif output_mode == "regression":
        label = float(label)
    else:
        raise KeyError(output_mode)
    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)
