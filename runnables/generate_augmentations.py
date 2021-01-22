from src import ROOT_PATH, RESOURCES_PATH
from os import mkdir
from os.path import exists

import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from src.data.custom_augmentors import BatchBackTranslationAug, BatchAbstSummAug

import itertools
from tqdm import tqdm
tqdm.pandas()

# Reading data
DATA_PATH = f'{ROOT_PATH}/data/IMDB-clean'
OUT_PATH = {
    'labelled': f'{DATA_PATH}/augmentations_labelled',
    'unlabelled': f'{DATA_PATH}/augmentations_unlabelled'
}
for out_path in OUT_PATH.values():
    mkdir(out_path) if not exists(out_path) else None

print(f'Reading original data from {DATA_PATH}')

original_dfs = {
    'labelled': pd.read_csv(f'{DATA_PATH}/train.tsv', sep='\t'),
    'unlabelled': pd.read_csv(f'{DATA_PATH}/unlabelled.tsv', sep='\t')
}

# Params,
augmentation_list = [
    # naw.WordEmbsAug,
    # BatchBackTranslationAug,
    BatchAbstSummAug,
    # naw.SynonymAug,
    # naw.ContextualWordEmbsAug,
    # nas.ContextualWordEmbsForSentenceAug
]

configs = {
    'BatchBackTranslationAug': {
        'n_times': {
            'labelled': 1,
            'unlabelled': 1
        },
        'conf': {
            'model_names': [('transformer.wmt19.en-de', 'transformer.wmt19.de-en'),
                            ('transformer.wmt19.en-ru', 'transformer.wmt19.ru-en')],
            'from_num_beam': [1, 3, 5],
            'to_num_beam': [1, 3, 5]
        },
        'type': 'batch',
    },
    'WordEmbsAug': {
        'n_times': {
            'labelled': 20,
            'unlabelled': 2
        },
        'conf': {
            'model_type': ['word2vec'],
            'model_path': [f'{RESOURCES_PATH}/GoogleNews-vectors-negative300.bin'],
            'top_k': [50],
            'aug_p': [0.3],
            'action': ['substitute']
        },
        'type': 'parallel',
    },
    'SynonymAug': {
        'n_times': {
            'labelled': 5,
            'unlabelled': 1
        },
        'conf': {
            'aug_src': ['wordnet'],
            'aug_p': [0.3, 0.5, 0.8, 0.99],
        },
        'type': 'single',
    },
    'BatchAbstSummAug': {
        'n_times': {
            'labelled': 1,  # Deterministic transfromation
            'unlabelled': 1
        },
        'conf': {
            'model_path': ['t5-large', 't5-base'],
            'num_beam': [1, 3, 5],
            'max_length': [0.2, 0.5, 0.8],
        },
        'type': 'batch',
    },
    'ContextualWordEmbsAug': {
        'n_times': {
            'labelled': 10,
            'unlabelled': 1
        },
        'conf': {
            'model_path': ['bert-base-uncased', 'xlnet-base-cased'],
            'top_k': [50],
            'aug_p': [0.3],
            'action': ['substitute']
        },
        'type': 'single',
    },
    'ContextualWordEmbsForSentenceAug': {
            'n_times': {
                'labelled': 10,
                'unlabelled': 1
            },
            'conf': {
                'model_path': ['gpt2', 'xlnet-base-cased'],
                'top_k': [50],
            },
            'type': 'single',
        },

}

for Augmentation in augmentation_list:
    print(Augmentation.__name__)
    configs_list = [dict(zip(configs[Augmentation.__name__]['conf'].keys(), values))
                    for values in itertools.product(*configs[Augmentation.__name__]['conf'].values())]
    print(configs_list)

    # Augmenting data
    for source in ['unlabelled']:
        augmented_df = {}
        # original_dfs[source]['sentence'] = original_dfs[source].sentence.astype(str).str.replace('\D+', '')
        for i in (range(configs[Augmentation.__name__]['n_times'][source])):
            for config in configs_list:
                aug = Augmentation(**config)

                if configs[Augmentation.__name__]['type'] == 'parallel':
                    augmented_df[(i, str(config))] = pd.DataFrame(aug.augment(list(original_dfs[source].sentence),
                                                                               num_thread=10),
                                                                  columns=['sentence'], index=original_dfs[source].id)
                elif configs[Augmentation.__name__]['type'] == 'single':
                    augmented_df[(i, str(config))] = pd.DataFrame(list(original_dfs[source].sentence.apply(aug.augment)),
                                                                  columns=['sentence'], index=original_dfs[source].id)
                elif configs[Augmentation.__name__]['type'] == 'batch':
                    augmented_df[(i, str(config))] = pd.DataFrame(aug.batch_augments(list(original_dfs[source].sentence)),
                                                                  columns=['sentence'], index=original_dfs[source].id)

        augmented_df = pd.concat(augmented_df, keys=augmented_df.keys(), names=['n', 'params']).reset_index()
        augmented_df.to_csv(f'{OUT_PATH[source]}/{Augmentation.__name__}.tsv', sep='\t', index=False)
