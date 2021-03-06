from os.path import abspath, dirname
import logging

from transformers import (WEIGHTS_NAME, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, BertConfig,
                          BertForSequenceClassification, BertTokenizer, DistilBertConfig,
                          DistilBertForSequenceClassification, DistilBertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, XLMConfig,
                          XLMForSequenceClassification, XLMRobertaConfig,
                          XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer)

logging.basicConfig(level=logging.INFO)

ROOT_PATH = dirname(dirname(abspath(__file__)))
CONFIG_PATH = f'{ROOT_PATH}/config'
RESOURCES_PATH = f'{ROOT_PATH}/resources'

MLFLOW_URI = 'http://127.0.0.1:5100'

ALL_CONFIGS = (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig, AlbertConfig, XLMRobertaConfig)
ALL_MODELS = sum((tuple(conf.model_type) for conf in ALL_CONFIGS), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer)
}
