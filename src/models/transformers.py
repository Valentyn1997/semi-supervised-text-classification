from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from argparse import Namespace
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from src import MODEL_CLASSES, OUTPUT_MODES
from src.data.processor import PROCESSORS, convert_examples_to_features
from src.data.dataset import AugmentableTextClassificationDataset
from src.utils import acc_and_f1
from torch.utils.data import TensorDataset, DataLoader

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

logger = logging.getLogger(__name__)


class PretrainedTransformer(LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args
        self.lr = None  # Placeholder for auto_lr_find

        self.processor = PROCESSORS[self.args.exp.task_name]()
        self.output_mode = OUTPUT_MODES[self.args.exp.task_name]
        label_list = self.processor.get_labels()
        self.num_labels = len(label_list)

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.args.model.model_type]

        self.config = self.config_class.from_pretrained(self.args.model.config_name if self.args.model.config_name
                                                        else self.args.model.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=self.args.exp.task_name,
                                                        cache_dir=self.args.model.cache_dir if self.args.model.cache_dir
                                                        else None)

        self.tokenizer = self.tokenizer_class.from_pretrained(self.args.model.tokenizer_name if self.args.model.tokenizer_name
                                                              else self.args.model.model_name_or_path,
                                                              do_lower_case=self.args.model.do_lower_case,
                                                              cache_dir=self.args.model.cache_dir if self.args.model.cache_dir
                                                              else None)

        self.model = self.model_class.from_pretrained(self.args.model.model_name_or_path,
                                                      from_tf=bool(".ckpt" in self.args.model.model_name_or_path),
                                                      config=self.config,
                                                      cache_dir=self.args.model.cache_dir if self.args.model.cache_dir else None)
        self.best_model = self.model

        # Will be logged to mlflow
        self.hparams = Namespace(**{
            'lr': self.args.optimizer.learning_rate,
            'model_type': self.args.model.model_type,
            'model_name_or_path': self.args.model.model_name_or_path,
            'task_name': self.args.exp.task_name,
            'test_id': self.args.data.test_id,
            'setting': self.args.data.setting,
            'batch_size': self.args.data.batch_size,
            'gradient_accumulation_steps': self.args.exp.gradient_accumulation_steps,
            'max_epochs': self.args.exp.max_epochs,
            'early_stopping_patience': self.args.exp.early_stopping_patience,
            'max_seq_length': self.args.data.max_seq_length,
            'auto_lr_find': args.optimizer.auto_lr_find
        })

    def prepare_data(self):
        if 'data_size' not in self.args:
            self.train_dataset = self.load_and_cache_examples(mode='train')
            self.test_dataset = self.load_and_cache_examples(mode='test')
            self.val_dataset = self.load_and_cache_examples(mode='val')
            self.hparams.data_size = DictConfig({
                'train': len(self.train_dataset),
                'val': len(self.val_dataset),
                'test': len(self.test_dataset)
            })

    def configure_optimizers(self):
        if self.lr is not None:
            self.hparams.lr = self.lr

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.optimizer.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr,
                          eps=self.args.optimizer.adam_epsilon)
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.optimizer.warmup_steps,
                                                         num_training_steps=self.args.exp.max_steps),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.args.data.batch_size.train, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.args.data.batch_size.val, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.args.data.batch_size.test, num_workers=4)

    def forward(self, batch, batch_idx=None):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        if self.args.model.model_type != "distilbert":  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs["token_type_ids"] = (batch[2] if self.args.model.model_type in ["bert", "xlnet", "albert"] else None)
        return self.model(**inputs)[0]  # Returning only logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch[3]
        loss = F.cross_entropy(logits, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = np.array([x['loss'].item() for x in outputs]).mean()
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0]
        mlflow_metrics = {'train_loss': loss, 'lr': learning_rate}
        return {'loss': loss, 'log': mlflow_metrics}

    def validation_step(self, batch, batch_idx):
        # try:
        results = {}
        logits = self(batch)
        labels = batch[3]
        loss = F.cross_entropy(logits, labels, reduction='mean')
        results['loss'] = loss.detach()
        results['preds'] = logits.detach()
        results['labels'] = batch[3].detach()
        # except StopIteration:
        #     print('StopIteration')
        return results

    test_step = validation_step

    def validation_epoch_end(self, outputs):
        loss_val_mean = torch.tensor(np.array([x['loss'].cpu().numpy() for x in outputs]).mean())
        mlflow_metrics = {'val_loss': loss_val_mean, **self.calculate_metrics(outputs, prefix='val')}
        return {'loss': loss_val_mean, 'log': mlflow_metrics}

    def test_epoch_end(self, outputs):
        loss_test_mean = torch.tensor(np.array([x['loss'].cpu().numpy() for x in outputs]).mean())
        mlflow_metrics = {'test_loss': loss_test_mean, **self.calculate_metrics(outputs, prefix='test')}
        return {'loss': loss_test_mean, 'log': mlflow_metrics}

    @staticmethod
    def calculate_metrics(outputs, prefix):
        preds = np.array([prob for x in outputs for prob in x['preds'].cpu().numpy()])
        preds = np.argmax(preds, axis=1)
        labels = np.array([label for x in outputs for label in x['labels'].cpu().numpy()])
        result = acc_and_f1(preds, labels, prefix=prefix)
        return result

    def load_and_cache_examples(self, mode):
        label_list = self.processor.get_labels()
        examples = getattr(self.processor, f'get_{mode}_examples')(self.args)

        cached_features_file = os.path.join(
            self.args.data.path,
            "cached_{}_{}_{}_{}_{}".format(mode, list(filter(None, self.args.model.model_name_or_path.split("/"))).pop(),
                                           str(self.args.data.max_seq_length), str(self.args.exp.task_name),
                                           str(self.args.data.test_id)),
        )

        if self.args.data.load_from_cache and os.path.exists(cached_features_file):
            logger.info("Loading features from cache file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", self.args.data.path)
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=label_list,
                max_length=self.args.data.max_seq_length,
                output_mode=self.output_mode,
                pad_on_left=bool(self.args.model.model_type in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.args.model.model_type in ["xlnet"] else 0,
            )

        if self.args.data.write_to_cache:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)


class SSLPretrainedTransformer(PretrainedTransformer):
    def __init__(self, args: DictConfig):
        super().__init__(args)

        self.weak_transform = naf.Sometimes([
            nac.RandomCharAug(action="delete"),
            nac.RandomCharAug(action="insert"),
            naw.RandomWordAug()
        ])

        self.strong_transform = naf.Sequential([
            nac.RandomCharAug(action="insert"),
            naw.RandomWordAug()
        ])

        self.hparams.weak_transform = [str(tr) for tr in self.weak_transform]
        self.hparams.strong_transform = [str(tr) for tr in self.strong_transform]
        self.hparams.ssl = dict(args.ssl)

    def prepare_data(self):
        if 'data_size' not in self.args:
            self.train_lab_dataset = self.load_and_cache_examples(mode='train')
            self.train_unlab_dataset = self.load_and_cache_examples(mode='unlab')
            self.test_dataset = self.load_and_cache_examples(mode='test')
            self.val_dataset = self.load_and_cache_examples(mode='val')
            self.hparams.data_size = DictConfig({
                'train': {
                    'lab': len(self.train_lab_dataset),
                    'unlab': len(self.train_unlab_dataset)
                },
                'val': len(self.val_dataset),
                'test': len(self.test_dataset)
            })

    def train_dataloader(self) -> DataLoader:
        train_lab_loader = DataLoader(self.train_lab_dataset, shuffle=True, batch_size=self.args.data.batch_size.train,
                                      num_workers=4)
        self.train_unlab_loader = DataLoader(self.train_unlab_dataset, shuffle=True,
                                             batch_size=self.args.ssl.mu * self.args.data.batch_size.train, num_workers=4)
        self.train_unlab_loader_iterator = iter(self.train_unlab_loader)
        return train_lab_loader

    def training_step(self, batch, batch_idx):
        l_batch = batch
        ul_batch = next(self.train_unlab_loader_iterator)
        ul_batch = [x.type_as(batch[0]) for x in ul_batch]
        uw_batch = ul_batch[0], ul_batch[1], ul_batch[2], ul_batch[6]
        us_batch = ul_batch[3], ul_batch[4], ul_batch[5], ul_batch[6]

        # Supervised loss
        logits_l = self(l_batch)
        labels_l = batch[3]
        loss_l = F.cross_entropy(logits_l, labels_l, reduction='mean')

        # Unsupervised loss
        logits_us = self(us_batch)
        with torch.no_grad():
            logits_uw = self(uw_batch)
            pseudo_label = torch.softmax(logits_uw.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.args.ssl.threshold).float()
        loss_ul = (F.cross_entropy(logits_us, targets_u, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = loss_l + self.args.ssl.lambda_u * loss_ul

        return {'loss': loss, 'loss_l': loss_l, 'loss_ul': loss_ul}

    def training_epoch_end(self, outputs):
        loss = np.array([x['loss'].mean().item() for x in outputs]).mean()
        loss_l = np.array([x['loss_l'].mean().item() for x in outputs]).mean()
        loss_ul = np.array([x['loss_ul'].mean().item() for x in outputs]).mean()
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0]
        mlflow_metrics = {'train_loss': loss, 'train_loss_l': loss_l, 'train_loss_ul': loss_ul, 'lr': learning_rate}
        return {'loss': loss, 'log': mlflow_metrics}

    def on_epoch_end(self) -> None:
        self.train_unlab_loader_iterator = iter(self.train_unlab_loader)

    def load_and_cache_examples(self, mode):
        if mode == 'val' or mode == 'test':
            return super().load_and_cache_examples(mode)
        else:
            label_list = self.processor.get_labels()
            examples = getattr(self.processor, f'get_{mode}_examples')(self.args)

            if mode == 'train':
                return AugmentableTextClassificationDataset(examples,
                                                            weak_transform=self.weak_transform,
                                                            tokenizer=self.tokenizer,
                                                            label_list=label_list,
                                                            max_seq_length=self.args.data.max_seq_length,
                                                            model_type=self.args.model.model_type)

            elif mode == 'unlab':
                return AugmentableTextClassificationDataset(examples,
                                                            weak_transform=self.weak_transform,
                                                            strong_transform=self.strong_transform,
                                                            tokenizer=self.tokenizer,
                                                            label_list=label_list,
                                                            max_seq_length=self.args.data.max_seq_length,
                                                            model_type=self.args.model.model_type)

