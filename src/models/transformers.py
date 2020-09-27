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
from src.data.dataset import AugmentableTextClassificationDataset, FixMatchCompositeTrainDataset
from src.utils import acc_and_f1
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class PretrainedTransformer(LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.lr = None  # Placeholder for auto_lr_find
        self.hparams = args  # Will be logged to mlflow

        self.processor = PROCESSORS[self.hparams.exp.task_name](load_augmentations=self.hparams.setting == 'ssl' or
                                                                self.hparams.data.augment)
        self.output_mode = OUTPUT_MODES[self.hparams.exp.task_name]
        label_list = self.processor.get_labels()
        self.num_labels = len(label_list)

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.hparams.model.model_type]

        self.config = self.config_class.from_pretrained(self.hparams.model.config_name if self.hparams.model.config_name
                                                        else self.hparams.model.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=self.hparams.exp.task_name,
                                                        cache_dir=self.hparams.model.cache_dir if self.hparams.model.cache_dir
                                                        else None)

        self.tokenizer = self.tokenizer_class.from_pretrained(self.hparams.model.tokenizer_name if self.hparams.model.tokenizer_name
                                                              else self.hparams.model.model_name_or_path,
                                                              do_lower_case=self.hparams.model.do_lower_case,
                                                              cache_dir=self.hparams.model.cache_dir if self.hparams.model.cache_dir
                                                              else None)

        self.model = self.model_class.from_pretrained(self.hparams.model.model_name_or_path,
                                                      from_tf=bool(".ckpt" in self.hparams.model.model_name_or_path),
                                                      config=self.config,
                                                      cache_dir=self.hparams.model.cache_dir if self.hparams.model.cache_dir else None)
        self.best_model = self.model

    def prepare_data(self):
        if 'data_size' not in self.hparams.data:
            self.train_dataset = self.load_and_cache_examples(mode='train')
            self.test_dataset = self.load_and_cache_examples(mode='test')
            self.val_dataset = self.load_and_cache_examples(mode='val')
            self.hparams.data.data_size = DictConfig({
                'train': len(self.train_dataset),
                'val': len(self.val_dataset),
                'test': len(self.test_dataset)
            })

            train_dataloader = self.train_dataloader()
            # Max number of epochs/steps - secondary parameter
            if self.hparams.exp.max_steps > 0:
                self.hparams.exp.max_epochs = (
                            self.hparams.exp.max_steps // (len(train_dataloader) // self.hparams.exp.gradient_accumulation_steps) + 1)
            else:
                self.hparams.exp.max_steps = (
                            len(train_dataloader) // self.hparams.exp.gradient_accumulation_steps * self.hparams.exp.max_epochs)

    def configure_optimizers(self):
        if self.lr is not None:
            self.hparams.optimizer.lr = self.lr

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.optimizer.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.optimizer.lr,
                          eps=self.hparams.optimizer.adam_epsilon)
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.optimizer.warmup_steps,
                                                         num_training_steps=self.hparams.exp.max_steps),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.data.batch_size.train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.test)

    def forward(self, batch, batch_idx=None, model=None):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        if self.hparams.model.model_type != "distilbert":  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs["token_type_ids"] = (batch[2] if self.hparams.model.model_type in ["bert", "xlnet", "albert"] else None)
        if model is None:
            return self.model(**inputs)[0]  # Returning only logits
        elif model == 'best':
            return self.best_model(**inputs)[0]
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if self.hparams.data.augment:  # Using only first augmentation
            batch = batch[0]
        logits = self(batch)
        labels = batch[3]
        loss = F.cross_entropy(logits, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = np.array([x['loss'].mean().item() for x in outputs]).mean()
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0]
        mlflow_metrics = {'train_loss': loss, 'lr': learning_rate}
        return {'loss': loss, 'log': mlflow_metrics}

    def validation_step(self, batch, batch_idx):
        results = {}
        logits = self(batch)
        labels = batch[3]
        loss = F.cross_entropy(logits, labels, reduction='mean')
        results['loss'] = loss.detach()
        results['preds'] = logits.detach()
        results['labels'] = batch[3].detach()
        return results

    def test_step(self, batch, batch_idx):
        results = {}
        logits = self(batch, model='best')
        labels = batch[3]
        loss = F.cross_entropy(logits, labels, reduction='mean')
        results['loss'] = loss.detach()
        results['preds'] = logits.detach()
        results['labels'] = batch[3].detach()
        return results

    def validation_epoch_end(self, outputs):
        loss_val_mean = torch.tensor(np.array([x['loss'].cpu().numpy() for x in outputs]).mean())
        mlflow_metrics = {'val_loss': loss_val_mean, **self.calculate_metrics(outputs, prefix='val')}
        return {'loss': loss_val_mean, 'log': mlflow_metrics}

    def test_epoch_end(self, outputs):
        loss_test_mean = torch.tensor(np.array([x['loss'].mean().cpu().numpy() for x in outputs]).mean())
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
        examples, examples_aug = getattr(self.processor, f'get_{mode}_examples')(self.hparams)

        cached_features_file = os.path.join(
            self.hparams.data.path,
            "cached_{}_{}_{}_{}_{}".format(mode, list(filter(None, self.hparams.model.model_name_or_path.split("/"))).pop(),
                                           str(self.hparams.data.max_seq_length), str(self.hparams.exp.task_name),
                                           str(self.hparams.data.test_id)),
        )

        if self.hparams.data.load_from_cache and os.path.exists(cached_features_file):
            logger.info("Loading features from cache file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating {mode} features from dataset file at {cached_features_file}")
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_list=label_list,
                max_length=self.hparams.data.max_seq_length,
                output_mode=self.output_mode,
                pad_on_left=bool(self.hparams.model.model_type in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.hparams.model.model_type in ["xlnet"] else 0,
                examples_aug=examples_aug
            )

        if self.hparams.data.write_to_cache:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        if examples_aug is None:
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features.values()], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features.values()], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features.values()], dtype=torch.long)
            if self.output_mode == "classification":
                all_labels = torch.tensor([f.label for f in features.values()], dtype=torch.long)
            elif self.output_mode == "regression":
                all_labels = torch.tensor([f.label for f in features.values()], dtype=torch.float)

            return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        else:
            if mode == 'train':
                n_branches = 1
            elif mode == 'unlab':
                n_branches = self.hparams.model.ul_branches
            return AugmentableTextClassificationDataset(list(features.values()), n_branches=n_branches)


class SSLPretrainedTransformer(PretrainedTransformer):

    def prepare_data(self):
        super().prepare_data()
        self.hparams.data.data_size = DictConfig({
            'train': {
                'lab': len(self.train_dataset.l_dataset),
                'unlab': len(self.train_dataset.ul_dataset)
            },
            'val': len(self.val_dataset),
            'test': len(self.test_dataset)
        })

    def training_step(self, composite_batch, batch_idx):
        # Batch collation
        l_batch = composite_batch[0][0][0]
        ul_branches = [[torch.cat(sub_item) for sub_item in zip(*item)] for item in zip(*composite_batch[1])]

        # Supervised loss
        l_logits = self(l_batch)
        l_labels = l_batch[3]
        l_loss = F.cross_entropy(l_logits, l_labels, reduction='mean')

        # Unsupervised loss
        # Choosing pseudo-labels and branches to back-propagate
        u_max_probs_2d = torch.empty((len(ul_branches), len(ul_branches[0][0]))).type_as(l_loss)
        u_targets_2d = torch.empty((len(ul_branches), len(ul_branches[0][0]))).type_as(l_loss)
        with torch.no_grad():
            for i, ul_branch in enumerate(ul_branches):
                u_logits = self(ul_branch)
                pseudo_labels = torch.softmax(u_logits.detach(), dim=-1)
                u_max_probs_2d[i], u_targets_2d[i] = torch.max(pseudo_labels, dim=-1)
        mask_2d = u_max_probs_2d.ge(self.hparams.model.threshold).int()
        mask = (mask_2d.sum(dim=0) > 1)  # Threshold mask per instance, at least two branches should pass the threshold
        u_max_probs, u_best_branches = torch.max(u_max_probs_2d, dim=0)

        u_loss = torch.tensor(0.0).type_as(l_loss)
        if mask.int().sum() > 0:
            # Creating one batch for unlabelled loss
            u_batch = []
            u_targets = []
            for i in range(len(ul_branches[0][0])):
                if mask[i]:
                    nonmax_branches = [ul_branch for (ind, ul_branch) in enumerate(ul_branches) if ind != u_best_branches[i] and
                                   bool(mask_2d[ind, i])]
                    u_batch.extend([[item[i] for item in branch] for branch in nonmax_branches])
                    u_targets.extend(u_targets_2d[u_best_branches[i]][i].repeat(len(nonmax_branches)))
            u_batch = [torch.stack(item) for item in zip(*u_batch)]
            u_targets = torch.stack(u_targets).long()

            # Unlabelled loss
            u_logits = self(u_batch)
            u_loss = F.cross_entropy(u_logits, u_targets, reduction='mean')

        # Train loss / labelled accuracy
        loss = l_loss + self.hparams.model.lambda_u * u_loss

        return {'loss': loss, 'l_loss': l_loss, 'u_loss': u_loss}

    def training_epoch_end(self, outputs):
        loss = np.array([x['loss'].mean().item() for x in outputs]).mean()
        loss_l = np.array([x['l_loss'].mean().item() for x in outputs]).mean()
        loss_ul = np.array([x['u_loss'].mean().item() for x in outputs]).mean()
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0]
        mlflow_metrics = {'train_loss': loss, 'train_loss_l': loss_l, 'train_loss_ul': loss_ul, 'lr': learning_rate}
        return {'loss': loss, 'log': mlflow_metrics}

    def load_and_cache_examples(self, mode):
        if mode == 'val' or mode == 'test':
            return super().load_and_cache_examples(mode)
        elif mode == 'train':
            train_l = super().load_and_cache_examples('train')
            train_ul = super().load_and_cache_examples('unlab')
            return FixMatchCompositeTrainDataset(train_l, train_ul, self.hparams.model.mu)
