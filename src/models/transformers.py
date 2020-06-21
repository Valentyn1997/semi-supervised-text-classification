from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback
from omegaconf import DictConfig
import torch
import os
import logging
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.loggers import MLFlowLogger

from src import MODEL_CLASSES, OUTPUT_MODES, DATA_PATH
from src.data.processor import PROCESSORS, convert_examples_to_features
from src.utils import acc_and_f1
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class PretrainedTransformer(LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args

        self.processor = PROCESSORS[args.task_name]()
        self.output_mode = OUTPUT_MODES[args.task_name]
        label_list = self.processor.get_labels()
        self.num_labels = len(label_list)

        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model.model_type]

        self.config = self.config_class.from_pretrained(args.model.config_name if args.model.config_name
                                                        else args.model.model_name_or_path,
                                                        num_labels=self.num_labels,
                                                        finetuning_task=args.task_name,
                                                        cache_dir=args.model.cache_dir if args.model.cache_dir else None)

        self.tokenizer = self.tokenizer_class.from_pretrained(args.model.tokenizer_name if args.model.tokenizer_name
                                                              else args.model.model_name_or_path,
                                                              do_lower_case=args.model.do_lower_case,
                                                              cache_dir=args.model.cache_dir if args.model.cache_dir else None)

        self.model = self.model_class.from_pretrained(args.model.model_name_or_path,
                                                      from_tf=bool(".ckpt" in args.model.model_name_or_path),
                                                      config=self.config,
                                                      cache_dir=args.model.cache_dir if args.model.cache_dir else None)
        # Will be logged to mlflow
        self.hparams = {
            'model_type': self.args.model.model_type,
            'model_name_or_path': self.args.model.model_name_or_path,
            'task_name': self.args.task_name,
            'test_id': self.args.data.test_id,
            'setting': 'In-Topic' if self.args.data.test_id is None else 'Cross-Topic',
            'batch_size': self.args.data.batch_size,
            'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
            'num_train_epochs': self.args.num_train_epochs,
            'early_stopping_patience': self.args.early_stopping_patience,
        }

    def prepare_data(self):
        if 'data_size' not in self.hparams:
            self.train_dataset = self.load_and_cache_examples(evaluate=False, validate=False)
            self.test_dataset = self.load_and_cache_examples(evaluate=True)
            self.val_dataset = self.load_and_cache_examples(evaluate=False, validate=True)
            self.hparams['data_size'] = {
                'train': len(self.train_dataset),
                'val': len(self.val_dataset),
                'test': len(self.test_dataset)
            }

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.optimizer.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.optimizer.learning_rate,
                          eps=self.args.optimizer.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args.optimizer.warmup_steps,
                                                    num_training_steps=self.args.t_total)

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args.model.model_name_or_path, "optimizer.pt")) \
                and os.path.isfile(os.path.join(self.args.model.model_name_or_path, "scheduler.pt")):
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model.model_name_or_path, "scheduler.pt")))

        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.data.batch_size.train, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        val_sampler = SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, sampler=val_sampler, batch_size=self.args.data.batch_size.val, num_workers=4)

    # def test_dataloader(self) -> DataLoader:
    #     test_sampler = SequentialSampler(self.test_dataset)
    #     return DataLoader(self.val_dataset, sampler=test_sampler, batch_size=self.args.data.batch_size.test)

    def forward(self, x):
        inputs = {"input_ids": x[0], "attention_mask": x[1], "labels": x[3]}
        if self.args.model.model_type != "distilbert":  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs["token_type_ids"] = (x[2] if self.args.model.model_type in ["bert", "xlnet", "albert"] else None)
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_train_mean = np.array([x['loss'].item() for x in outputs]).mean()
        learning_rate = self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0]
        mlflow_metrics = {'train_loss': loss_train_mean, 'lr': learning_rate}
        return {'loss': loss_train_mean, 'log': mlflow_metrics}

    def validation_step(self, batch, batch_idx):
        results = {}
        outputs = self(batch)
        loss, logits = outputs[:2]
        results['loss'] = loss.detach().cpu().numpy()
        results['preds'] = logits.detach().cpu().numpy()
        results['labels'] = batch[3].detach().cpu().numpy()
        return results

    def validation_epoch_end(self, outputs):
        loss_val_mean = torch.tensor(np.array([x['loss'] for x in outputs]).mean())
        preds = np.array([prob for x in outputs for prob in x['preds']])
        preds = np.argmax(preds, axis=1)
        labels = np.array([label for x in outputs for label in x['labels']])
        result = acc_and_f1(preds, labels, prefix='val')
        mlflow_metrics = {'val_loss': loss_val_mean, **result}
        return {'loss': loss_val_mean, 'log': mlflow_metrics}

    def load_and_cache_examples(self, evaluate=False, validate=False):
        # Load data features from cache or dataset file
        if validate and not evaluate:
            cached_features_file = os.path.join(
                DATA_PATH,
                "cached_{}_{}_{}_{}_{}".format(
                    "valid",
                    list(filter(None, self.args.model.model_name_or_path.split("/"))).pop(),
                    str(self.args.data.max_seq_length),
                    str(self.args.task_name),
                    str(self.args.data.test_id),
                ),
            )

        elif evaluate and not validate:
            cached_features_file = os.path.join(
                DATA_PATH,
                "cached_{}_{}_{}_{}_{}".format(
                    "test",
                    list(filter(None, self.args.model.model_name_or_path.split("/"))).pop(),
                    str(self.args.data.max_seq_length),
                    str(self.args.task_name),
                    str(self.args.data.test_id),
                ),
            )

        elif not evaluate and not validate:
            # if active learning, the train data will be saved inside each learning iteration directory
            cached_features_file = os.path.join(
                DATA_PATH,
                "cached_{}_{}_{}_{}_{}".format(
                    "train",
                    list(filter(None, self.args.model.model_name_or_path.split("/"))).pop(),
                    str(self.args.data.max_seq_length),
                    str(self.args.task_name),
                    str(self.args.data.test_id),
                ),
            )

        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", DATA_PATH)
            label_list = self.processor.get_labels()
            if validate and not evaluate:
                examples = self.processor.get_valid_examples(self.args)
            elif evaluate and not validate:
                examples = self.processor.get_test_examples(self.args)
            elif not evaluate and not validate:
                examples = self.processor.get_train_examples(self.args)

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

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if self.output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif self.output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
