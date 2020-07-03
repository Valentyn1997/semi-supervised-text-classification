import logging
import hydra
from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import MLFLOW_URI, CONFIG_PATH, ROOT_PATH
from src.models.transformers import PretrainedTransformer
from src.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path=f'{CONFIG_PATH}/config.yaml', strict=False)
def main(args: DictConfig):

    # Secondary data args
    args.data.setting = 'in-topic' if args.data.test_id is None else 'cross-topic'
    args.data.path = f'{ROOT_PATH}/{args.data.path}'

    # MlFlow Logging
    experiment_name = f'supervised-{args.data.setting}'
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=MLFLOW_URI)
    run_id = mlf_logger.run_id
    experiment_id = mlf_logger.experiment.get_experiment_by_name(experiment_name).experiment_id

    # Load pretrained model and tokenizer
    set_seed(args)
    model = PretrainedTransformer(args)
    model.prepare_data()
    train_dataloader = model.train_dataloader()

    # Max number of epochs/steps - secondary parameter
    if args.exp.max_steps > 0:
        args.exp.max_epochs = (args.exp.max_steps // (len(train_dataloader) // args.exp.gradient_accumulation_steps) + 1)
    else:
        args.exp.max_steps = (len(train_dataloader) // args.exp.gradient_accumulation_steps * args.exp.max_epochs)

    # Early stopping & Checkpointing
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.exp.early_stopping_patience,
                                        verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(filepath=f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts',
                                          verbose=True, monitor='val_loss', mode='min', save_top_k=1, period=0)
    # Setting seeds & printing paranms
    logger.info(f'Run arguments: \n{args.pretty()}')

    # Training
    trainer = Trainer(gpus=list(args.exp.gpus),
                      logger=mlf_logger,
                      max_epochs=args.exp.max_epochs,
                      gradient_clip_val=args.optimizer.max_grad_norm,
                      early_stop_callback=early_stop_callback,
                      val_check_interval=0.5,
                      checkpoint_callback=checkpoint_callback if args.exp.checkpoint else None,
                      accumulate_grad_batches=args.exp.gradient_accumulation_steps,
                      auto_lr_find=args.optimizer.auto_lr_find)
    trainer.fit(model, train_dataloader=train_dataloader)

    # Testing
    model.model = model.best_model
    trainer.run_evaluation(test_mode=True)
    checkpoint_callback._del_model(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
