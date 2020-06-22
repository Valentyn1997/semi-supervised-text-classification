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
    model_wrapper = PretrainedTransformer(args=args)
    model_wrapper.prepare_data()
    train_dataloader = model_wrapper.train_dataloader()

    # Max number of epochs/steps
    if args.max_steps > 0:
        args.max_epochs = (args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        args.max_steps = (len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs)

    # Early stopping & Checkpointing
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.early_stopping_patience,
                                        verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(filepath=f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts',
                                          verbose=True, monitor='val_loss', mode='min', save_top_k=1, period=0)
    # Setting seeds & printing paranms
    set_seed(args)
    logger.info(f'Run arguments: \n{args.pretty()}')

    # Training
    trainer = Trainer(gpus=1,
                      logger=mlf_logger,
                      max_epochs=args.max_epochs,
                      gradient_clip_val=args.optimizer.max_grad_norm,
                      early_stop_callback=early_stop_callback,
                      val_check_interval=0.5,
                      checkpoint_callback=checkpoint_callback if args.checkpoint else None,
                      accumulate_grad_batches=args.gradient_accumulation_steps)
    trainer.fit(model_wrapper, train_dataloader=train_dataloader)
    trainer.test()


if __name__ == "__main__":
    main()
