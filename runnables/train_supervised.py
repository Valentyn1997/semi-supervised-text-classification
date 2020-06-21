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

    mlf_logger = MLFlowLogger(experiment_name='supervised-stance-detection', tracking_uri=MLFLOW_URI)
    run_id = mlf_logger.run_id
    experiment_id = mlf_logger.experiment.get_experiment_by_name('supervised-stance-detection').experiment_id

    # Load pretrained model and tokenizer
    model_wrapper = PretrainedTransformer(args=args)
    model_wrapper.prepare_data()
    train_dataloader = model_wrapper.train_dataloader()

    if args.max_steps > 0:
        args.t_total = args.max_steps  # total optimization tests
        args.num_train_epochs = (args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        args.t_total = (len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

    set_seed(args)

    # Early stopping
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.early_stopping_patience,
                                        verbose=False, mode='min')
    # Checkpoint
    checkpoint_callback = ModelCheckpoint(filepath=f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts',
                                          verbose=True, monitor='val_loss', mode='min', save_top_k=1, period=0)

    logger.info(f'Run arguments: \n{args.pretty()}')
    trainer = Trainer(gpus=1,
                      logger=mlf_logger,
                      max_epochs=args.num_train_epochs,
                      gradient_clip_val=args.optimizer.max_grad_norm,
                      early_stop_callback=early_stop_callback,
                      val_check_interval=0.5,
                      checkpoint_callback=checkpoint_callback if args.checkpoint else None)
    trainer.fit(model_wrapper, train_dataloader=train_dataloader)


if __name__ == "__main__":
    main()
