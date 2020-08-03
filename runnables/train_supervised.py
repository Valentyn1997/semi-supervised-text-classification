import logging
import hydra
import torch
from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import MLFLOW_URI, CONFIG_PATH, ROOT_PATH
from src.models.transformers import PretrainedTransformer
from src.models.checkpoint import CustomModelCheckpoint
from src.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path=f'{CONFIG_PATH}/config.yaml', strict=False)
def main(args: DictConfig):

    # Secondary data args
    args.data.setting = 'in-topic' if args.data.test_id is None else 'cross-topic'
    dataset_name = args.data.path.split('/')[1]
    args.data.path = f'{ROOT_PATH}/{args.data.path}'

    # MlFlow Logging
    if args.exp.logging:
        experiment_name = f'{dataset_name}/supervised-{args.data.setting}/{args.exp.task_name}'
        mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=MLFLOW_URI)
        run_id = mlf_logger.run_id
        experiment_id = mlf_logger.experiment.get_experiment_by_name(experiment_name).experiment_id
        cpnt_path = f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts'
    else:
        cpnt_path = None

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
    checkpoint_callback = CustomModelCheckpoint(model=model, filepath=cpnt_path, verbose=True, monitor='val_loss', mode='min',
                                                save_top_k=1, period=0)

    logger.info(f'Run arguments: \n{args.pretty()}')

    # Training
    trainer = Trainer(gpus=list(args.exp.gpus),
                      logger=mlf_logger if args.exp.logging else None,
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

    # Cleaning cache
    torch.cuda.empty_cache()

    # Ending the run
    if args.exp.logging:
        mlf_logger.finalize()


if __name__ == "__main__":
    main()
