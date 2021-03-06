import logging
import hydra
from hydra.utils import instantiate
import torch
from omegaconf import DictConfig
import mlflow

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

from src import MLFLOW_URI, CONFIG_PATH, ROOT_PATH
from src.models.checkpoint import CustomModelCheckpoint
from src.utils import set_seed, calculate_hash


logger = logging.getLogger(__name__)


@hydra.main(config_name=f'{CONFIG_PATH}/config.yaml', config_path=CONFIG_PATH, strict=False)
def main(args: DictConfig):
    # Distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    if str(args.exp.gpus) == '-1':
        args.exp.gpus = torch.cuda.device_count()

    # Secondary data args
    args.data.setting = 'in-topic' if args.data.test_id is None else 'cross-topic'
    dataset_name = args.data.path.split('/')[1]
    args.data.path = f'{ROOT_PATH}/{args.data.path}'

    # MlFlow Logging
    if args.exp.logging:
        experiment_name = f'{dataset_name}/{args.setting}-{args.data.setting}/{args.exp.task_name}'
        mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=MLFLOW_URI)
        experiment = mlf_logger._mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id

            if args.exp.check_exisisting_hash:
                args.hash = calculate_hash(args)
                existing_runs = mlf_logger._mlflow_client.search_runs(filter_string=f"params.hash = '{args.hash}'",
                                                                      run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
                                                                      experiment_ids=[experiment_id])
                if len(existing_runs) > 0:
                    logger.info('Skipping existing run.')
                    return
                else:
                    logger.info('No runs found - perfoming one.')

    #     cpnt_path = f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts'
    # else:
    #     cpnt_path = None

    # Load pretrained model and tokenizer
    set_seed(args)
    model = instantiate(args.lightning_module, args=args)
    logger.info(f'Run arguments: \n{args.pretty()}')

    # Early stopping & Checkpointing
    early_stop_callback = EarlyStopping(min_delta=0.00, patience=args.exp.early_stopping_patience, verbose=False, mode='min')
    checkpoint_callback = CustomModelCheckpoint(model=model, verbose=True, mode='min', save_top_k=1,
                                                period=0 if args.exp.val_check_interval < 1.0 else 1)
    lr_logging_callback = LearningRateLogger(logging_interval='epoch')

    # Training
    trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                      logger=mlf_logger if args.exp.logging else None,
                      max_epochs=args.exp.max_epochs,
                      gradient_clip_val=args.optimizer.max_grad_norm,
                      early_stop_callback=early_stop_callback,
                      val_check_interval=args.exp.val_check_interval,
                      checkpoint_callback=checkpoint_callback if args.exp.checkpoint else None,
                      accumulate_grad_batches=args.exp.gradient_accumulation_steps,
                      auto_lr_find=args.optimizer.auto_lr_find,
                      precision=args.exp.precision,
                      distributed_backend='dp',
                      callbacks=[lr_logging_callback])
    trainer.fit(model)
    trainer.test(model)

    # Cleaning cache
    torch.cuda.empty_cache()

    # Ending the run
    if args.exp.logging:
        mlf_logger.finalize()


if __name__ == "__main__":
    main()
