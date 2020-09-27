from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.transformers import PretrainedTransformer

from copy import deepcopy


class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, model: PretrainedTransformer, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # modify saving
    def _save_model(self, filepath):
        self.model.best_model = deepcopy(self.model.model)
        if self.model.hparams.exp.logging:
            self.model.trainer.logger.log_metrics({'best_epoch': self.model.trainer.current_epoch + 1},
                                                  step=self.model.trainer.global_step)
