# -*- coding: utf-8 -*-

from ..loggers import STDLogger, FolderLogger
from ..utils import seed_everything


class BaseExperiment:

    def __init__(
        self,
        rank=0,
        seed=42,
        loggers=None,
        h_params=None,
        experiment_name=None,
        **kwargs
    ):
        # #
        self.rank = rank
        self.seed = seed
        seed_everything(self.seed)
        self.experiment_name = experiment_name or 'debug'

        self.h_params = h_params or {}
        self.h_params['seed'] = seed
        self.h_params['experiment_name'] = self.experiment_name

        self.loggers = loggers if loggers is not None else [STDLogger(), FolderLogger()]
        self._create_experiment()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _create_experiment(self):
        if self.rank == 0:
            for logger in self.loggers:
                logger.create_experiment(self.experiment_name, self.h_params)

    def _log_on_start_training(self, n_epochs, steps_per_epoch):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_on_start_training(n_epochs, steps_per_epoch)

    def _log_on_end_training(self):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_on_end_training()

    def _log_on_start_epoch(self, stage, lr):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_on_start_epoch(stage, lr)

    def _log_on_end_epoch(self, stage, *args, **kwargs):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_on_end_epoch(stage, *args, **kwargs)

    def _log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_on_step(stage, step, epoch, global_step, *args, **kwargs)

    def destroy(self):
        if self.rank == 0:
            for logger in self.loggers:
                logger.destroy()
