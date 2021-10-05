# -*- coding: utf-8 -*-
from tqdm import tqdm

from .base import BaseLogger


class ProgressBarLogger(BaseLogger):

    def __init__(self, verbose_ndigits=5, verbose_step=10**5):
        self.verbose_ndigits = verbose_ndigits
        self.verbose_step = verbose_step

    def create_experiment(self, experiment_name, h_params):
        pass

    def destroy(self, *args, **kwargs):
        pass

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if stage == 'train':
            self.one_epoch_progress_bar.update(1)

    def log_on_start_training(self, n_epochs, steps_per_epoch, *args, **kwargs):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_progress_bar = tqdm(total=n_epochs)
        self.one_epoch_progress_bar = tqdm(total=self.steps_per_epoch, leave=False)

    def log_on_end_training(self, *args, **kwargs):
        pass

    def log_on_start_epoch(self, stage, lr, *args, **kwargs):
        pass

    def log_on_end_epoch(self, stage, *args, **kwargs):
        if stage == 'train':
            self.train_progress_bar.update(1)
        if stage == 'valid':
            self.one_epoch_progress_bar.update(-self.steps_per_epoch)

    def log_artifact(self, abs_path, *args, **kwargs):
        pass
