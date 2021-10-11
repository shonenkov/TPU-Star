# -*- coding: utf-8 -*-
from tqdm import tqdm

from .base import BaseLogger


class ProgressBarLogger(BaseLogger):

    def __init__(self, verbose_ndigits=5, verbose_step=10**5):
        super().__init__()
        self.verbose_ndigits = verbose_ndigits
        self.verbose_step = verbose_step
        self.current_step = 0
        self.current_epoch = 0

    def create_experiment(self, experiment_name, h_params):
        pass

    def destroy(self, *args, **kwargs):
        pass

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if stage == 'train':
            self.one_epoch_progress_bar.update(1)
            self.current_step += 1

    def log_on_start_training(self, n_epochs, steps_per_epoch, *args, **kwargs):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_progress_bar = tqdm(total=n_epochs)
        self.one_epoch_progress_bar = tqdm(total=self.steps_per_epoch, leave=False)

    def log_on_end_training(self, *args, **kwargs):
        pass

    def log_on_start_epoch(self, stage, lr, epoch, global_step, *args, **kwargs):
        pass

    def log_on_end_epoch(self, stage, epoch, global_step, *args, **kwargs):
        if stage == 'train':
            self.train_progress_bar.update(1)
            self.current_epoch += 1
        if stage == 'valid':
            self.one_epoch_progress_bar.update(-self.steps_per_epoch)
            self.current_step -= self.steps_per_epoch

    def log_artifact(self, abs_path, name):
        pass

    def state_dict(self):
        return {
            'logger_name': self.name,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch
        }

    def resume(self, logger_state_dict):
        self.current_step = logger_state_dict['current_step']
        self.current_epoch = logger_state_dict['current_epoch']
        self.one_epoch_progress_bar.update(self.current_step)
        self.train_progress_bar.update(self.current_epoch)
