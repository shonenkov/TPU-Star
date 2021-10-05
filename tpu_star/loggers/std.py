# -*- coding: utf-8 -*-
from datetime import datetime

from .base import BaseLogger
from .utils import prepare_text_msg


class STDLogger(BaseLogger):

    def __init__(self, verbose_ndigits=5, verbose_step=10**5):
        self.verbose_ndigits = verbose_ndigits
        self.verbose_step = verbose_step

    def create_experiment(self, experiment_name, h_params):
        pass

    def destroy(self, *args, **kwargs):
        pass

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if step and step % self.verbose_step == 0:
            if stage == 'train':
                msg = f'Train step {step}/{self.steps_per_epoch}'
            elif stage == 'valid':
                msg = f'Valid step {step}/{self.steps_per_epoch}'
            else:
                msg = f'{step}/{self.steps_per_epoch}'
            msg = prepare_text_msg(msg, self.verbose_ndigits,  *args, **kwargs)
            print(msg)

    def log_on_start_training(self, n_epochs, steps_per_epoch, *args, **kwargs):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch

    def log_on_end_training(self, *args, **kwargs):
        pass

    def log_on_start_epoch(self, stage, lr, *args, **kwargs):
        if stage == 'train':
            print(f'\n{datetime.utcnow().isoformat()}\nlr: {lr:{self.verbose_ndigits}}')

    def log_on_end_epoch(self, stage, *args, **kwargs):
        if stage == 'train':
            msg = 'Train'
        elif stage == 'valid':
            msg = 'Valid'
        else:
            msg = ''
        msg = prepare_text_msg(msg, self.verbose_ndigits,  *args, **kwargs)
        print(msg)

    def log_artifact(self, abs_path, name):
        pass
