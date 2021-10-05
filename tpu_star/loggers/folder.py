import os
from glob import glob
from datetime import datetime

from .base import BaseLogger
from .utils import prepare_text_msg


class FolderLogger(BaseLogger):

    def __init__(self, base_dir='./saved_models', main_script_abs_path=None, verbose_ndigits=5, verbose_step=10**5):
        self.verbose_ndigits = verbose_ndigits
        self.verbose_step = verbose_step
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.main_script_abs_path = main_script_abs_path

    def create_experiment(self, experiment_name, h_params):
        self.experiment_name = experiment_name
        experiment_count = len(glob(f'{self.base_dir}/{self.experiment_name}*'))
        if experiment_count != 0:
            self.experiment_name = self.experiment_name + '-' + str(experiment_count)
        self.experiment_dir = f'{self.base_dir}/{self.experiment_name}'
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if self.main_script_abs_path is not None:
            self.log_artifact(self.main_script_abs_path)
        self.log_path = f'{self.experiment_dir}/log.txt'

    def destroy(self, *args, **kwargs):
        pass

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if stage == 'train':
            msg = f'Train step {step}/{self.steps_per_epoch}'
        elif stage == 'valid':
            msg = f'Valid step {step}/{self.steps_per_epoch}'
        else:
            msg = f'{step}/{self.steps_per_epoch}'
        msg = prepare_text_msg(msg, self.verbose_ndigits,  *args, **kwargs)
        if step and step % self.verbose_step == 0:
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{msg}\n')

    def log_on_start_training(self, n_epochs, steps_per_epoch, *args, **kwargs):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch

    def log_on_end_training(self, *args, **kwargs):
        pass

    def log_on_start_epoch(self, stage, lr, *args, **kwargs):
        if stage == 'train':
            msg = f'\n{datetime.utcnow().isoformat()}\nlr: {lr:{self.verbose_ndigits}}'
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{msg}\n')

    def log_on_end_epoch(self, stage, *args, **kwargs):
        if stage == 'train':
            msg = 'Train'
        elif stage == 'valid':
            msg = 'Valid'
        else:
            msg = ''
        msg = prepare_text_msg(msg, self.verbose_ndigits,  *args, **kwargs)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{msg}\n')

    def log_artifact(self, abs_path, *args, **kwargs):
        name = os.path.basename(abs_path)
        os.system(f'cp "{abs_path}" "{self.experiment_dir}/{name}"')
