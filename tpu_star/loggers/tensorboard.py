# -*- coding: utf-8 -*-
import os
from glob import glob

from torch.utils.tensorboard import SummaryWriter

from .base import BaseLogger


class TensorBoardLogger(BaseLogger):

    def __init__(self, log_dir='./tb_logs', main_script_abs_path=None, verbose_step=10, max_queue=50):
        super().__init__()
        self.log_dir = log_dir
        self.verbose_step = verbose_step
        self.max_queue = max_queue
        self.main_script_abs_path = main_script_abs_path

    def get_dict_scalars(self, scalar_subkey='', **kwargs):
        if scalar_subkey:
            scalar_subkey += '/'
        hparams = {}
        for key, value in kwargs.items():
            key = f'{scalar_subkey}{key}'
            if isinstance(value, dict):
                hparams.update(self.get_dict_scalars(key, **value))
            elif isinstance(value, (float, int, str)):
                hparams[key] = value
            else:
                print(f'Warning! Unknown format for h_params key={key}, value={value}')
                continue
        return hparams

    def create_experiment(self, experiment_name, h_params):
        self.save_dir = os.path.join(self.log_dir, experiment_name)
        version_number = len(glob(os.path.join(self.save_dir, '*')))
        self.save_dir = os.path.join(self.save_dir, f'version_{version_number}')
        self.writer = SummaryWriter(log_dir=self.save_dir, comment='loh-loh', max_queue=self.max_queue)
        hparams = self.get_dict_scalars('h_params', **h_params)
        for key, value in hparams.items():
            self.writer.add_text(key, str(value))
        if self.main_script_abs_path is not None:
            self.log_artifact(self.main_script_abs_path)

    def destroy(self):
        self.writer.flush()
        self.writer.close()

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        if global_step and global_step % self.verbose_step == 0:
            for key, value in kwargs.items():
                self.writer.add_scalar(f'{stage}/global_step/{key}', value, global_step)

    def log_on_start_training(self, n_epochs, steps_per_epoch):
        pass

    def log_on_end_training(self):
        pass

    def log_on_start_epoch(self, stage, lr, epoch, global_step):
        if stage == 'train':
            self.writer.add_scalar('train/epoch/lr', lr, epoch)

    def log_on_end_epoch(self, stage, epoch, global_step, *args, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(f'{stage}/epoch/{key}', value, epoch)

    def log_artifact(self, abs_path, name=None):
        name = os.path.basename(abs_path)
        os.system(f'cp "{abs_path}" "{self.save_dir}/{name}"')
