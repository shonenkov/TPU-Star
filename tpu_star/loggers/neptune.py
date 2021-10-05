# -*- coding: utf-8 -*-
from .base import BaseLogger


class NeptuneLogger(BaseLogger):

    def __init__(self, run, main_script_abs_path=None):
        self.run = run
        self.main_script_abs_path = main_script_abs_path

    def create_experiment(self, experiment_name, h_params):
        self.run['model/experiment_name'] = experiment_name
        self.run['model/parameters'] = h_params
        if 'tags' in h_params:
            self.run['sys/tags'].add(h_params['tags'])
        if self.main_script_abs_path is not None:
            self.log_artifact(self.main_script_abs_path, 'main_script')

    def destroy(self):
        self.run.stop()

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        pass

    def log_on_start_training(self, n_epochs, steps_per_epoch):
        pass

    def log_on_end_training(self):
        pass

    def log_on_start_epoch(self, stage, lr):
        if stage == 'train':
            self.run['train/epoch/lr'].log(lr)

    def log_on_end_epoch(self, stage, *args, **kwargs):
        for key, value in kwargs.items():
            self.run[f'{stage}/epoch/{key}'].log(value)

    def log_artifact(self, abs_path, name):
        self.run[name].track_files(abs_path)
