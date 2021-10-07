# -*- coding: utf-8 -*-
from .base import BaseLogger


class WandBLogger(BaseLogger):
    """ WANDB_API_KEY """

    def __init__(self, run, main_script_abs_path=None):
        self.run = run
        self.main_script_abs_path = main_script_abs_path

    def create_experiment(self, experiment_name, h_params):
        self.run.name = experiment_name
        self.run.config.update(h_params)
        if 'tags' in h_params:
            for tag in h_params['tags']:
                self.run.tags += (tag,)

        if self.main_script_abs_path is not None:
            self.log_artifact(self.main_script_abs_path, 'main_script')

    def destroy(self):
        self.run.finish()

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        pass

    def log_on_start_training(self, n_epochs, steps_per_epoch):
        pass

    def log_on_end_training(self):
        pass

    def log_on_start_epoch(self, stage, lr, epoch, global_step):
        if stage == 'train':
            self.run.log({'lr': lr})

    def log_on_end_epoch(self, stage, epoch, global_step, *args, **kwargs):
        log_metrics = {}
        for key, value in kwargs.items():
            log_metrics[f'{stage}_{key}'] = value
        self.run.log(log_metrics)

    def log_artifact(self, abs_path, name):
        self.run.save(abs_path)
