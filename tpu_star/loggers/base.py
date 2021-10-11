# -*- coding: utf-8 -*-
class BaseLogger:

    def __init__(self):
        self.history = []

    def create_experiment(self, experiment_name, h_params):
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        raise NotImplementedError

    def log_on_start_training(self, n_epochs, steps_per_epoch):
        raise NotImplementedError

    def log_on_end_training(self):
        raise NotImplementedError

    def log_on_start_epoch(self, stage, epoch, global_step, lr):
        raise NotImplementedError

    def log_on_end_epoch(self, stage, epoch, global_step, *args, **kwargs):
        raise NotImplementedError

    def log_artifact(self, abs_path, name):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def state_dict(self):
        return {
            'logger_name': self.name,
            'history': self.history
        }

    def _save_history(self, method, *args, **kwargs):
        self.history.append([method, args, kwargs])

    def resume(self, logger_state_dict):
        for method, args, kwargs in logger_state_dict['history']:
            getattr(self, method)(*args, **kwargs)
