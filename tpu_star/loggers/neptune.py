from .base import BaseLogger


class NeptuneLogger(BaseLogger):

    def __init__(self, run):
        self.run = run

    def create_experiment(self, experiment_name, h_params):
        self.run['model/parameters'] = h_params

    def destroy(self, *args, **kwargs):
        raise NotImplementedError

    def log_on_step(self, stage, step, epoch, global_step, *args, **kwargs):
        raise NotImplementedError

    def log_on_start_training(self, n_epochs, steps_per_epoch, *args, **kwargs):
        raise NotImplementedError

    def log_on_end_training(self, *args, **kwargs):
        raise NotImplementedError

    def log_on_start_epoch(self, stage, lr, *args, **kwargs):
        raise NotImplementedError

    def log_on_end_epoch(self, stage, *args, **kwargs):
        raise NotImplementedError

    def log_artifact(self, abs_path, *args, **kwargs):
        raise NotImplementedError


