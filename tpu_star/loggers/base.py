class BaseLogger:

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

    def log_on_start_epoch(self, stage, lr):
        raise NotImplementedError

    def log_on_end_epoch(self, stage, *args, **kwargs):
        raise NotImplementedError

    def log_artifact(self, abs_path):
        raise NotImplementedError
