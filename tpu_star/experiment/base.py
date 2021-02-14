# -*- coding: utf-8 -*-
import os
import random
from glob import glob

import numpy as np
import torch


class BaseExperiment:

    def __init__(
        self,
        rank=0,
        seed=42,
        verbose=True,
        verbose_end='\n',
        verbose_ndigits=5,
        base_dir='./saved_models',
        jupyters_path=None,
        notebook_name=None,
        experiment_name=None,
        neptune=None,
        neptune_params=None,
        optuna=None,
        optuna_trial=None,
        optuna_report_metric=None,
        **kwargs
    ):
        # #
        self.rank = rank
        self.seed = seed
        self.verbose = verbose
        self.verbose_end = verbose_end
        self.verbose_ndigits = verbose_ndigits
        # #
        # #
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir) and self.rank == 0:
            os.makedirs(self.base_dir)
        # #
        # #
        self.experiment_name = experiment_name or 'debug'
        experiment_count = len(glob(f'{self.base_dir}/{self.experiment_name}*'))
        if experiment_count != 0:
            self.experiment_name = self.experiment_name + '-' + str(experiment_count)
        self.experiment_dir = f'{self.base_dir}/{self.experiment_name}'
        if not os.path.exists(self.experiment_dir) and self.rank == 0:
            os.makedirs(self.experiment_dir)
        # #
        # #
        self.jupyters_path = jupyters_path or '.'
        self.notebook_name = notebook_name
        if self.notebook_name and self.rank == 0:
            os.system(f'cp "{self.jupyters_path}/{self.notebook_name}" "{self.experiment_dir}/{self.notebook_name}"')
        # #
        # #
        self.log_path = f'{self.experiment_dir}/log.txt'
        self._seed_everything(self.seed)
        # #
        # #
        self.neptune_params = neptune_params or {}
        self._init_neptune(neptune)
        # #
        # #
        self.optuna = optuna
        self.optuna_trial = optuna_trial
        self.optuna_report_metric = optuna_report_metric

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def _init_neptune(self, neptune):
        if neptune and self.rank == 0:
            self.neptune = neptune.create_experiment(
                name=self.experiment_name,
                **self.neptune_params
            )
            if self.notebook_name:
                self.neptune.log_artifact(f'{self.jupyters_path}/{self.notebook_name}')
        else:
            self.neptune = None

    def _log_neptune(self, stage=None, **kwargs):
        if self.neptune and self.rank == 0:
            prefix = f'_{stage}' if stage else ''
            for key, arg in kwargs.items():
                self.neptune.log_metric(f'{key}{prefix}', arg)

    def _log(self, msg, *args, **kwargs):
        msg = self._prepare_msg(msg, *args, **kwargs)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{msg}\n')
        if self.verbose:
            print(msg)

    def _print(self, msg, *args, **kwargs):
        if self.verbose:
            msg = self._prepare_msg(msg, *args, **kwargs)
            print(msg, end=self.verbose_end)

    def _prepare_msg(self, msg, *args, **kwargs):
        msg = str(msg)
        for i, arg in enumerate(args):
            msg = f'{msg}, arg_{i}: {arg:.{self.verbose_ndigits}f}'
        for key, arg in kwargs.items():
            msg = f'{msg}, {key}={arg:.{self.verbose_ndigits}f}'
        return msg

    def _report_optuna(self, metrics, epoch):
        if self.optuna_trial:
            self.optuna_trial.report(metrics, epoch)
            if self.optuna_trial.should_prune():
                self.destroy()
                raise self.optuna.exceptions.TrialPruned()

    def destroy(self):
        pass
