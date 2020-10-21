# -*- coding: utf-8 -*-
import os
import gc
import time
import random
from datetime import datetime

import torch
import numpy as np

from .metrics import MetricsGrabber


class TorchTPUExperiment:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        xm,
        pl,
        xser,
        rank,
        best_saving=True,
        last_saving=True,
        verbose=True,
        verbose_step=100,
        seed=42,
        base_dir='./saved_models',
        experiment_name=None,
    ):
        # #
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.seed = seed
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir) and rank == 0:
            os.makedirs(self.base_dir)
        self.experiment_name = experiment_name or f'debug-{round(datetime.utcnow().timestamp())}'
        self.experiment_dir = f'{self.base_dir}/{self.experiment_name}'
        if not os.path.exists(self.experiment_dir) and rank == 0:
            os.makedirs(self.experiment_dir)
        self.log_path = f'{self.experiment_dir}/log.txt'
        # #
        # #
        self.xm = xm
        self.pl = pl
        self.xser = xser,
        self.verbose_step = verbose_step
        self.rank = rank
        # #
        # #
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        # #
        # #
        self.epoch = -1
        self.is_train = None
        self.metrics = MetricsGrabber()
        self.train_metrics = {}
        self.valid_metrics = {}
        self.seed_everything(self.seed)
        # #
        # #
        self.last_saving = last_saving
        if best_saving is False or best_saving is None:
            self.best_saving = {}
        elif best_saving is True:
            self.best_saving = {'loss': 'min'}
        else:
            self.best_saving = best_saving
        # #

    def handle_one_batch(self, batch, *args, **kwargs):
        """
        you can use this structure, for example:

        [model forward using self.model]

        [call criterion using self.criterion]

        [calculate metrics]

        self.metrics.update(bs=bs, <metric_name_1>=<value_1>, ..., <metric_name_n>=<value_n>)

        if self.is_train:
            loss.backward()
            self.xm.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()
            self.scheduler.step()
        """
        raise NotImplementedError

    def custom_action_before_train_one_epoch(self):
        pass

    def custom_action_after_train_one_epoch(self):
        pass

    def custom_action_after_valid_one_epoch(self):
        pass

    def train_one_epoch(self, train_loader):
        t = time.time()
        self.__train()
        for step, batch in enumerate(train_loader):
            self.handle_one_batch(batch)
            if step and step % self.verbose_step == 0:
                self._print(
                    f'Train step {step}/{len(train_loader)}, time: {(time.time() - t):.1f}s',
                    **self.metrics.train_metrics[self.epoch].avg
                )

    def validation(self, valid_loader):
        t = time.time()
        self.__eval()
        for step, batch in enumerate(valid_loader):
            with torch.no_grad():
                self.handle_one_batch(batch)
            if step and step % self.verbose_step == 0:
                self._print(
                    f'Valid step {step}/{len(valid_loader)}, time: {(time.time() - t):.1f}s',
                    **self.metrics.valid_metrics[self.epoch].avg
                )

    def fit(self, train_loader, valid_loader, n_epochs):
        for e in range(n_epochs):
            self.__update_epoch()
            # #
            # #
            self.custom_action_before_train_one_epoch()
            # #
            # #
            lr = self.optimizer.param_groups[0]['lr']
            self._log(f'\n{datetime.utcnow().isoformat()}\nlr: {lr}')
            # #
            # #
            t = time.time()
            para_loader = self.pl.ParallelLoader(train_loader, [self.device])
            self.train_one_epoch(para_loader.per_device_loader(self.device))
            del para_loader
            gc.collect()
            # #
            stage = 'train'
            self.train_metrics[self.epoch] = self.__mesh_reduce_metrics(
                stage,
                **self.metrics.train_metrics[self.epoch].avg
            )
            self._log(f'Train epoch {self.epoch}, time: {(time.time() - t):.1f}s', **self.train_metrics[self.epoch])
            # #
            # #
            self.custom_action_after_train_one_epoch()
            # #
            # #
            t = time.time()
            para_loader = self.pl.ParallelLoader(valid_loader, [self.device])
            self.validation(para_loader.per_device_loader(self.device))
            del para_loader
            gc.collect()
            # #
            stage = 'valid'
            self.valid_metrics[self.epoch] = self.__mesh_reduce_metrics(
                stage,
                **self.metrics.valid_metrics[self.epoch].avg
            )
            self._log(f'Valid epoch {self.epoch}, time: {(time.time() - t):.1f}s', **self.valid_metrics[self.epoch])
            # #
            # #
            self.custom_action_after_valid_one_epoch()
            # #
            # #
            if self.last_saving:
                self.save(f'{self.experiment_dir}/last.pt')
            last_saved_path = None
            for key, mode in self.best_saving.items():
                if e == self.metrics.get_best_epoch(key, mode)['epoch']:
                    if self.last_saving:
                        os.system(f'cp "{self.experiment_dir}/last.pt" "{self.experiment_dir}/best_{key}.pt"')
                    elif last_saved_path:
                        os.system(f'cp "{last_saved_path}" "{self.experiment_dir}/best_{key}.pt"')
                    else:
                        last_saved_path = f'{self.experiment_dir}/best_{key}.pt'
                        self.save(last_saved_path)
            # #

    def save(self, path):
        self.model.eval()
        self.xm.save(self.model.state_dict(), path)
        if self.rank == 0:
            metrics_path = os.path.join(os.path.dirname(path), 'metrics.pt')
            torch.save({
                'train_metrics': self.train_metrics,
                'valid_metrics': self.valid_metrics,
                'master_metrics_grabber': self.metrics.state_dict(),
            }, metrics_path)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.xm.set_rng_state(seed)

    def _print(self, msg, *args, **kwargs):
        if self.verbose:
            msg = self.__prepare_msg(msg, *args, **kwargs)
            self.xm.master_print(msg)

    def _log(self, msg, *args, **kwargs):
        msg = self.__prepare_msg(msg, *args, **kwargs)
        if self.verbose:
            self._print(msg)
        if self.rank == 0:
            with open(self.log_path, 'a+') as logger:
                self.xm.master_print(f'{msg}', fd=logger)

    @staticmethod
    def __prepare_msg(msg, *args, **kwargs):
        msg = str(msg)
        for i, arg in enumerate(args):
            msg = f'{msg}, arg_{i}: {arg:.5f}'
        for key, arg in kwargs.items():
            msg = f'{msg}, {key}={arg:.5f}'
        return msg

    def __update_epoch(self):
        self.epoch += 1
        self.metrics.update_epoch()

    def __train(self):
        self.model.train()
        self.is_train = True
        self.metrics.is_train = True

    def __eval(self):
        self.model.eval()
        self.is_train = False
        self.metrics.is_train = False

    @staticmethod
    def __reduce_fn(vals):
        return sum(vals) / len(vals)

    def __mesh_reduce_metrics(self, stage=None, **kwargs):
        mesh_metrics = {}
        prefix = f'_{stage}' if stage else ''
        for metric, value in kwargs.items():
            mesh_value = self.xm.mesh_reduce(f'{metric}{prefix}_reduce', value, self.__reduce_fn)
            mesh_metrics[metric] = mesh_value
        return mesh_metrics
