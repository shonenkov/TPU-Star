# -*- coding: utf-8 -*-
import os
import gc
import time
from datetime import datetime

import torch

from .base import BaseExperiment
from .metrics import MetricsGrabber


class TorchGPUExperiment(BaseExperiment):

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        rank=0,
        seed=42,
        verbose=True,
        verbose_end='\n',
        verbose_ndigits=5,
        verbose_step=1,
        base_dir='./saved_models',
        jupyters_path=None,
        notebook_name=None,
        experiment_name=None,
        neptune=None,
        neptune_params=None,
        best_saving=True,
        last_saving=True,
    ):
        self.verbose_step = verbose_step
        super().__init__(
            rank=rank,
            seed=seed,
            verbose=verbose,
            verbose_end=verbose_end,
            verbose_ndigits=verbose_ndigits,
            base_dir=base_dir,
            jupyters_path=jupyters_path,
            notebook_name=notebook_name,
            experiment_name=experiment_name,
            neptune=neptune,
            neptune_params=neptune_params,
        )
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
            self.optimizer_step()
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
        self._train()
        for step, batch in enumerate(train_loader):
            self.handle_one_batch(batch)
            if step and step % self.verbose_step == 0:
                self._print(
                    f'Train step {step}/{len(train_loader)}, time: {(time.time() - t):.1f}s',
                    **self.metrics.train_metrics[self.epoch].avg
                )

    def validation(self, valid_loader):
        t = time.time()
        self._eval()
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
            self._update_epoch()
            # #
            # #
            self._custom_action_before_train_one_epoch()
            # #
            # #
            lr = self.optimizer.param_groups[0]['lr']
            self._log(f'\n{datetime.utcnow().isoformat()}\nlr: {lr}')
            self._log_neptune(lr=lr)
            self._log_neptune(epoch=self.epoch)
            # #
            # #
            t = time.time()
            self.train_one_epoch(self._rebuild_loader(train_loader))
            # #
            stage = 'train'
            metrics = self._get_current_metrics(stage)
            self._log(f'Train epoch {self.epoch}, time: {(time.time() - t):.1f}s', **metrics)
            self._log_neptune(stage, **metrics)
            # #
            # #
            self._custom_action_after_train_one_epoch()
            # #
            # #
            t = time.time()
            self.validation(self._rebuild_loader(valid_loader))
            # #
            stage = 'valid'
            metrics = self._get_current_metrics(stage)
            self._log(f'Valid epoch {self.epoch}, time: {(time.time() - t):.1f}s', **metrics)
            self._log_neptune(stage, **metrics)
            # #
            # #
            self._custom_action_after_valid_one_epoch()
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
        torch.save({
            'epoch': self.epoch,
            'best_saving': self.best_saving,
            'last_saving': self.last_saving,
            'is_train': self.is_train,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_state_dict': self.metrics.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.best_saving = checkpoint['best_saving']
        self.last_saving = checkpoint['last_saving']
        self.is_train = checkpoint['is_train']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics.load_state_dict(checkpoint['metrics_state_dict'])

    def destroy(self):
        def _optimizer_to(optimizer, device):
            for param in optimizer.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)
        gc.collect()
        torch.cuda.empty_cache()
        cpu_device = torch.device('cpu')
        _optimizer_to(self.optimizer, cpu_device)
        self.model.to(cpu_device)
        try:
            self.criterion.to(cpu_device)
        except Exception:  # noqa
            pass
        torch.cuda.empty_cache()
        del self.optimizer
        del self.model
        del self.criterion
        del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()

        if self.rank == 0 and self.neptune:
            self.neptune.stop()

    def optimizer_step(self):
        self.optimizer.step()

    def _rebuild_loader(self, loader):
        return loader

    def _wipe_memory(self):
        gc.collect()

    def _get_current_metrics(self, stage):
        if stage == 'train':
            return self.metrics.train_metrics[self.epoch].avg
        elif stage == 'valid':
            return self.metrics.valid_metrics[self.epoch].avg
        else:
            raise ValueError(f'Incorrect stage: "{stage}".')

    def _update_epoch(self):
        self.epoch += 1
        self.metrics.update_epoch()

    def _train(self):
        self.model.train()
        self.is_train = True
        self.metrics.is_train = True

    def _eval(self):
        self.model.eval()
        self.is_train = False
        self.metrics.is_train = False

    def _custom_action_before_train_one_epoch(self):
        self._wipe_memory()
        self.custom_action_before_train_one_epoch()

    def _custom_action_after_train_one_epoch(self):
        self._wipe_memory()
        self.custom_action_after_train_one_epoch()

    def _custom_action_after_valid_one_epoch(self):
        self._wipe_memory()
        self.custom_action_after_valid_one_epoch()
