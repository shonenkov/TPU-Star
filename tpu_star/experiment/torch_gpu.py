# -*- coding: utf-8 -*-
import os
import gc
import time
from datetime import datetime

import torch
from tqdm.auto import tqdm

from .base import BaseExperiment
from .metrics import MetricsGrabber, SystemMetricsGrabber


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
        use_progress_bar=False,
        base_dir='./saved_models',
        jupyters_path=None,
        notebook_name=None,
        experiment_name=None,
        neptune=None,
        neptune_params=None,
        best_saving=True,
        last_saving=True,
        low_memory=False,
        optuna=None,
        optuna_trial=None,
        optuna_report_metric=None,
        **kwargs,
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
            optuna=optuna,
            optuna_trial=optuna_trial,
            optuna_report_metric=optuna_report_metric,
            **kwargs,
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
        self.system_metrics = SystemMetricsGrabber()
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
        self.use_progress_bar = use_progress_bar
        self.train_progress_bar = None
        self.epoch_progress_bar = None
        # #
        self.low_memory = low_memory

    def handle_one_batch(self, batch, *args, **kwargs):
        """
        you can use this structure, for example:

        [model forward using self.model]

        [calculate criterion using self.criterion]

        [calculate metrics]

        self.metrics.update(<metric_name_1>=<value_1>, ..., <metric_name_n>=<value_n>)

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
            if self.rank == 0 and self.use_progress_bar:
                self.epoch_progress_bar.update(1)

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
        # #
        if self.rank == 0 and self.use_progress_bar:
            self.train_progress_bar = tqdm(total=n_epochs)
            self.epoch_progress_bar = tqdm(total=len(train_loader), leave=False)
        # #
        for e in range(n_epochs):
            self._update_epoch()
            # #
            # #
            self._custom_action_before_train_one_epoch()
            # #
            # #
            lr = self.optimizer.param_groups[0]['lr']
            self._log(f'\n{datetime.utcnow().isoformat()}\nlr: {lr}')
            self.system_metrics.update(lr=lr, epoch=self.epoch)
            # #
            # #
            epoch_time = time.time()
            self.train_one_epoch(self._rebuild_loader(train_loader))
            # #
            stage = 'train'
            metrics = self._get_current_metrics(stage)
            epoch_time = time.time() - epoch_time
            self.system_metrics.update(train_epoch_time=epoch_time)
            self._log(f'Train epoch {self.epoch}, time: {epoch_time:.1f}s', **metrics)
            self._log_neptune(stage, **metrics)
            # #
            # #
            self._custom_action_after_train_one_epoch()
            # #
            # #
            epoch_time = time.time()
            self.validation(self._rebuild_loader(valid_loader))
            # #
            stage = 'valid'
            metrics = self._get_current_metrics(stage)
            epoch_time = time.time() - epoch_time
            self.system_metrics.update(valid_epoch_time=epoch_time, valid_iterations=len(valid_loader))
            self._log(f'Valid epoch {self.epoch}, time: {epoch_time:.1f}s', **metrics)
            self._log_neptune(stage, **metrics)
            # #
            # #
            system_metrics = self.system_metrics.metrics[self.epoch].avg
            self._log_neptune(**system_metrics)
            # #
            # #
            self._custom_action_after_valid_one_epoch()
            # #
            self._low_memory()
            # #
            if self.last_saving:
                self.save(f'{self.experiment_dir}/last.pt')
            last_saved_path = None
            for key, mode in self.best_saving.items():
                if self.epoch == self.metrics.get_best_epoch(key, mode)['epoch']:
                    if self.last_saving:
                        os.system(f'cp "{self.experiment_dir}/last.pt" "{self.experiment_dir}/best_{key}.pt"')
                    elif last_saved_path:
                        os.system(f'cp "{last_saved_path}" "{self.experiment_dir}/best_{key}.pt"')
                    else:
                        last_saved_path = f'{self.experiment_dir}/best_{key}.pt'
                        self.save(last_saved_path)
            # #
            # #
            if self.rank == 0 and self.use_progress_bar:
                self.train_progress_bar.update(1)
                self.epoch_progress_bar.update(-len(train_loader))
            # #
            if self.optuna_report_metric:
                self._report_optuna(metrics[self.optuna_report_metric], self.epoch)

    @classmethod
    def resume(
        cls,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_path,
        train_loader,
        valid_loader,
        n_epochs,
        neptune=None,
        seed=None,
        **kwargs,
    ):
        checkpoint = torch.load(checkpoint_path)
        experiment_state_dict = checkpoint['experiment_state_dict']
        neptune_state_dict = checkpoint['neptune_state_dict']

        experiment_name = 'R+' + experiment_state_dict['experiment_name']

        experiment = cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            rank=experiment_state_dict['rank'],
            seed=experiment_state_dict['seed'] if seed is None else seed,
            verbose=experiment_state_dict['verbose'],
            verbose_end=experiment_state_dict['verbose_end'],
            verbose_ndigits=experiment_state_dict['verbose_ndigits'],
            verbose_step=experiment_state_dict['verbose_step'],
            use_progress_bar=experiment_state_dict['use_progress_bar'],
            base_dir=experiment_state_dict['base_dir'],
            jupyters_path=experiment_state_dict['jupyters_path'],
            notebook_name=experiment_state_dict['notebook_name'],
            experiment_name=experiment_name,
            neptune=None,
            neptune_params=neptune_state_dict['params'],
            best_saving=experiment_state_dict['best_saving'],
            last_saving=experiment_state_dict['last_saving'],
            low_memory=experiment_state_dict.get('low_memory', True),
            **kwargs
        )

        experiment.epoch = experiment_state_dict['epoch']
        experiment.model.load_state_dict(checkpoint['model_state_dict'])
        experiment.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        experiment.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        experiment.metrics.load_state_dict(checkpoint['metrics_state_dict'])
        experiment.system_metrics.load_state_dict(checkpoint['system_metrics_state_dict'])

        experiment._init_neptune(neptune)

        experiment.verbose = False
        for e in range(experiment.epoch + 1):
            system_metrics = experiment.system_metrics.metrics[e].avg
            experiment._log(f'\n{datetime.utcnow().isoformat()}\nlr: {system_metrics["lr"]}')
            # #
            metrics = experiment.metrics.train_metrics[e].avg
            experiment._log(f'Train epoch {e}, time: {system_metrics["train_epoch_time"]}s', **metrics)
            experiment._log_neptune('train', **metrics)
            # #
            metrics = experiment.metrics.valid_metrics[e].avg
            experiment._log(f'Valid epoch {e}, time: {system_metrics["valid_epoch_time"]}s', **metrics)
            experiment._log_neptune('valid', **metrics)
            # #
            experiment._log_neptune(**system_metrics)
            # #
            if experiment.low_memory:
                experiment.metrics.train_metrics[e].history = {}
                experiment.metrics.valid_metrics[e].history = {}

        experiment.verbose = experiment_state_dict['verbose']
        experiment.fit(train_loader, valid_loader, n_epochs - experiment.epoch - 1)

        return experiment

    def save(self, path):
        torch.save({
            'experiment_state_dict': {
                'rank': self.rank,
                'seed': self.seed,
                'verbose': self.verbose,
                'verbose_end': self.verbose_end,
                'verbose_ndigits': self.verbose_ndigits,
                'verbose_step': self.verbose_step,
                'base_dir': self.base_dir,
                'jupyters_path': self.jupyters_path,
                'notebook_name': self.notebook_name,
                'experiment_name': self.experiment_name,
                'epoch': self.epoch,
                'best_saving': self.best_saving,
                'last_saving': self.last_saving,
                'is_train': self.is_train,
                'use_progress_bar': self.use_progress_bar,
                'low_memory': self.low_memory,
            },
            'neptune_state_dict': {
                'params': self.neptune_params,
                'name': self.experiment_name,
            },
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_state_dict': self.metrics.state_dict(),
            'system_metrics_state_dict': self.system_metrics.state_dict(),
        }, path)

    def load(self, path):
        """
        Load weights for model and optimizer.
        If you want to load state of experiment you can use "resume" init
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

    def _low_memory(self):
        if self.low_memory:
            # prune odd history of train/valid metrics
            self.metrics.train_metrics[self.epoch].history = {}
            self.metrics.valid_metrics[self.epoch].history = {}

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
        self.system_metrics.update_epoch()

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
