# -*- coding: utf-8 -*-
import os
import random
import time

import torch
import numpy as np

from .torch_gpu import TorchGPUExperiment


class TorchTPUExperiment(TorchGPUExperiment):

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
        seed=42,
        verbose=True,
        verbose_step=1,
        verbose_ndigits=5,
        base_dir='./saved_models',
        jupyters_path=None,
        notebook_name=None,
        experiment_name=None,
        neptune=None,
        neptune_params=None,
        best_saving=True,
        last_saving=True,
        **kwargs,
    ):
        if rank == 0:
            time.sleep(1)
        # #
        self.xm = xm
        self.pl = pl
        self.xser = xser,
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            rank=rank,
            seed=seed,
            verbose=verbose,
            verbose_end='\n',
            verbose_ndigits=verbose_ndigits,
            verbose_step=verbose_step,
            base_dir=base_dir,
            jupyters_path=jupyters_path,
            notebook_name=notebook_name,
            experiment_name=experiment_name,
            neptune=neptune,
            neptune_params=neptune_params,
            best_saving=best_saving,
            last_saving=last_saving,
            **kwargs,
        )

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

    def _rebuild_loader(self, loader):
        para_loader = self.pl.ParallelLoader(loader, [self.device])
        return para_loader.per_device_loader(self.device)

    def _get_current_metrics(self, stage):
        if stage == 'train':
            return self.__mesh_reduce_metrics(stage, **self.metrics.train_metrics[self.epoch].avg)
        elif stage == 'valid':
            return self.__mesh_reduce_metrics(stage, **self.metrics.valid_metrics[self.epoch].avg)
        else:
            raise ValueError(f'Incorrect stage: "{stage}".')

    def save(self, path):
        self.model.eval()
        self.xm.save(self.model.state_dict(), path)
        if self.rank == 0:
            metrics_path = os.path.join(os.path.dirname(path), 'metrics.pt')
            torch.save({
                'metrics_state_dict': self.metrics.state_dict(),
            }, metrics_path)

    def load(self, path):
        raise ValueError

    @classmethod
    def resume(cls, *args, **kwargs):
        raise

    def destroy(self):
        if self.rank == 0 and self.neptune:
            self.neptune.stop()

    def optimizer_step(self):
        self.xm.optimizer_step(self.optimizer)

    def _seed_everything(self, seed):
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
            msg = self._prepare_msg(msg, *args, **kwargs)
            self.xm.master_print(msg)

    def _log(self, msg, *args, **kwargs):
        msg = self._prepare_msg(msg, *args, **kwargs)
        if self.verbose:
            self._print(msg)
        if self.rank == 0:
            with open(self.log_path, 'a+') as logger:
                self.xm.master_print(f'{msg}', fd=logger)

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
