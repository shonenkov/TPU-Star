# -*- coding: utf-8 -*-
from collections import defaultdict


class SystemMetricsGrabber:
    def __init__(self):
        self.epoch = -1
        self.metrics = {}

    def update(self, **kwargs):
        self.metrics[self.epoch].update(**kwargs)

    def update_epoch(self):
        self.epoch += 1
        self.metrics[self.epoch] = MetricsMeter()

    def state_dict(self):
        metrics_state_dict = {}
        for e in range(self.epoch + 1):
            metrics_state_dict[e] = self.metrics[e].state_dict()
        return {
            'epoch': self.epoch,
            'metrics': metrics_state_dict,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        for e, metrics_state_dict in state_dict['metrics'].items():
            self.metrics[e] = MetricsMeter()
            self.metrics[e].load_state_dict(metrics_state_dict)


class MetricsGrabber:

    def __init__(self):
        self.epoch = -1
        self.is_train = True
        self.train_metrics = {}
        self.valid_metrics = {}

    def get_best_epoch(self, key, mode):
        best_metric_value, best_epoch = None, None
        for e in range(self.epoch + 1):
            metric_value = self.valid_metrics[e].avg[key]
            if e == 0:
                best_metric_value, best_epoch = metric_value, e
            elif mode == 'min':
                if metric_value < best_metric_value:
                    best_metric_value, best_epoch = metric_value, e
            elif mode == 'max':
                if metric_value > best_metric_value:
                    best_metric_value, best_epoch = metric_value, e
            else:
                raise ValueError('Incorrect mode')
        return {'key': key, 'mode': mode, 'epoch': best_epoch, 'metric_value': best_metric_value}

    def get_last_epoch(self, key):
        return {'key': key, 'epoch': self.epoch, 'metric_value': self.valid_metrics[self.epoch].avg[key]}

    def update(self, **kwargs):
        if self.is_train:
            self.train_metrics[self.epoch].update(**kwargs)
        else:
            self.valid_metrics[self.epoch].update(**kwargs)

    def update_epoch(self):
        self.epoch += 1
        self.train_metrics[self.epoch] = MetricsMeter()
        self.valid_metrics[self.epoch] = MetricsMeter()

    def state_dict(self):
        train_metrics_state_dict, valid_metrics_state_dict = {}, {}
        for e in range(self.epoch + 1):
            train_metrics_state_dict[e] = self.train_metrics[e].state_dict()
            valid_metrics_state_dict[e] = self.valid_metrics[e].state_dict()

        return {
            'epoch': self.epoch,
            'is_train': self.is_train,
            'train_metrics': train_metrics_state_dict,
            'valid_metrics': valid_metrics_state_dict,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.is_train = state_dict['is_train']
        for e, metrics_state_dict in state_dict['train_metrics'].items():
            self.train_metrics[e] = MetricsMeter()
            self.train_metrics[e].load_state_dict(metrics_state_dict)
        for e, metrics_state_dict in state_dict['valid_metrics'].items():
            self.valid_metrics[e] = MetricsMeter()
            self.valid_metrics[e].load_state_dict(metrics_state_dict)


class MetricsMeter:

    def __init__(self):
        self.avg = defaultdict(float)
        self.sum = defaultdict(float)
        self.history = defaultdict(list)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.history[key].append(value)
            self.sum[key] += value
            self.avg[key] = self.sum[key] / len(self.history[key])

    def state_dict(self):
        history_state_dict = {}
        avg_state_dict = {}
        sum_state_dict = {}
        for key, value in self.history.items():
            history_state_dict[key] = value
        for key, value in self.avg.items():
            avg_state_dict[key] = value
        for key, value in self.sum.items():
            sum_state_dict[key] = value
        return {
            'history': history_state_dict,
            'avg': avg_state_dict,
            'sum': sum_state_dict,
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict['history'].items():
            self.history[key] = value
        for key, value in state_dict['avg'].items():
            self.avg[key] = value
        for key, value in state_dict['sum'].items():
            self.sum[key] = value
