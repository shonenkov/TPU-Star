# -*- coding: utf-8 -*-
from random import random

import torch

from tpu_star.experiment.metrics import MetricsGrabber, MetricsMeter


def test_metrics_grabber_save_and_load_state_dict():
    metrics = MetricsGrabber()
    for e in range(10):
        metrics.update_epoch()
        for i in range(100):
            metrics.update(auc=random(), acc=random())

    torch.save(metrics.state_dict(), '/tmp/metrics.pt')

    loaded_metrics = MetricsGrabber()
    loaded_metrics.load_state_dict(torch.load('/tmp/metrics.pt'))

    assert metrics.epoch == loaded_metrics.epoch
    assert metrics.is_train == loaded_metrics.is_train

    assert max(metrics.train_metrics[9].history.keys()) == max(loaded_metrics.train_metrics[9].history.keys())


def test_metrics_meter_save_and_load_state_dict():
    metrics = MetricsMeter()
    for i in range(100):
        metrics.update(auc=random(), acc=random())

    torch.save(metrics.state_dict(), '/tmp/metrics.pt')

    loaded_metrics = MetricsMeter()
    loaded_metrics.load_state_dict(torch.load('/tmp/metrics.pt'))

    assert max(metrics.history.keys()) == max(loaded_metrics.history.keys())
