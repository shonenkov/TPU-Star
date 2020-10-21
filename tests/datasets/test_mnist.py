# -*- coding: utf-8 -*-
import os
import shutil

from tpu_star.datasets import mnist


def test_downloading_mnist_dataframe():
    if os.path.exists('/tmp/mnist'):
        shutil.rmtree('/tmp/mnist')
    df = mnist.load()
    assert df.shape[0] == 42000
    shutil.rmtree('/tmp/mnist')
