# -*- coding: utf-8 -*-
import os


def load():
    import gdown
    import pandas as pd
    file_id = '1DJDyB0vow-WSoS-6Wk1Jm9ItLMQvQBel'
    mnist_path = '/tmp/mnist'
    csv_path = f'{mnist_path}/train.csv'
    os.makedirs(mnist_path, exist_ok=True)
    if not os.path.exists(f'{mnist_path}/MNIST.zip'):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', f'{mnist_path}/MNIST.zip', quiet=False)
    if not os.path.exists(csv_path):
        os.system(f'unzip {mnist_path}/MNIST.zip -d {mnist_path}')
    return pd.read_csv(csv_path)
