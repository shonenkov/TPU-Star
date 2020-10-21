# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from torch.utils.data import Dataset


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


class MNISTDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row['label']
        image = row.values[1:].reshape((28, 28, 1)).astype(np.uint8)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return {
            'id': row.name,
            'target': torch.tensor(target, dtype=torch.int32),
            'image': image,
        }

    def get_labels(self):
        return list(self.df['label'].values)
