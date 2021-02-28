# -*- coding: utf-8 -*-
import shutil

import albumentations as A
import torchvision
import torch
import sklearn
from sklearn.model_selection import train_test_split
from albumentations.pytorch.transforms import ToTensorV2

from tpu_star.experiment import TorchGPUExperiment
from tpu_star.datasets import mnist


def build_datasets():
    df = mnist.load()[:200]
    train_index, valid_index = train_test_split(df.index, train_size=0.8, stratify=df['label'], random_state=42)
    df_train, df_valid = df.loc[train_index], df.loc[valid_index]
    transforms = A.Compose([A.Resize(height=28, width=28, p=1.0), ToTensorV2()])
    train_dataset = mnist.MNISTDataset(df_train, transforms)
    valid_dataset = mnist.MNISTDataset(df_valid, transforms)
    return train_dataset, valid_dataset


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


class MNISTExperiment(TorchGPUExperiment):

    @staticmethod
    def calculate_metrics(targets, outputs):
        targets = targets.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy().argmax(axis=1)
        return {'acc': sklearn.metrics.accuracy_score(targets, outputs)}

    def handle_one_batch(self, batch):
        targets = batch['target'].to(self.device, dtype=torch.int64)
        outputs = self.model(batch['image'].to(self.device, dtype=torch.float32))
        loss = self.criterion(outputs, targets)
        metrics = self.calculate_metrics(targets, outputs)
        self.metrics.update(loss=loss.detach().cpu().item(), **metrics)
        if self.is_train:
            loss.backward()
            self.optimizer_step()
            self.optimizer.zero_grad()
            self.scheduler.step()


def test_run_experiment():
    lr = 0.0001 * 8
    batch_size = 32
    num_epochs = 5
    max_lr = 0.001 * 8
    pct_start = 0.1
    experiment_name = 'test-cpu-mnist'
    device = torch.device('cpu')

    mx = create_model()
    train_dataset, valid_dataset = build_datasets()

    model = mx.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        epochs=num_epochs,
    )
    experiment = MNISTExperiment(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        base_dir='/tmp/saved_models',
        experiment_name=experiment_name,
        verbose_step=10**5,
        seed=42,
        use_progress_bar=False,
        low_memory=True,
    )
    experiment.fit(train_loader, valid_loader, num_epochs)
    experiment.destroy()

    shutil.rmtree('/tmp/saved_models')


def test_resume_experiment():
    lr = 0.0001 * 8
    batch_size = 32
    num_epochs = 5
    max_lr = 0.001 * 8
    pct_start = 0.1
    experiment_name = 'test-cpu-mnist'
    device = torch.device('cpu')

    mx = create_model()
    train_dataset, valid_dataset = build_datasets()

    model = mx.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        epochs=num_epochs,
    )
    experiment = MNISTExperiment(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        base_dir='/tmp/saved_models',
        experiment_name=experiment_name,
        verbose_step=10**5,
        seed=42,
        use_progress_bar=False,
        low_memory=True,
    )
    experiment.fit(train_loader, valid_loader, 2)
    experiment.destroy()

    experiment = MNISTExperiment.resume(
        checkpoint_path=f'/tmp/saved_models/{experiment_name}/last.pt',
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=num_epochs,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        seed=43
    )
    experiment.destroy()

    shutil.rmtree('/tmp/saved_models')
