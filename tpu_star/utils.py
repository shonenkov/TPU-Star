# -*- coding: utf-8 -*-
import os
import random

import torch
import torch.utils.data
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def TPULoaderWrapper(
    dataset,
    xm,
    batch_size=1,
    shuffle=False,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    num_workers=0,
):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader
