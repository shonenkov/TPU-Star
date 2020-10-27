# -*- coding: utf-8 -*-
import torch
import torch.utils.data


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
