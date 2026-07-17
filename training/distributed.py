"""
Distributed training utilities.

Uses PyTorch DistributedDataParallel (DDP).

Purpose:

Single GPU:

    model
      |
    batch


Multi GPU:

    GPU 0 -> batch 0
    GPU 1 -> batch 1
    GPU 2 -> batch 2


Each GPU trains a copy of the model.

Gradients are synchronized
after backward pass.

"""

import os

import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel





def setup_distributed():

    """
    Initialize distributed process group.

    Each GPU becomes one process.

    """


    dist.init_process_group(
        backend="nccl"
    )


    local_rank = int(
        os.environ["LOCAL_RANK"]
    )


    torch.cuda.set_device(
        local_rank
    )


    return local_rank





def wrap_model_ddp(
    model,
    local_rank
):

    """
    Move model to GPU
    and wrap with DDP.
    """


    model = model.to(
        local_rank
    )


    model = DistributedDataParallel(
        model,
        device_ids=[
            local_rank
        ]
    )


    return model





def cleanup_distributed():

    """
    Shutdown distributed training.
    """

    dist.destroy_process_group()
