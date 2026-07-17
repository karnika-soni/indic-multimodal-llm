"""
Checkpoint management.

Handles:

- saving model weights
- saving optimizer state
- saving scheduler state
- resuming training
- experiment recovery


Used for fault-tolerant training.
"""

import os
import torch



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    loss,
    path
):
    """
    Save complete training state.


    Stores:

    1. Model parameters
    2. Optimizer state
    3. Scheduler state
    4. Epoch number
    5. Current loss


    """

    checkpoint = {


        "epoch": epoch,


        "model_state_dict":
            model.state_dict(),


        "optimizer_state_dict":
            optimizer.state_dict(),


        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler
            else None,


        "loss": loss

    }



    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    torch.save(
        checkpoint,
        path
    )



    print(
        f"Checkpoint saved: {path}"
    )





def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cuda"
):
    """
    Restore training state.

    Returns:

    starting epoch
    """



    checkpoint = torch.load(
        path,
        map_location=device
    )



    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )



    if optimizer:

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )



    if scheduler and checkpoint[
        "scheduler_state_dict"
    ]:


        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )



    epoch = checkpoint[
        "epoch"
    ]


    loss = checkpoint[
        "loss"
    ]



    print(
        f"""
        Loaded checkpoint

        Epoch: {epoch}

        Loss: {loss}
        """
    )



    return epoch
