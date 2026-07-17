"""
Evaluation loop.

Runs model on validation data.

Calculates:

- validation loss
- perplexity

"""

import torch

from tqdm import tqdm

from evaluation.metrics import (
    calculate_perplexity
)



@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    loss_fn,
    device
):


    model.eval()


    total_loss = 0


    batches = 0



    progress = tqdm(
        dataloader,
        desc="Evaluating"
    )



    for batch in progress:


        inputs = batch["input"].to(
            device
        )


        targets = batch["target"].to(
            device
        )


        logits = model(
            inputs
        )


        loss = loss_fn(
            logits.view(
                -1,
                logits.size(-1)
            ),

            targets.view(-1)
        )


        total_loss += loss.item()


        batches += 1



    avg_loss = (
        total_loss / batches
    )


    perplexity = calculate_perplexity(
        avg_loss
    )



    return {

        "loss": avg_loss,

        "perplexity": perplexity

    }
