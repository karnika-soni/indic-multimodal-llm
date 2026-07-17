"""
Evaluation metrics for language models.

Includes:

- Cross entropy loss
- Perplexity

"""

import math



def calculate_perplexity(
    loss
):
    """
    Convert loss into perplexity.


    Formula:

    PPL = exp(loss)


    Lower is better.

    """

    return math.exp(
        loss
    )



def average_loss(
    losses
):
    """
    Average batch losses.
    """

    return sum(losses) / len(losses)
