"""
Text generation utilities.

Supports:

- greedy decoding
- temperature sampling
- top-k sampling
- nucleus (top-p) sampling

"""

import torch
import torch.nn.functional as F



def generate_text(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None
):

    """
    Autoregressive text generation.


    Input:

    idx:

    existing token IDs

    Shape:

    (batch, sequence_length)



    Output:

    extended token sequence

    """



    model.eval()



    for _ in range(max_new_tokens):


        # Only keep model context window

        context = idx[
            :,
            -model.max_length:
        ]



        # Get predictions

        with torch.no_grad():

            logits = model(
                context
            )


        # Last token prediction

        logits = logits[:, -1, :]



        # Temperature scaling

        logits = logits / temperature



        # Convert to probabilities

        probabilities = F.softmax(
            logits,
            dim=-1
        )



        # Top-k filtering

        if top_k is not None:

            probabilities = apply_top_k(
                probabilities,
                top_k
            )


        # Top-p filtering

        if top_p is not None:

            probabilities = apply_top_p(
                probabilities,
                top_p
            )



        # Sample next token

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )


        # Append token

        idx = torch.cat(
            [
                idx,
                next_token
            ],
            dim=1
        )



    return idx





def apply_top_k(
    probabilities,
    k
):

    """
    Keep only k highest probability tokens.

    Example:

    Vocabulary:

    30000 tokens


    top-k=50:

    keep only 50 candidates

    """

    values, _ = torch.topk(
        probabilities,
        k
    )


    threshold = values[:, -1].unsqueeze(
        -1
    )


    probabilities = torch.where(
        probabilities < threshold,
        torch.zeros_like(probabilities),
        probabilities
    )


    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True
    )


    return probabilities





def apply_top_p(
    probabilities,
    p
):

    """
    Nucleus sampling.


    Keep smallest set of tokens
    whose cumulative probability >= p.


    Example:

    p=0.9


    Keep tokens contributing
    to 90% probability mass.

    """


    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True
    )


    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1
    )


    mask = cumulative_probs > p



    # keep first token above threshold

    mask[:,1:] = mask[:,:-1].clone()

    mask[:,0] = False



    sorted_probs[mask] = 0



    probabilities = torch.zeros_like(
        probabilities
    )


    probabilities.scatter_(
        -1,
        sorted_indices,
        sorted_probs
    )



    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True
    )


    return probabilities
