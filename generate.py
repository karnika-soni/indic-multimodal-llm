"""
Text generation script for Indic Decoder-only Transformer.

Pipeline:

Prompt
 |
Tokenizer
 |
Token IDs
 |
Transformer
 |
Next token prediction
 |
Sampling
 |
Generated text


Supports:
- greedy decoding
- temperature sampling
- top-k sampling
"""

import torch
import torch.nn.functional as F


from models.transformer import IndicTransformer

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer



# -----------------------------
# Config
# -----------------------------

CHECKPOINT = (
    "checkpoints/epoch_9.pt"
)

TOKENIZER_PATH = (
    "tokenizer/indic.model"
)


MAX_NEW_TOKENS = 100

TEMPERATURE = 0.8

TOP_K = 50



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# -----------------------------
# Load model
# -----------------------------

def load_model():

    model = IndicTransformer()


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(
        DEVICE
    )


    model.eval()


    return model



# -----------------------------
# Top-k sampling
# -----------------------------

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None
):


    logits = logits / temperature



    if top_k:


        values, indices = torch.topk(
            logits,
            top_k
        )


        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )


        filtered_logits[
            indices
        ] = values


        logits = filtered_logits



    probabilities = F.softmax(
        logits,
        dim=-1
    )


    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )


    return next_token



# -----------------------------
# Generate text
# -----------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens
):


    tokens = tokenizer.encode(
        prompt
    )


    input_ids = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0)


    input_ids = input_ids.to(
        DEVICE
    )



    for _ in range(max_new_tokens):


        # Forward pass

        logits = model(
            input_ids
        )



        # Last token prediction

        next_token_logits = logits[:, -1, :]



        next_token = sample_next_token(

            next_token_logits.squeeze(0),

            temperature=TEMPERATURE,

            top_k=TOP_K

        )



        next_token = next_token.unsqueeze(0).unsqueeze(0)



        input_ids = torch.cat(

            [
                input_ids,
                next_token
            ],

            dim=1

        )



    generated_tokens = (
        input_ids
        .squeeze(0)
        .tolist()
    )


    text = tokenizer.decode(
        generated_tokens
    )


    return text



# -----------------------------
# Main
# -----------------------------

def main():


    tokenizer = SentencePieceTokenizer(
        TOKENIZER_PATH
    )


    model = load_model()



    prompt = input(
        "Enter prompt: "
    )


    output = generate(

        model,

        tokenizer,

        prompt,

        MAX_NEW_TOKENS

    )


    print(
        "\nGenerated:\n"
    )


    print(output)



if __name__ == "__main__":

    main()
