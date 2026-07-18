"""
Training script for Multimodal mBART.

Pipeline:

Image
 |
Detectron2 visual features
 |
Multimodal Dataset
 |
mBART Encoder-Decoder
 |
Cross Entropy Loss
 |
Optimization
 |
BLEU evaluation

"""


import torch

from torch.utils.data import DataLoader

from torch.optim import AdamW

from transformers import MBartTokenizer


from models.mbart_multimodal import (
    MultimodalMBart
)


from data.multimodal_dataset import (
    MultimodalDataset
)


from evaluation.bleu import (
    calculate_bleu
)



# -------------------------
# Config
# -------------------------

MODEL_NAME = (
    "facebook/mbart-large-50"
)


BATCH_SIZE = 8


EPOCHS = 5


LR = 5e-5


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# -------------------------
# Load tokenizer
# -------------------------

def load_tokenizer():

    tokenizer = MBartTokenizer.from_pretrained(
        MODEL_NAME
    )

    return tokenizer



# -------------------------
# Training epoch
# -------------------------

def train_epoch(
    model,
    loader,
    optimizer
):


    model.train()


    total_loss = 0



    for batch in loader:


        image_features = (
            batch["image_features"]
            .to(DEVICE)
        )


        input_ids = (
            batch["input_ids"]
            .to(DEVICE)
        )


        attention_mask = (
            batch["attention_mask"]
            .to(DEVICE)
        )


        labels = (
            batch["labels"]
            .to(DEVICE)
        )



        optimizer.zero_grad()



        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            image_features=image_features,

            labels=labels

        )



        loss = outputs.loss



        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    return (
        total_loss /
        len(loader)
    )



# -------------------------
# Main
# -------------------------

def main():


    tokenizer = load_tokenizer()



    # Dataset

    dataset = MultimodalDataset(

        tokenizer=tokenizer

    )



    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )



    # Model

    model = MultimodalMBart(

        model_name=MODEL_NAME

    )


    model.to(
        DEVICE
    )



    optimizer = AdamW(

        model.parameters(),

        lr=LR

    )



    # Training

    for epoch in range(EPOCHS):


        loss = train_epoch(

            model,

            loader,

            optimizer

        )


        print(
            f"Epoch {epoch+1} Loss: {loss:.4f}"
        )



    torch.save(

        model.state_dict(),

        "checkpoints/mbart_multimodal.pt"

    )



if __name__ == "__main__":

    main()
