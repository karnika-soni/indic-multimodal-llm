"""
Training script for Indic Decoder-only Transformer LLM.

Pipeline:

Raw Indic Text
        |
SentencePiece tokenizer
        |
IndicTextDataset
        |
DataLoader
        |
IndicTransformer
        |
DistributedDataParallel
        |
Trainer
        |
Checkpoint


Supports:
- single GPU training
- multi GPU training with DDP
- mixed precision
"""

import os

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# project imports

from models.transformer import IndicTransformer

from data.dataset import IndicTextDataset

from training.trainer import Trainer

from training.distributed import setup_distributed

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer



# -----------------------------
# Config
# -----------------------------

BATCH_SIZE = 32

MAX_LENGTH = 128

EPOCHS = 10

LR = 3e-4

CHECKPOINT_DIR = "checkpoints"

TEXT_FILE = "data/train.txt"

TOKENIZER_PATH = "tokenizer/indic.model"



# -----------------------------
# Load text
# -----------------------------

def load_text(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        texts = f.readlines()


    return texts



# -----------------------------
# Main training
# -----------------------------

def main():

    # -------------------------
    # Distributed setup
    # -------------------------

    if "LOCAL_RANK" in os.environ:

        local_rank = setup_distributed()

        device = torch.device(
            f"cuda:{local_rank}"
        )

        distributed = True

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        local_rank = 0

        distributed = False



    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = SentencePieceTokenizer(
        TOKENIZER_PATH
    )



    # -------------------------
    # Dataset
    # -------------------------

    texts = load_text(
        TEXT_FILE
    )


    dataset = IndicTextDataset(

        texts=texts,

        tokenizer=tokenizer,

        max_length=MAX_LENGTH

    )



    train_size = int(
        0.9 * len(dataset)
    )


    val_size = (
        len(dataset)
        -
        train_size
    )


    train_dataset, val_dataset = random_split(

        dataset,

        [
            train_size,
            val_size
        ]

    )



    # -------------------------
    # DataLoaders
    # -------------------------

    if distributed:


        train_sampler = DistributedSampler(
            train_dataset
        )


        train_loader = DataLoader(

            train_dataset,

            batch_size=BATCH_SIZE,

            sampler=train_sampler

        )


    else:


        train_loader = DataLoader(

            train_dataset,

            batch_size=BATCH_SIZE,

            shuffle=True

        )



    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE

    )



    # -------------------------
    # Model
    # -------------------------

    model = IndicTransformer()



    model.to(device)



    # DDP wrapper

    if distributed:

        model = DDP(

            model,

            device_ids=[
                local_rank
            ]

        )



    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=LR

    )



    # -------------------------
    # Trainer
    # -------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer,

        device=device,

        use_amp=True

    )



    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(EPOCHS):


        if distributed:

            train_loader.sampler.set_epoch(
                epoch
            )



        train_loss = trainer.train_epoch()


        val_loss = trainer.evaluate()



        if local_rank == 0:


            print(
                f"""
Epoch {epoch+1}

Train Loss:
{train_loss:.4f}

Validation Loss:
{val_loss:.4f}

"""
            )


            save_checkpoint(
                model,
                optimizer,
                epoch
            )




# -----------------------------
# Checkpoint saving
# -----------------------------

def save_checkpoint(
    model,
    optimizer,
    epoch
):

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )


    # remove DDP wrapper

    if isinstance(
        model,
        DDP
    ):

        state_dict = model.module.state_dict()

    else:

        state_dict = model.state_dict()



    torch.save(

        {

            "epoch": epoch,

            "model": state_dict,

            "optimizer":
                optimizer.state_dict()

        },

        f"{CHECKPOINT_DIR}/epoch_{epoch}.pt"

    )



if __name__ == "__main__":

    main()
