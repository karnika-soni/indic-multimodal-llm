import torch

from utils.seed import set_seed

from utils.device import get_device

from models.indic_transformer import IndicTransformer

from training.trainer import Trainer



def main():

    set_seed(42)


    device = get_device()


    print(
        "Training on:",
        device
    )


    model = IndicTransformer(
        vocab_size=30000
    )


    model.to(device)



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )


    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )


    for epoch in range(10):


        loss = trainer.train_epoch()


        print(
            f"Epoch {epoch}: {loss}"
        )



if __name__ == "__main__":

    main()
