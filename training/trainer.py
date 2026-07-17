"""
Training engine for Indic Transformer Language Model.

Handles:

- forward pass
- loss calculation
- backpropagation
- optimization
- validation
- mixed precision training

"""

import torch
import torch.nn as nn

from tqdm import tqdm



class Trainer:


    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        scheduler=None,
        use_amp=True
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device


        self.use_amp = use_amp


        # Mixed precision scaler

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp
        )


        self.loss_fn = nn.CrossEntropyLoss()



    def train_epoch(self):

        """
        One complete training epoch.
        """


        self.model.train()


        total_loss = 0



        progress = tqdm(
            self.train_loader,
            desc="Training"
        )


        for batch in progress:


            # Move data to GPU

            inputs = batch["input"].to(
                self.device
            )

            targets = batch["target"].to(
                self.device
            )


            self.optimizer.zero_grad()



            # Mixed precision forward

            with torch.cuda.amp.autocast(
                enabled=self.use_amp
            ):


                logits = self.model(
                    inputs
                )


                loss = self.loss_fn(
                    logits.view(
                        -1,
                        logits.size(-1)
                    ),

                    targets.view(-1)
                )



            # Backpropagation

            self.scaler.scale(
                loss
            ).backward()



            # Gradient clipping

            self.scaler.unscale_(
                self.optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )



            # Update weights

            self.scaler.step(
                self.optimizer
            )


            self.scaler.update()



            total_loss += loss.item()



            progress.set_postfix(
                loss=loss.item()
            )



        avg_loss = (
            total_loss /
            len(self.train_loader)
        )


        return avg_loss




    @torch.no_grad()
    def evaluate(self):

        """
        Validation loop.
        """


        self.model.eval()


        total_loss = 0



        for batch in self.val_loader:


            inputs = batch["input"].to(
                self.device
            )


            targets = batch["target"].to(
                self.device
            )



            logits = self.model(
                inputs
            )


            loss = self.loss_fn(
                logits.view(
                    -1,
                    logits.size(-1)
                ),

                targets.view(-1)
            )



            total_loss += loss.item()



        return (
            total_loss /
            len(self.val_loader)
        )
