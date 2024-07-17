"""A module for training models."""

import torch

from tqdm.auto import tqdm

from typing import Dict, List

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int
) -> Dict[str, List]:
    """Trains a model for a number of epochs.

    Args:
    ----
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): A dataloader for training the model.
        val_dataloader (torch.utils.data.DataLoader): A dataloader for validating the model.
        optimizer (torch.optim.Optimizer): An optimizer for minimizing the loss.
        epochs (int): Number of epochs to train the model for.

    Returns:
    -------
        Dict[str, List]: The training and validation loss at each epoch.

    """
    results = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss = model.training_step(
            train_dataloader,
            optimizer
        )

        val_loss = model.validation_step(
            val_dataloader
        )

        print(
            f"{epoch + 1}/{epochs} | "
            f"train_loss: {train_loss} | "
            f"val_loss: {val_loss}"
        )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    return results