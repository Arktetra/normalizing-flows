"""A module for training models."""

import torch

from tqdm.auto import tqdm

from typing import Dict, List


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    checkpoint: bool = False
) -> Dict[str, List]:
    """Trains a model for a number of epochs.

    Args:
    ----
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): A dataloader for training the
        model.
        val_dataloader (torch.utils.data.DataLoader): A dataloader for validating the
        model.
        optimizer (torch.optim.Optimizer): An optimizer for minimizing the loss.
        epochs (int): Number of epochs to train the model for.

    Returns:
    -------
        Dict[str, List]: The training and validation loss at each epoch.

    """
    results = {"train_loss": [], "val_loss": []}

    loss = 1000.0       # A high value of loss for saving the state dict of model that has the minimum loss

    tepochs = tqdm(range(epochs))
    for epoch in tepochs:
        tepochs.set_description(f"Epoch {epoch + 1}")

        train_loss = model.training_step(train_dataloader, optimizer)

        if train_loss < loss:
            loss = train_loss
            torch.save(model.state_dict, "models/mnist_flow.pth")

        val_loss = model.validation_step(val_dataloader)

        tepochs.set_postfix(train_loss=train_loss.item(), val_loss=val_loss.item())

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    return results
