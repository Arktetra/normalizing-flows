"""A module containing functions to setup the data."""

import torch

from typing import Tuple, Sequence


def create_dataloaders(
    dataset: torch.utils.data.Dataset,
    lengths: Sequence[int | float],
    batch_size: int = 256,
    generator: torch.Generator = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """A function to create dataloaders.

    Args:
    ----
        dataset (torch.utils.data.Dataset): The dataset to use.
        batch_size (int, optional): The number of examples in a batch. Defaults to 256.

    Returns:
    -------
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: train and
        validation dataloaders.

    """
    if not generator:
        train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    else:
        train_set, val_set = torch.utils.data.random_split(dataset, lengths, generator)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader
