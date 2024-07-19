"""A module containing utility functions for data setup."""

import torch
from sklearn import datasets


class TwoMoonsDataset(torch.utils.data.Dataset):

    """Creates a two moons dataset."""

    def __init__(self):
        super().__init__()

        data = torch.from_numpy(
            datasets.make_moons(3000, noise=0.05)[0].astype("float32")
        )
        self.data = (data - data.mean(0)) / data.std(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx, 0], self.data[idx, 1]])

def discretize(x: torch.Tensor):
    """Converts images from 0-1 to 0-255.

    Args:
    ----
        x (torch.Tensor): The input image.

    Returns:
    -------
        torch.Tensor: The discretized image.

    """
    return (x * 255).to(torch.int32)
