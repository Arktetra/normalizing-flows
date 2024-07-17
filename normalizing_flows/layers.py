"""Module for creating layers to use in normalizing flows."""

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):

    """Creates a Coupling Layer.

    Args:
    ----
        in_features (int): The size of each input sample.
        hidden_features (int): The size of the hidden layers.
        out_features (int): The size of each output sample.
        device (str): The device on which to compute. Can be "cpu" or "cuda"

    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, device: str):
        super().__init__()

        self.scale_nn = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.Tanh()
        )

        self.translate_nn = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

        self.to(device)

    def forward(self, z: torch.tensor, log_det_inv: torch.tensor, mask: torch.tensor, training: bool = True):
        """Performs a forward pass in the Coupling Layer.

        Args:
        ----
            z (torch.tensor): The input sample to the coupling layer.
            log_det_inv (torch.tensor): The log determinant of the jacobian.
            mask (torch.tensor): The mask for masking the input sample.
            training (bool, optional): Determines whether the forward pass is for training or sampling. Defaults to True.

        """
        z_masked = mask * z
        reversed_mask = (1 - mask)
        s = self.scale_nn(z_masked)
        t = self.translate_nn(z_masked)
        s = s * reversed_mask
        t = t * reversed_mask

        if training:
            z = (
                reversed_mask
                * (z * torch.exp(s) + t)
                + z_masked
            )
            log_det_inv += torch.sum(s, 1)
        else:
            z = (
                reversed_mask
                * (z * torch.exp(-s) - t * torch.exp(-s))
                + z_masked
            )
            log_det_inv -= torch.sum(s, 1)

        return z, log_det_inv