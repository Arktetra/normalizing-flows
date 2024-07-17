"""Module for creating real NVP."""

import torch
import torch.nn as nn
import torch.distributions as D

from normalizing_flows.layers import CouplingLayer

from typing import Tuple

class RealNVP(nn.Module):

    """Creates a RealNVP.

    Args:
    ----
        num_coupling_layers (int): The number of coupling layers in the RealNVP.
        in_features (int): The size of the each input sample.
        hidden_features (int): The size of the hidden layers in the coupling layers.
        out_features (int): The size of the each output sample.
        device (str): The device to compute on. Can be "cpu" or "cuda"

    """

    def __init__(
        self,
        num_coupling_layers: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        device: str
    ):
        super().__init__()

        self.num_coupling_layers = num_coupling_layers
        self.device = device

        self.base_dist = D.independent.Independent(
            D.normal.Normal(
                loc = torch.tensor([0.0, 0.0], device = device),
                scale = torch.tensor([1.0, 1.0], device = device)
            ),
            1
        )

        self.masks = torch.tensor(
            [[1, 0], [0, 1]] * (num_coupling_layers // 2),
            dtype = torch.float32,
            device = device
        )

        self.layers = nn.ModuleList([
            CouplingLayer(
                in_features = in_features,
                hidden_features = hidden_features,
                out_features = out_features,
                device = device
            ) for _ in range(num_coupling_layers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass of the RealNVP.

        Args:
        ----
            z (torch.Tensor): The input sample from data space.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: Output sample in latent space, log determinant of the jacobian

        """
        log_det_inv = 0

        for i in range(self.num_coupling_layers):
            z, log_det_inv = self.layers[i](z, log_det_inv, self.masks[i])

        return z, log_det_inv

    @torch.no_grad()
    def sample(self, shape: torch.Size, z_init: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample in the data space.

        Args:
        ----
            shape (torch.Size): The shape of the input in the latent space.
            z_init (torch.Tensor, optional): An input from the latent space. Defaults to None.

        """
        if z_init is None:
            z = self.base_dist.sample(shape).to(self.device)
        else:
            z = z_init.to(self.device)

        log_det_inv = 0

        for i in range(self.num_coupling_layers)[::-1]:
            z, log_det_inv = self.layers[i](z, log_det_inv, self.masks[i], training = False)

        return z, log_det_inv
