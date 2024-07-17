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

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass of the RealNVP.

        Args:
        ----
            z (torch.Tensor): The input sample from data space.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: Output sample in latent space, log determinant of the jacobian.

        """
        log_det_inv = 0

        for i in range(self.num_coupling_layers):
            z, log_det_inv = self.layers[i](z, log_det_inv, self.masks[i])

        return z, log_det_inv

    @torch.no_grad()
    def sample(
        self,
        shape: torch.Size,
        z_init: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample in the data space.


        Args:
        ----
            shape (torch.Size): The shape of the input in the latent space.
            z_init (torch.Tensor, optional): An input from the latent space. Defaults to None.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: Output sample in the data space and the log determinant of the jacobian.

        """
        if z_init is None:
            z = self.base_dist.sample(shape).to(self.device)
        else:
            z = z_init.to(self.device)

        log_det_inv = 0

        for i in range(self.num_coupling_layers)[::-1]:
            z, log_det_inv = self.layers[i](z, log_det_inv, self.masks[i], training = False)

        return z, log_det_inv

    def log_loss(self, input: torch.Tensor) -> torch.Tensor:
        """Returns the negative loglikelihood for the input from the data space.

        Args:
        ----
            input (torch.tensor): Samples from the data space.

        Returns:
        -------
            torch.Tensor: The negative loglikelihood loss.

        """
        z, log_det = self(input)
        log_pz = self.base_dist.log_prob(z)
        log_px = log_pz + log_det
        nll = -log_px
        return nll

    def training_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """Performs a single training step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): An instance of the PyTorch DataLoader.
            optimizer (torch.optim.Optimizer): An optimizer for minimizing the loss.

        Returns:
        -------
            torch.Tensor: The loss in the training step.

        """
        self.train()

        train_loss = 0

        for data in dataloader:
            optimizer.zero_grad()

            loss = self.log_loss(data.to(self.device))
            loss.mean().backward()

            optimizer.step()

            train_loss += loss.mean()

        return train_loss / len(dataloader)

    def validation_step(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Performs a single validation step.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): An instance of the PyTorch DataLoader.

        Returns:
        -------
            torch.Tensor: The loss in the validation step

        """
        self.eval()

        val_loss = 0

        with torch.inference_mode():
            for data in dataloader:
                loss = self.log_loss(data.to(self.device))
                val_loss += loss.mean()

        return val_loss / len(dataloader)