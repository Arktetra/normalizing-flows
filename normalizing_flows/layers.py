"""Module for creating layers to use in normalizing flows."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple

class CouplingLayer(nn.Module):

    """Creates a Coupling Layer.

    Args:
    ----
        in_features (int): The size of each input sample.
        hidden_features (int): The size of the hidden layers.
        out_features (int): The size of each output sample.
        device (str): The device on which to compute. Can be "cpu" or "cuda"

    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, device: str
    ):
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
            nn.Tanh(),
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
            nn.Linear(hidden_features, out_features),
        )

        self.to(device)

    def forward(
        self,
        z: torch.tensor,
        log_det_inv: torch.tensor,
        mask: torch.tensor,
        training: bool = True,
    ):
        """Performs a forward pass in the Coupling Layer.

        Args:
        ----
            z (torch.tensor): The input sample to the coupling layer.
            log_det_inv (torch.tensor): The log determinant of the jacobian.
            mask (torch.tensor): The mask for masking the input sample.
            training (bool, optional): Determines whether the forward pass is for
            training or sampling. Defaults to True.

        """
        z_masked = mask * z
        reversed_mask = 1 - mask
        s = self.scale_nn(z_masked)
        t = self.translate_nn(z_masked)
        s = s * reversed_mask
        t = t * reversed_mask

        if training:
            z = reversed_mask * (z * torch.exp(s) + t) + z_masked
            log_det_inv += torch.sum(s, 1)
        else:
            z = reversed_mask * (z * torch.exp(-s) - t * torch.exp(-s)) + z_masked
            log_det_inv -= torch.sum(s, 1)

        return z, log_det_inv


class Dequantization(nn.Module):

    """A module for performing dequantization.

    Args:
    ----
        alpha (float): small constant used for scaling the original input.
        quants (int): Number of possible discrete values.

    """

    def __init__(self, alpha: float = 1e-5, quants: int = 256):
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(
        self,
        z: torch.Tensor,
        log_det_inv: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass of dequantization.

        Args:
        ----
            z (torch.Tensor): An input consisting of discrete values in forward flow and
            dequantized values in reversed flow.
            log_det_inv (torch.Tensor): The log of the determinant of the Jacobian from
            the previous invertible function.
            reverse (bool, optional): The direction of the flow. Defaults to False.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: Output consisting of dequantized values
            in forward flow and discrete values in
            reversed flow, and the log of the determinant of the Jacobian of the current
            invertible function.

        """
        if not reverse:
            z, log_det_inv = self.dequant(z, log_det_inv)
            z, log_det_inv = self.sigmoid(z, log_det_inv, invert=True)
        else:
            z, log_det_inv = self.sigmoid(z, log_det_inv, invert=False)
            z = z * self.quants
            log_det_inv += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.quants - 1).to(torch.int32)

        return z, log_det_inv

    def sigmoid(
        self,
        z: torch.Tensor,
        log_det_inv: torch.Tensor,
        invert: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies inverted sigmoid in the forward flow and sigmoid in the reverse flow.

        Args:
        ----
            z (torch.Tensor): The input tensor.
            log_det_inv (torch.Tensor): The log of the determinant of the Jacobian of
            the previous invertible function.
            invert (bool, optional): Determines whether to apply sigmoid or inverted
            sigmoid.
            Defaults to False.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: The output tensor and the log of the
            determinant of the Jacobian of the invertible function.

        """
        if not invert:
            log_det_inv += (-z - 2 * F.softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)

            log_det_inv -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            log_det_inv += np.log(1 - self.alpha) * np.prod(z.shape[1:])

            log_det_inv += (-torch.log(z) - torch.log(1 - z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1 - z)

        return z, log_det_inv

    def dequant(
        self,
        z: torch.Tensor,
        log_det_inv: torch.Tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Performs dequantization on the input.

        Args:
        ----
            z (torch.Tensor): The input consisting of discrete values.
            log_det_inv (torch.Tensor): The log of the determinant of the Jacobian of
            the previous invertible function.

        Returns:
        -------
            Tuple[torch.tensor, torch.tensor]: The output consisting of dequantized
            values, and the log of the determinant of the invertible function.

        """
        z = z.to(torch.float32)
        z = (z + torch.rand_like(z).detach())  # for fitting continuous density model to discrete values
        z = (z / self.quants)  # base distribution -> Gaussian distribution with mean 0 and standard deviation 1
        log_det_inv -= np.log(self.quants) * np.prod(z.shape[1:])

        return z, log_det_inv

class VariationalDequantization(Dequantization):

    """A module for performing variational dequantization.

    Args:
    ----
        num_coupling_layers (int): The number of coupling layers in the variational
        dequantization.
        network (int): The network to be used by the coupling layers.
        alpha (float): A small constant used for scaling the original input.

    """

    def __init__(self, num_coupling_layers: int, network: nn.Module, alpha: float = 1e-5):
        super().__init__(alpha = alpha)

        self.num_coupling_layers = num_coupling_layers

    def dequant(self, z: torch.Tensor, log_det_inv: torch.Tensor):
        """Performs dequantization on the input.

        Args:
        ----
            z (torch.Tensor): The input consisting of discrete values.
            log_det_inv (torch.Tensor): The log of the determinant of the Jacobian of
            the previous invertible function.

        Returns:
        -------
            Tuple[torch.tensor, torch.tensor]: The output consisting of dequantized
            values, and the log of the determinant of the invertible function.

        """
        z = z.to(torch.float32)

        deq_noise = torch.randn_like(z).detach()
        deq_noise, log_det_inv = self.sigmoid(z, log_det_inv, invert = True)

        for i in range(self.num_coupling_layers):
            deq_noise, log_det_inv = self.layers[i](deq_noise, log_det_inv, reverse = False)

        deq_noise, log_det_inv = self.sigmoid(deq_noise, log_det_inv, invert = False)

        z = (z + deq_noise) / 256.0
        log_det_inv -= np.log(256.0) * np.prod(z.shape[1:])

        return z, log_det_inv

class LayerNormChannels(nn.Module):

    """A module for applying layer normalization across channels in an image.

    Args:
    ----
        c (int): number of channels in the image.
        eps (float): a small constant for numerical stability.

    """

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization across the channels of an image.

        Args:
        ----
            x (torch.Tensor): input image.

        Returns:
        -------
            torch.Tensor: normalized image.

        """
        mean = x.mean(dim = 1, keepdim = True)
        var = x.var(dim = 1, unbiased = False, keepdim = True)

        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta

        return y