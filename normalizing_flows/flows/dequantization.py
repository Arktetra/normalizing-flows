"""A module for dequantization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple

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