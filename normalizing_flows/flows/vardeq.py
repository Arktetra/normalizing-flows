"""A module for variational dequantization."""

import torch
import torch.nn as nn

import numpy as np

from normalizing_flows.flows.dequantization import Dequantization

from typing import Tuple

class VariationalDequantization(Dequantization):

    """A module for performing variational dequantization.

    Args:
    ----
        num_coupling_layers (int): The number of coupling layers in the variational
        dequantization.
        network (nn.Module): The network to be used by the coupling layers.
        alpha (float): A small constant used for scaling the original input.

    """

    def __init__(self, num_coupling_layers: int, network: nn.Module, alpha: float = 1e-5):
        super().__init__(alpha = alpha)

        self.num_coupling_layers = num_coupling_layers
        self.layers = network

    def forward(
        self,
        z: torch.Tensor,
        log_det_inv: torch.Tensor,
        mask: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass of dequantization.

        Args:
        ----
            z (torch.Tensor): an input consisting of discrete values in forward flow and
            dequantized values in reversed flow.
            log_det_inv (torch.Tensor): the log of the determinant of the Jacobian from
            the previous invertible function.
            mask (torch.tensor): mask to apply.
            reverse (bool, optional): the direction of the flow. Defaults to False.

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: output consisting of dequantized values
            in forward flow and discrete values in
            reversed flow, and the log of the determinant of the Jacobian of the current
            invertible function.

        """
        if not reverse:
            z, log_det_inv = self.dequant(z, log_det_inv, mask)
            z, log_det_inv = self.sigmoid(z, log_det_inv, invert=True)
        else:
            z, log_det_inv = self.sigmoid(z, log_det_inv, invert=False)
            z = z * self.quants
            log_det_inv += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.quants - 1).to(torch.int32)

        return z, log_det_inv

    def dequant(self, z: torch.Tensor, log_det_inv: torch.Tensor,  mask: torch.tensor,):
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
            deq_noise, log_det_inv = self.layers[i](deq_noise, log_det_inv, mask, reverse = False)

        deq_noise, log_det_inv = self.sigmoid(deq_noise, log_det_inv, invert = False)

        z = (z + deq_noise) / 256.0
        log_det_inv -= np.log(256.0) * np.prod(z.shape[1:])

        return z, log_det_inv