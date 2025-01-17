"""Module for creating layers to use in normalizing flows."""

import torch
import torch.nn as nn

from normalizing_flows.activations import ConcatELU

class CouplingLayer(nn.Module):

    """Creates a Coupling Layer.

    Args:
    ----
        network (nn.Module): the network to use inside the Coupling Layer.
        device (str): the device on which to compute. Can be "cpu" or "cuda".

    """

    def __init__(
        self, network: nn.Module, device: str
    ):
        super().__init__()

        self.nn = network

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
        s, t = self.nn(z_masked).chunk(2, dim = 1)
        s = s * reversed_mask
        t = t * reversed_mask

        if training:
            z = reversed_mask * (z * torch.exp(s) + t) + z_masked
            log_det_inv += torch.sum(s, 1)
        else:
            z = reversed_mask * (z * torch.exp(-s) - t * torch.exp(-s)) + z_masked
            log_det_inv -= torch.sum(s, 1)

        return z, log_det_inv

class LayerNormChannels(nn.Module):

    """A module for applying layer normalization across channels in an image.

    Args:
    ----
        in_channels (int): number of channels in the image.
        eps (float): a small constant for numerical stability.

    """

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
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

class GatedConv(nn.Module):

    """A two layer convultional ResNet block with input gate.

    Args:
    ----
        in_channels(int): number of input channels.
        c_hidden (int): number of hidden channels.

    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(
                in_channels = 2 * in_channels,
                out_channels = hidden_channels,
                kernel_size = 3, padding = 1
            ),
            ConcatELU(),
            nn.Conv2d(
                in_channels = 2 * hidden_channels,
                out_channels = 2 * in_channels,
                kernel_size = 1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass in the GatedConv block.

        Args:
        ----
            x (torch.Tensor): input to the GatedConv block.

        Returns:
        -------
            torch.Tensor: output from the GatedConv block.

        """
        out = self.net(x)
        val, gate = out.chunk(2, dim = 1)
        return x + val * torch.sigmoid(gate)