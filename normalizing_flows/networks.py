"""A module for neural networks."""

import torch
import torch.nn as nn

from normalizing_flows.activations import ConcatELU
from normalizing_flows.layers import GatedConv
from normalizing_flows.layers import LayerNormChannels

class GatedConvNet(nn.Module):

    """A module for gated convolutional neural network.

    Args:
    ----
        in_channels (int): number of input channels.
        hidden_channels (int, Optional): number of hidden channels to model. Defaults to 32.
        out_channels (int, Optional): number of output channels. Defaults to -1.
        num_layers (int, Optional): number of layers in the network. Defaults to 3.

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = -1,
        num_layers: int = 3
    ):
        super().__init__()

        out_channels = out_channels if out_channels > 0 else 2 * in_channels

        layers = []
        layers += [
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = hidden_channels,
                kernel_size = 3, padding = 1
            )
        ]

        for idx in range(num_layers):
            layers += [
                GatedConv(
                    in_channels= hidden_channels,
                    hidden_channels = hidden_channels
                ),
                LayerNormChannels(c = hidden_channels)
            ]

        layers += [
            ConcatELU(),
            nn.Conv2d(
                in_channels = 2 * hidden_channels,
                out_channels = out_channels,
                kernel_size = 3,
                padding = 1
            )
        ]

        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass in a GatedConvNet.

        Args:
        ----
            x (torch.Tensor): input to the GatedConvNet.

        Returns:
        -------
            torch.Tensor: output from the GatedConvNet.

        """
        return self.nn(x)