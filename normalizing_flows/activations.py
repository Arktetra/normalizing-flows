"""A module containing activation functions."""

import torch
import torch.nn.functional as F

class ConcatELU(torch.nn.Module):

    """Applies ELU in both directions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies ConcatELU on the input.

        Args:
        ----
            x (torch.Tensor): an input of any shape.

        Returns:
        -------
            torch.Tensor: output of same shape as input.

        """
        return torch.cat([F.elu(x), F.elu(-x)], dim = 1)
