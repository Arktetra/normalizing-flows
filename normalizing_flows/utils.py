"""A module containing utility functions for normalizing flows."""

import torch

def spatial_checkerboard_mask(w: int, h: int, c: int, invert = False) -> torch.Tensor:
    """Creates a spatial checkerboard mask.

    Args:
    ----
        w (int): the width of the input.
        h (int): the height of the input.
        c (int): the number of channels in the input.
        invert (bool, optional): Determines whether to invert the mask or not. Defaults to False.

    Returns:
    -------
        torch.Tensor: the mask.

    """
    x, y = torch.arange(w, dtype = torch.int32), torch.arange(h, dtype = torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing = "ij")

    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, w, h)
    mask.expand(1, c, -1, -1)

    if invert:
        mask = 1 - mask

    return mask


