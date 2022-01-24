import torch
import torch.nn.functional as F
from einops.layers.torch import Reduce

from typing import List


def n_largest_indexes(array: list, n: int = 1) -> list:
    """returns the indexes of the n largest elements of the array
    url:https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    """
    if n == 0:
        return None
    return sorted(range(len(array)), key=lambda x: array[x])[-n:]


def wrap_edges(images: torch.Tensor) -> torch.Tensor:
    """Pads the boundary of all images to simulate a torus

    Args:
        images (torch.Tensor): Images to pad

    Returns:
        torch.Tensor: Padded images
    """
    return F.pad(images, pad=(1, 1, 1, 1), mode='circular', value=0)


def get_living_mask(images: torch.Tensor, channels: List[int]) -> torch.Tensor:
    """Returns the a mask of the living cells in the image

    Args:
        images (torch.Tensor): images to get the living mask
        channels (List[int]): channels where to compute the living mask

    Returns:
        torch.Tensor: Living mask
    """
    if isinstance(channels, int):
        channels = [channels]
    alpha = images[:, channels, :, :]

    neighbors = F.max_pool2d(wrap_edges(alpha), 3, stride=1) > 0.1
    return torch.max(neighbors, keepdim=True)[0]


def multiple_living_mask(alphas: torch.Tensor):
    """It gives the mask where the CA rules apply in the case where multiple alphas
    are included in the CA

    Args:
        alphas (torch.Tensor):
            The first index refers to the batch, the second to the alphas,
            the third and the fourth to the pixels in the image

    Returns:
        (torch.Tensor) A tensor with bool elements with the same shape on the input tensor
        that represents where each CA rule applies
    """

    # gives the biggest alpha per pixel
    biggest = Reduce('b c w h-> b 1 w h', reduction='max')(alphas)

    # the free cells are the ones who have all of the alphas lower than 0.1
    free = biggest < 0.1

    # this is the mask where already one of the alpha is bigger than 0.1, if more than one
    # alpha is bigger than 0.1, than the biggest one wins
    old = (alphas == biggest) & (alphas >= 0.1)

    # this is the mask of the cells neighboring each alpha
    neighbor = F.max_pool2d(wrap_edges(alphas), 3, stride=1) >= 0.1

    # the cells where the CA can expand are the one who are free and neighboring
    expanding = free & neighbor

    # the CA evolves in the cells where it can expand and the ones where is already present
    mask = expanding | old

    return mask  # the mask has the same shape of alphas and the values inside are bool


def multiple_to_single(images: torch.Tensor, n_channels: int) -> torch.Tensor:
    """
    maronn maronn
    """
    return torch.cat((images[:, :3],
                      images[:, n_channels:].sum(dim=1, keepdim=True),
                      images[:, 3:n_channels]), dim=1)


def single_to_multiple(dx: torch.Tensor, shape, n_channels: int, alpha_channel: int):
    """
    è un miracolo!
    """
    dx_new = torch.zeros(shape)
    dx_new[:, :3] = dx[:, :3]
    dx_new[:, 3:n_channels] = dx[:, 4:]
    dx_new[:, alpha_channel] = dx[:, 3]
    return dx_new
