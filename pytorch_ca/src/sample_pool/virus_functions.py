from abc import abstractmethod
import torch
from torch.nn import functional as F

from ..utils import erode, square_mask


def apply_mask(images: torch.Tensor, virus_mask: torch.Tensor, original_channel: int, virus_channel: int) -> torch.Tensor:
    """Applies the virus mask to the given images
    Args:
        images (torch.Tensor): Images to add the virus to.
        virus_mask (torch.Tensor): Mask to apply
        original_channel (int): Alpha channel of the original cells
        virus_channel (int): Alpha channel of the virus cells

    Returns:
        torch.Tensor: Images with the virus added
    """
    virus_mask = virus_mask.float()
    images[:, virus_channel] = images[:, original_channel] * virus_mask
    images[:, original_channel] = images[:, original_channel] * (1 - virus_mask)

    return images


def add_virus(images: torch.Tensor, original_channel: int,
              virus_channel: int, virus_rate: float = 0.1) -> torch.Tensor:
    """Adds a virus to the given images

    Args:
        images (torch.Tensor): Images to add the virus to.
        original_channel (int): Alpha channel of the original cells
        virus_channel (int): Alpha channel of the virus cells
        virus_rate (float, optional): Ratio of cells to add the virus.
            Defaults to 0.1

    Returns:
        torch.Tensor: Images with the virus added
    """
    virus_mask = torch.rand_like(images[:, original_channel], device=images.device) < virus_rate

    return apply_mask(images, virus_mask, original_channel, virus_channel)


class RandomVirus():
    def __init__(self, virus_rate: float = 0.1):
        self.virus_rate = virus_rate

    def add_virus(self, images: torch.Tensor, original_channel: int, virus_channel: int) -> torch.Tensor:
        return add_virus(images, original_channel, virus_channel, self.virus_rate)


class SquareVirus():
    def __init__(self, square_side: int = 3, edge_distance: int = 0):
        self.square_side = square_side
        self.edge_distance = edge_distance

    def add_virus(self, images: torch.Tensor, original_channel: int, virus_channel: int) -> torch.Tensor:
        original_cells = (images[:, original_channel].unsqueeze(1) > 0.1).float()
        virus_center_masks = erode(original_cells, erosion_depth=self.square_side // 2 + self.edge_distance)

        virus_masks = torch.zeros_like(images[:, 0], dtype=int)
        for i in range(images.size()[0]):
            virus_centers = torch.nonzero(virus_center_masks[i, 0])
            center = virus_centers[torch.randint(0, virus_centers.size()[0], (1,))]
            virus_masks[i] = square_mask(images.size()[-1], center.squeeze(), self.square_side)

        return apply_mask(images, virus_masks, original_channel, virus_channel)
