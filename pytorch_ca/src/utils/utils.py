import torch
from torch import nn

from random import randint


def make_seed(n_images: int,
              n_channels: int,
              image_size: int,
              n_CAs: int = 1,
              alpha_channel: int = -1,
              device: torch.device = "cpu") -> torch.Tensor:
    """Makes n_images seeds to start the CA, the seed is a black dot

    Args:
        n_images (int): Number of seed images to generate
        n_channels (int): Number of channels per image
        image_size (int): Side of the square image
        alpha_channel (int): channel to insert the seed. Defaults to -1

        device (torch.device, optional): Device where to save the images.
            Defaults to "cpu".

    Returns:
        torch.Tensor: Seed images
    """
    start_point = torch.zeros(
        (n_images, n_channels+n_CAs, image_size, image_size), device=device)
    start_point[:, alpha_channel, image_size//2, image_size//2] = 1.
    return start_point


def side(size, constant_side=False):
    """Return size of the side to be used to erase square portion of the images"""
    if constant_side:
        return size//2
    return randint(size//6, size//2)


def make_squares(images, target_size=None, side=side, constant_side=False):
    """Sets square portion of input images to zero"""
    images = images.clone()
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x = randint(target_size//2-target_size//4,
                    target_size//2+target_size//4)
        y = randint(target_size//2-target_size//4,
                    target_size//2+target_size//4)
        images[i, :, x-side(target_size, constant_side)//2:x+side(target_size, constant_side) //
               2, y-side(target_size, constant_side)//2:y+side(target_size, constant_side)//2] = 0.

    return images.clone()


def checkered_mask(matrix_size):
    grid = torch.zeros(matrix_size, matrix_size)
    for i in range(matrix_size):
        for j in range(matrix_size):
            if (i+j) % 2 == 0:
                grid[i, j] = 1
    return grid.unsqueeze(0).unsqueeze(0)
