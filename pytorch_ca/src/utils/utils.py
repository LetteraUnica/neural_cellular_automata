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


def number_of_params(model: nn.Module):
    """Given a torch model returns the number of parameters"""
    return sum(p.numel() for p in model.parameters())


def parameters_to_vector(model: nn.Module):
    """Given a torch model returns a torch tensor with all its parameters"""
    return torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor):
    """Computes the cosine similarity between two vectors"""
    v1, v2 = v1 / v1.norm(), v2 / v2.norm()
    return v1 @ v2


def model_distance(model1: nn.Module, model2: nn.Module):
    """Computes the distance between the parameters of two models"""
    p1, p2 = parameters_to_vector(model1), parameters_to_vector(model2)
    return nn.MSELoss()(p1, p2)
