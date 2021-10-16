from scipy.stats import truncexpon
import torch
import torchvision.transforms as T

import numpy as np

from .utils import *


class ExponentialSampler:
    def __init__(self, b: float = 2.5, min: float = 5, max: float = 40):
        """Initializes a sampler that draws values from a truncated exponential
        distribution, the higher b the more uniform will be the samples.

        Args:
            b (float, optional): Decay of the exponential. Defaults to 2.5.
            min (float, optional): Minimum value to draw. Defaults to 5.
            max (float, optional): Maximum value to draw. Defaults to 40.
        """
        self.b = b
        self.min = min
        self.max = max

    def __call__(self, size: int = 1) -> np.ndarray:
        """Draws size samples from the distribution

        Args:
            size (int, optional): Samples to draw. Defaults to 1.

        Returns:
            np.ndarray: Samples
        """
        samples = truncexpon.rvs(2.5, size=size) * \
            (self.max-self.min) / self.b + self.min
        return samples.astype(int)


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
    virus_mask = torch.rand_like(images[:, original_channel]) < virus_rate

    images[:, virus_channel] = images[:, original_channel] * virus_mask.float()
    images[:, original_channel] = images[:,
                                         original_channel] * (~virus_mask).float()

    return images


class VirusGenerator:
    def __init__(self, n_channels, image_size, n_CAs, CA, virus_rate=0.1, iter_func=ExponentialSampler()):
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_CAs = n_CAs
        self.CA = CA
        self.virus_rate = virus_rate
        self.iter_func = iter_func

        self.alpha_channel = CA.alpha_channel
        if type(self.alpha_channel) == list:
            self.alpha_channel = self.alpha_channel[0]

        self.model_device = self.CA.device

    def __call__(self, n_images, device):
        start_point = make_seed(n_images, self.n_channels, self.image_size,
                                self.n_CAs, self.alpha_channel, self.model_device)

        batch_size = 32
        i = 0
        while i < n_images:
            start_point[i:i+batch_size] = self.CA.evolve(
                start_point[i:i+batch_size], self.iter_func()[0])
            i += batch_size

        # TODO check if the line below should be deactivated in some cases
        start_point = add_virus(start_point, -2, -1, self.virus_rate)
        return start_point.to(device)
