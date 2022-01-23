from scipy.stats import truncexpon
import torch
import torchvision.transforms as T

import numpy as np

from ..utils import *
from .virus_functions import *


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


class VirusGenerator:
    def __init__(self, n_channels: int,
                 original_channel: int,
                 virus_channel: int,
                 image_size: int,
                 n_CAs: int,
                 CA: "CAModel",
                 iter_func=ExponentialSampler(),
                 virus_func=RandomVirus()):
        self.n_channels = n_channels
        self.original_channel = original_channel
        self.virus_channel = virus_channel
        self.image_size = image_size
        self.n_CAs = n_CAs
        self.CA = CA

        self.iter_func = iter_func
        self.virus_func = virus_func

    def __call__(self, n_images, device):
        starting_state = make_seed(n_images, self.n_channels, self.image_size,
                                   self.n_CAs, self.original_channel, self.CA.device)
        batch_size = 32
        for i in range(0, n_images, batch_size):
            n_steps = self.iter_func()[0]
            starting_state[i:i+batch_size] = self.CA.evolve(
                starting_state[i:i+batch_size], n_steps)

        starting_state = self.virus_func.add_virus(
            starting_state, self.original_channel, self.virus_channel)
        return starting_state.to(device)
