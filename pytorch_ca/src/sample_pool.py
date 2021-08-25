import torch
from torch.utils.data import Dataset

import numpy as np

from .utils import *

from typing import Tuple, Callable
"""
Samples the training images

    Parameters
    ----------
    images	tensor of samples to be trained 
    size	size of the pool
    n_channels	number of image channels
    image_size	size of the input image

"""


class SamplePool(Dataset):
    """Samples the training images"""

    def __init__(self,
                 pool_size: int,
                 n_channels: int,
                 image_size: int,
                 transform: Callable[[torch.Tensor], torch.Tensor] = None,
                 device: torch.device = "cpu") -> None:
        """Initializes the sample pool with pool_size seed images

        Args:
            pool_size (int): Number of images in the pool
            n_channels (int): Number of channels of the images
            image_size (int): Size of the square images
            transform (Callable[[torch.Tensor], torch.Tensor], optional):
                Transform to apply before returning the images, i.e. 
                center cropping, dithering, and so on.
                Defaults to None i.e. no transform is applied.
            device (torch.device, optional): Device where to store the images. Defaults to "cpu".
        """
        self.images = make_seed(pool_size, n_channels, image_size, device)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size

        if transform is None:
            def transform(x): return x
        self.transform = transform

        self.device = torch.device(device)

    def __len__(self) -> int:
        """Returns the number of images in the pool

        Returns:
            int: Number of images in the pool
        """
        return self.size

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns the images at index idx from the pool

        Args:
            idx (torch.Tensor): Index of the images to return

        Returns:
            torch.Tensor: Images at index idx
        """
        return self.transform(self.images[idx])

    def transform_pool(self,
                       transform: Callable[[torch.Tensor], torch.Tensor]):
        """Applies the given transform to all the images of the pool inplace

        Args:
            transform (Callable[[torch.Tensor], torch.Tensor]):
                Transform to apply
        """
        self.images = transform(self.images)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the pool batch_size images and returns them,
        along with the corresponding indexes

        Args:
            batch_size (int): Number of images to extract

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                The extraxted images,
                the corresponding indexes in the sample pool
        """
        idx = np.random.choice(self.size, batch_size, False)
        return self.transform(self.images[idx]), idx

    def update(self, new_images: torch.Tensor, idx: torch.Tensor,
               idx_max_loss: torch.Tensor = None):
        """Replaces images at indexes "idx" of the pool with "new_images".
        if idx_max_loss is not None then the image at idx[idx_max_loss]
        is replaced with a new seed

        Args:
            new_images (torch.Tensor): New images to replace the old ones
            idx (torch.Tensor): Indexes of the images to replace
            idx_max_loss (torch.Tensor, optional): 
                Index of the image with maximum loss. Defaults to None.
        """

        self.images[idx] = new_images.detach().to(self.device)
        if idx_max_loss is not None:
            seed = make_seed(1, self.n_channels, self.image_size)[0]
            self.images[idx[idx_max_loss]] = seed
