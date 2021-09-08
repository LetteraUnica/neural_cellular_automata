import torch
from torch.utils.data import Dataset

import numpy as np

from .utils import *

from typing import Tuple, Callable


class SamplePool(Dataset):
    """Samples the training images

    Parameters
    ----------
    images	tensor of samples to be trained 
    size	size of the pool
    n_channels	number of image channels
    image_size	size of the input image
    """

    def __init__(self,
                 pool_size: int,
                 generator: Callable[[int],torch.Tensor],
                 transform: Callable[[torch.Tensor], torch.Tensor] = None,
                 device: torch.device = "cpu") -> None:
        """Initializes the sample pool with pool_size seed images

        Args:
            pool_size (int): Number of images in the pool
            transform (Callable[[torch.Tensor], torch.Tensor], optional):
                Transform to apply before returning the images, i.e. 
                center cropping, dithering, and so on.
                Defaults to None i.e. no transform is applied.
            device (torch.device, optional): Device where to store the images.
                Defaults to "cpu".
            generator (Callable[[int,*args,**kwargs],torch.Tensor]=make_seed):
                generates the new images
        """

        self.generator = generator
        self.images = generator(pool_size)
        self.size = pool_size
        self.n_channels = self.images.shape[1]
        self.image_size = self.images.shape[2]

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

    def update(self, indexes: List[int]) -> None:
        """Replaces images at indexes "indexes" of the pool with new_images

        Args:
            indexes (List[int]): Indexes of the images to replace
        """
        if isinstance(indexes, int):
            self.images[indexes] = self.generator(1)[0]
            return    
        self.images[indexes] = self.generator(len(indexes))