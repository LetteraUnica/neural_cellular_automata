import random
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
                 generator: Callable[[int], torch.Tensor],
                 transform: Callable[[torch.Tensor], torch.Tensor] = None,
                 device: torch.device = "cpu",
                 indexes_max_loss_size=32) -> None:
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
            indexes_max_loss_size (int, optional): Maximum number of images to 
                replace with seed states. Defaults to 32.
        """

        self.generator = generator
        self.size = pool_size
        self.device = torch.device(device)
        self.images = generator(pool_size, self.device)
        self.n_channels = self.images.shape[1]
        self.image_size = self.images.shape[2]

        if transform is None:
            def transform(x): return x
        self.transform = transform

        self.all_indexes = set(range(self.size))
        self.indexes_max_loss = set()
        self.indexes_max_loss_size = indexes_max_loss_size

        self.evolutions_per_image = np.zeros(self.size)

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
        idx = random.sample(self.all_indexes -
                            self.indexes_max_loss, batch_size)
        return self.transform(self.images[idx]).clone(), idx

    def replace(self, indexes: List[int]) -> None:
        """Replaces images at indexes "indexes" of the pool with new_images

        Args:
            indexes (List[int]): Indexes of the images to replace with seed states
        """
        if isinstance(indexes, int):
            self.images[indexes] = self.generator(1, self.device)[0]
            return
        self.images[indexes] = self.generator(len(indexes), self.device)
        self.evolutions_per_image[indexes] = 0

    def resample_indexes(self, indexes: List[int], idx_to_replace: List[int]):
        if idx_to_replace is not None:
            idx_to_replace = [indexes[i] for i in idx_to_replace]
            self.indexes_max_loss.update(idx_to_replace)

        if len(self.indexes_max_loss) > self.indexes_max_loss_size:
            self.replace(list(self.indexes_max_loss))
            self.indexes_max_loss = set()
    
    def update_evolution_iters(self, indexes: List[int], evolution_iters):
        if evolution_iters is not None:
            self.evolutions_per_image[indexes] += evolution_iters

    def update(self, indexes: List[int],
               images: torch.Tensor,
               idx_to_replace: List[int] = None,
               evolution_iters = None) -> None:
        """Updates the images in the pool with new images at the given indexes.

        Args:
            indexes (List[int]): Indexes of the images to update
            images (torch.Tensor): New images to insert at the given indexes
            indexes_max_loss (List[int], optional): Indexes of the images with
                maximum loss, these images will be replaced with seed states.
                Default None, no image will be replaced by seed states
        """
        self.images[indexes] = images.detach().to(self.device)

        self.resample_indexes(indexes, idx_to_replace)

        self.update_evolution_iters(indexes, evolution_iters)

    def reset(self):
        self.images = self.generator(self.size, self.device)

    def size(self):
        return self.size
