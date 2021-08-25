import torch
import torchvision.transforms as T

import numpy as np
import pylab as pl
from random import randint

from matplotlib import cm

from typing import Tuple


def RGBAtoFloat(images: torch.Tensor) -> torch.Tensor:
    """Converts images from 0-255 range into 0-1 range

    Args:
        images (torch.Tensor): Images in 0-255 range

    Returns:
        torch.Tensor: Images in 0-1 range
    """
    return torch.clip(images.float() / 255, 0., 1.)


def FloattoRGBA(images: torch.Tensor) -> torch.Tensor:
    """Converts images from 0-1 range into 0-255 range

    Args:
        images (torch.Tensor): Images in 0-1 range

    Returns:
        torch.Tensor: Images in 0-255 range
    """
    return torch.clip((images * 255), 0, 255).type(torch.uint8)


def RGBAtoRGB(images: torch.Tensor) -> torch.Tensor:
    """Converts a 0-1 RGBA image into RGB

    Args:
        images (torch.Tensor): RGBA images in 0-1 range

    Returns:
        torch.Tensor: RGB images in 0-1 range
    """

    if len(images.size()) < 4:
        images = torch.unsqueeze(images, 0)
    return torch.clip(images[:, :3, :, :] * images[:, 3, :, :] * 255 + (1-images[:, 3, :, :])*255, 0, 255).type(torch.uint8)


def GrayscaletoCmap(image: torch.Tensor, cmap="viridis") -> torch.Tensor:
    """Converts a 0-1 2D tensor representing an image into a colormap

    Args:
        image (torch.Tensor): 2D tensor with no channel dimension
        cmap (str, optional): color map to use, must be present in the
            matplotlib package. Defaults to "viridis".

    Returns:
        torch.Tensor: RGB image in 0-1 range
    """
    if len(image.size()) > 2:
        Exception(
            f"images must be a 1d or 2d tensor, got {image.shape} instead")
        return

    with torch.no_grad():
        scale = torch.max(image) - torch.min(image)
        if scale < 1e-6:
            image = torch.zeros_like(image)
        else:
            image = (image - torch.min(image)) / scale

    viridis = cm.get_cmap(cmap)
    return torch.tensor(viridis(image)).permute(2, 0, 1)


def center_crop(images: torch.Tensor, size: int) -> torch.Tensor:
    """Center crops a batch of images

    Args:
        images (torch.Tensor): images to center crop
        size (int): side of the square to crop

    Returns:
        torch.Tensor: Center portion of the images
    """
    return T.CenterCrop(size)(images)


def pad(images: torch.Tensor, padding: int, fill_value=0.) -> torch.Tensor:
    """Pads the images by adding "padding" pixels in both dimensions

    Args:
        images (torch.Tensor): Images to pad
        padding (int): amount of padding
        fill_value (float, optional): Value to fill the padding. Defaults to 0.

    Returns:
        torch.Tensor: Padded images
    """
    return T.Pad(padding//2, fill=fill_value)(images)


def imshow(image: torch.Tensor, fname: str = None) -> torch.Tensor:
    """Prints an image

    Args:
        image (torch.Tensor): Image to print
        fname (str): Path where to save the image.
            Defaults to None i.e. the image is not saved.
    """

    pl.imshow(np.asarray(image.cpu().permute(1, 2, 0)[:, :, :4]))
    pl.show()

    if fname is not None:
        pl.savefig(fname=fname)


def make_seed(n_images: int, n_channels: int, image_size: int,
              device: torch.device = "cpu") -> torch.Tensor:
    """Makes n_images seeds to start the CA, the seed is a black dot

    Args:
        n_images (int): Number of seed images to generate
        n_channels (int): Number of channels per image
        image_size (int): Side of the square image
        device (torch.device, optional): Device where to save the images.
            Defaults to "cpu".

    Returns:
        torch.Tensor: Seed images
    """
    start_point = torch.zeros(
        (n_images, n_channels, image_size, image_size), device=device)
    start_point[:, 3, image_size//2, image_size//2] = 1.
    return start_point


def moving_average(v: np.ndarray, window_size: int) -> np.ndarray:
    """Computes moving average of a vector "v"

    Args:
        v (np.ndarray): Vector to compute the moving average
        window_size (int): Size of the window to compute the average

    Returns:
        np.ndarray: The averaged version of "v"
    """
    return np.convolve(v, np.ones(window_size), 'valid') / window_size


def side(size, constant_side=False):
    """Return size of the side to be used to erase square portion of the images"""
    if constant_side:
        return size//2
    return randint(size//6, size//2)


def make_squares(images, target_size=None, side=side, constant_side=False):
    """Sets square portion of input images to zero"""
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x = randint(target_size//2-target_size//4,
                    target_size//2+target_size//4)
        y = randint(target_size//2-target_size//4,
                    target_size//2+target_size//4)
        images[i, :, x-side(target_size, constant_side)//2:x+side(target_size, constant_side) //
               2, y-side(target_size, constant_side)//2:y+side(target_size, constant_side)//2] = 0.

    return images


def make_poligon(images, target_size=None, side=side):
    """Sets random poligonal portion of input images to zero"""
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x1 = randint(target_size//2-target_size//4,
                     target_size//2+target_size//4)
        x2 = randint(target_size//2-target_size//4,
                     target_size//2+target_size//4)
        y1 = randint(target_size//2-target_size//4,
                     target_size//2+target_size//4)
        y2 = randint(target_size//2-target_size//4,
                     target_size//2+target_size//4)
        images[i, :, x1-side(target_size)//2:x2+side(target_size) //
               2, y1-side(target_size)//2:y2+side(target_size)//2] = 0.

    return images

# Custom loss function


def image_distance(x: torch.Tensor, y: torch.Tensor, order=2) -> torch.Tensor:
    """Returns the distance between two batches of images

    Args:
        x (torch.Tensor): First batch of images
        y (torch.Tensor): Second batch of images
        order (int, optional): Order of the norm. Defaults to 2.

    Returns:
        torch.Tensor: Distance between each of the images in the batches
    """

    return torch.mean(torch.abs(x - y)**order, dim=[1, 2, 3])


class loss_fn:
    """Custom loss function for the neural CA, simply computes the
        l1 or l2 norm of the target image vs the predicted image
    """

    def __init__(self, target: torch.Tensor, criterion=torch.nn.MSELoss):
        """Initializes the loss function by storing the target image

        Args:
            target (torch.Tensor): Target image
            criterion (Loss function, optional): 
                Loss criteria, used to compute the distance between two images.
                Defaults to torch.nn.MSELoss.
        """
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss

        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """

        losses = self.criterion(x[:, :4], self.target).mean(dim=[1,2,3])
        idx_max_loss = torch.argmax(losses)

        return torch.mean(losses), idx_max_loss
