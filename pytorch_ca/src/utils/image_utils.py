from scipy.stats import truncexpon
import torch

import torchvision.transforms as T
from torchvision.utils import save_image
from einops import rearrange

import numpy as np
import pylab as pl

from matplotlib import pyplot as plt


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


def RGBAtoRGB(images: torch.Tensor, alpha_channel: int = 3) -> torch.Tensor:
    """Converts a 0-1 RGBA image into RGB

    Args:
        images (torch.Tensor): RGBA images in 0-1 range

    Returns:
        torch.Tensor: RGB images in 0-255 range
    """

    if len(images.size()) < 4:
        images = torch.unsqueeze(images, 0)
    if type(alpha_channel) == int:
        alpha_channel = [alpha_channel]

    multip = torch.zeros_like(images[:, 0, :, :])

    for ac in alpha_channel:
        multip = multip+images[:, ac, :, :]
    return torch.clip(images[:, :3, :, :] * multip * 255 + (1-multip)*255, 0, 255).type(torch.uint8)


def GrayscaletoCmap(image: torch.Tensor, cmap="viridis") -> torch.Tensor:
    """Converts a 0-1 2D tensor representing an image into a colormap

    Args:
        image (torch.Tensor): 2D tensor with no channel dimension
        cmap (str, optional): color map to use, must be present in the
            matplotlib package. Defaults to "viridis".

    Returns:
        torch.Tensor: RGB image in 0-1 range
    """

    assert image.dim() == 2, "image must be 2D"
    assert cmap in plt.colormaps(), "cmap must be a valid matplotlib colormap"

    # here i clip the values to 1
    with torch.no_grad():
        image = (image <= 1)*image + (image > 1)*torch.ones_like(image)

    image = image.cpu()

    cmap = plt.get_cmap(cmap)
    image = image.numpy()
    image = cmap(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    return RGBAtoRGB(image)


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

    img = pl.imshow(np.asarray(image.cpu().permute(1, 2, 0)[:, :, :4]))
    if fname is not None:
        pl.axis('off')
        pl.savefig(fname=fname, bbox_inches="tight")

    pl.show()

    return img


def make_collage(images: torch.Tensor, width: int, fname: str = None, rescaling: int = 8) -> torch.Tensor:
    """Makes a collage out of a batch of images

    Args:
        images (torch.Tensor): Batch of images with the torch standard
        width (int): width of the collage, the batch size must be a multiple of it
        fname (str, optional): Where to save the collage. Defaults to not saving it.
        rescaling (int, optional): Rescaling factor. Defaults to 8.

    Returns:
        torch.Tensor: Collage of images
    """
    if images.size()[0] % width != 0:
        Exception("The batch is not a multiple of width")

    rescaler = T.Resize(images.size()[-1] * rescaling,
                        T.InterpolationMode.NEAREST)
    image = rearrange(rescaler(images),
                      '(b1 b2) c h w -> 1 c (b1 h) (b2 w)', b2=width)
    if fname is not None:
        save_image(RGBAtoRGB(image[0][:4])/255., fname)

    return image


class tensor_to_RGB():
    """ Converts a tensor to RGB, it will be used as an argument for the function make_video() """

    def __init__(self, rescaling: int = 8, function='RGBA', CA: "CAModel" = None):
        """
        Args:
            CA: the CA rule
            rescaling (int, optional): Rescaling factor,
                since the CA is a small image we need to rescale it
                otherwise it will be blurry. Defaults to 8.
            function (callable, optional):
                -If it is a callable, it represent the function that converts the torch tensor of shape
                    (1,n_channels,image_size,image_size) to the image of size
                    (3,image_size*rescaling,image_size*rescaling).
                -If it is a int, it will return a heatmap of the channel represented by the int
                -If it is the string "RGBA" it will return the RGBA image following the info from the CA
        """
        self.rescaling = rescaling
        self.function = function

        if function == 'RGBA':
            if CA == None:
                raise Exception(
                    'If the function is "RGBA" you must specify the CA rule')
            self.CA = CA
            self.function = self.RGBA

        if type(function) == int:
            self.channel = function
            self.function = self.gray

    def __call__(self, tensor: torch.Tensor):
        """Converts a tensor to RGB

        Args:
            tensor (torch.Tensor): tensor that represents the CA state

        Returns:
            torch.Tensor: with shape (3,image_size*rescaling,image_size*rescaling)
        """
        video_size = tensor.size()[-1] * self.rescaling
        rescaler = T.Resize((video_size, video_size),
                            interpolation=T.InterpolationMode.NEAREST)
        tensor = rescaler(tensor)
        return self.function(tensor)

    def RGBA(self, tensor):
        return RGBAtoRGB(tensor, self.CA.alpha_channel)[0].cpu()

    def gray(self, tensor):
        return GrayscaletoCmap(tensor[0, self.channel])
