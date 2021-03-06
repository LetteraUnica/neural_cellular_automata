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
#                                                               orange                   blue
def two_channels(image: torch.Tensor, colors=torch.tensor([[254/255,97/255,0.],[120/255,94/255,240/255]])) -> torch.Tensor:
    """Converts two-channel 2D tensor (2,W,H) representing an image into a colormap
        each of the two channels is a different color

    Args:
        image (torch.Tensor): (2,W,H) tensor in 0-1 range
        colors (str, optional): color map to use, must be present in the
            matplotlib package. Defaults to ['red','yellow'].

    Returns:
        torch.Tensor: RGB image in 0-1 range
    """
    
    assert image.dim() == 3, "image must have 3 dimentions"
    assert image.size(0) == 2, "image must have 2 channels"
    assert colors.shape==(2,3), "colors must be a 2x3 tensor"

    #adds an extra one at the tail, it will be useful later
    # [[1,0.25,0.1],[1,1,0.1]] -> [[1,0.25,0.1,1],[1,1,0.1,1]]
    colors=torch.cat((colors,torch.ones([2,1])), 1) 

    # here i clip the values to 1
    with torch.no_grad():
        image = (image <= 1)*image + (image > 1)*torch.ones_like(image)

    new_image=torch.tensordot(colors,image,([0],[0]))
    new_image[:-1]=new_image[:-1]/new_image[-1]
    
    return RGBAtoRGB(new_image)


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


def erode(images: torch.Tensor, erosion_depth: int) -> torch.Tensor:
    """Computes an erosion morphological operation on binary images
    Works also on non-binary images however the result is undefined
    """
    kernel_size = 2*erosion_depth + 1
    return -torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=erosion_depth)(-images)


def state_to_image(x, mask_channels):
    """Sums all the values of all mask channels into one, must be called before
    calling imshow"""
    alpha = x[:, mask_channels].sum(dim=1).unsqueeze(1)
    return torch.cat((x[:, :3], alpha), dim=1)


def make_collage(images: torch.Tensor):
    return rearrange(images, 'b c h w -> c h (b w)')


def imshow(image: torch.Tensor, fname: str = None):
    """Shows an image

    Args:
        image (torch.Tensor): Image to print
        fname (str): Path where to save the image.
            Defaults to None i.e. the image is not saved.
    """
    if len(image.size()) == 4:
        image = make_collage(image[:, :4])
    pl.imshow(image[:4].detach().cpu().permute(1,2,0))
    pl.axis('off')

    if fname is not None:
        pl.savefig(fname=fname, bbox_inches="tight")

    pl.show()


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
                -if it is a list with two elements, it will return the image of the two channels
                    in the list
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

        if type(function) == list and len(function) == 2 and type(function[0])==int:
            self.channel = function
            self.function = self.two

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

    def two(self, tensor):
        t=torch.empty([2,*tensor[0].size()[1:]])
        t[0]=tensor[0,self.channel[0]]
        t[1]=tensor[0,self.channel[1]]        
        return two_channels(t) 


def repeat_and_resize(image, n_repeats, rescale=8, image_size=48):
    resizer = T.Resize((image_size*rescale, image_size*rescale), interpolation=T.InterpolationMode.NEAREST)
    a = image.clone()
    a = resizer(GrayscaletoCmap(a[0,0]))
    a = a.repeat(n_repeats, 1, 1, 1)

    return a