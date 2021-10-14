from scipy.stats import truncexpon
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.io import write_video
from torchvision.utils import save_image
from einops.layers.torch import Reduce
from einops import rearrange

import numpy as np
import pylab as pl
from random import randint

from matplotlib import pyplot as plt

from typing import Tuple, List


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
    if type(alpha_channel)==int:
        alpha_channel=[alpha_channel]

    multip=torch.zeros_like(images[:,0,:,:])

    for ac in alpha_channel:
        multip=multip+images[:,ac,:,:]
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

    #here i clip the values to 1
    with torch.no_grad(): image = (image<=1)*image + (image>1)*torch.ones_like(image) 

    image=image.cpu()

    cmap = plt.get_cmap(cmap)
    image = image.numpy()
    image = cmap(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    return RGBAtoRGB(image)

class tensor_to_RGB():
    """ Converts a tensor to RGB, it will be used as an argument for the function make_video() """

    def __init__(self, rescaling : int =8,function='RGBA',CA : "CAModel"=None):
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
        self.rescaling=rescaling
        self.function=function
        
        if function == 'RGBA':
            if CA==None:
                raise Exception('If the function is "RGBA" you must specify the CA rule')
            self.CA=CA
            self.function=self.RGBA
        
        if type(function)==int:
            self.channel=function
            self.function=self.gray

    def __call__(self,tensor: torch.Tensor):
        """Converts a tensor to RGB

        Args:
            tensor (torch.Tensor): tensor that represents the CA state

        Returns:
            torch.Tensor: with shape (3,image_size*rescaling,image_size*rescaling)
        """
        video_size = tensor.size()[-1] * self.rescaling
        rescaler=T.Resize((video_size, video_size),interpolation=T.InterpolationMode.NEAREST)
        tensor=rescaler(tensor)
        return self.function(tensor)

    def RGBA(self,tensor):
        return RGBAtoRGB(tensor,self.CA.alpha_channel)[0].cpu()

    def gray(self,tensor):
        return GrayscaletoCmap(tensor[0,self.channel])



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


def make_video(CA: "CAModel",
               n_iters: int,
               init_state: torch.Tensor = None,
               regenerating: bool = False,
               fname: str = None,
               fps: int = 10,
               initial_video: torch.Tensor = None,
               converter: callable = RGBAtoRGB,
               **kwargs) -> torch.Tensor:
    """Returns the video (torch.Tensor of size (n_iters, init_state.size()))
        of the evolution of the CA starting from a given initial state

    Args:
        CA (CAModel): Cellular automata to evolve and make the video
        n_iters (int): Number of iterations to evolve the CA
        init_state (torch.Tensor, optional): Initial state to evolve.
            Defaults to None, which means a seed state.
        regenerating (bool, optional): Whether to erase a square portion
            of the image during the video, useful if you want to show
            the regenerating capabilities of the CA. Defaults to False.
        fname (str, optional): File where to save the video.
            Defaults to None.
        fps (int, optional): Fps of the video. Defaults to 10.
        initial_video (torch.Tensor, optional): Video that gets played before
            the new one
        converter (callable, optional):
            function that converts the torch.Tensor of the state to an image.
            Defaults to RGBAtoRGB
    """

    if init_state is None:
        n_channels = CA.n_channels
        init_state = make_seed(1, n_channels-1, 48, alpha_channel=alpha_channel)

    init_state = init_state.to(CA.device)

    if type(converter) is not list:
        converter=[converter]
    l=len(converter)
    for c in converter:
        if c.rescaling!=converter[0].rescaling:
            raise Exception("the rescaling must be the same for all converters!")

    # set video visualization features
    video_size = init_state.size()[-1] * converter[0].rescaling    
    video = [torch.empty((n_iters, 3, video_size, video_size), device="cpu") for _ in range(l)]

    
    if regenerating:
        target_size=None
        constant_side=None
        if 'target_size' in kwargs:
            target_size=kwargs['target_size']
        if 'constant_side' in kwargs:
            constant_side = kwargs['constant_side']

    # evolution
    with torch.no_grad():
        for i in range(n_iters):
            for k in range(l):
                video[k][i]=converter[k](init_state)
            init_state = CA.forward(init_state)

            if regenerating and i == n_iters//3:
                init_state = make_squares(init_state,target_size,constant_side=constant_side)

    # this concatenates the new video with the old one
    if initial_video is not None:
        if type(initial_video) is not list:
            initial_video=[initial_video]
        if len(initial_video)!=l:
            raise Exception("The lenght of the initial_video must be the same of the converter")
        for i in range(l):    
            video[i] = torch.cat((inititial_video[i], video[i]))

    if fname is not None:
        if type(fname) is not list:
            fname=[fname]
        if len(fname)!=l:
            raise Exception("The lenght of f_name must be the same of the converter")
        for i in range(l):
            write_video(fname[i], video[i].permute(0, 2, 3, 1), fps=fps)

    return video, init_state


def merge_videos(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Merges two videos together, first is played before second.

    Args:
        first (torch.Tensor): First video
        second (torch.Tensor): Second video

    Returns:
        torch.Tensor: Merged video
    """
    return torch.cat((first, second))


#TODO test this
def switch_video(old_CA: "CAModel",
                 new_CA: "CAModel",
                 switch_iters: int = 50,
                 n_iters: int = 200,
                 init_state: torch.Tensor = None,
                 regenerating: bool = False,
                 fname: str = None,
                 fps: int = 10,
                 converter: callable = RGBAtoRGB,
                 **kwargs) -> torch.Tensor:
    """Returns the video (torch.Tensor of size (n_iters, init_state.size()))
        of the evolution of two CAs starting from a given initial state,
        old_CA is used to evolve the initial image for switch_iters steps
        while new_CA evolves the resulting image for the next n_iters steps.

    Args:
        old_CA (CAModel): First cellular automata to evolve
        new_CA (CAModel): Second cell automata to evolve
        init_state
        switch_iters (int): Number of iterations to evolve the first CA
        n_iters (int): Number of iterations to evolve the second CA
        init_state (torch.Tensor, optional): Initial state to evolve.
            Defaults to None, which means a seed state.
        regenerating (bool, optional): Whether to erase a square portion
            of the image during the video, useful if you want to show
            the regenerating capabilities of the CA. Defaults to False.
        fname (str, optional): File where to save the video.
            Defaults to None.
        rescaling (int, optional): Rescaling factor,
            since the CA is a small image we need to rescale it
            otherwise it will be blurry. Defaults to 8.
        fps (int, optional): Fps of the video. Defaults to 10.
    """

    initial_video, initial_state = make_video(old_CA, switch_iters, init_state, fps=fps,converter=converter,**kwargs)
    return make_video(new_CA, n_iters, init_state=initial_state,
                      initial_video=initial_video,
                      fname=fname, fps=fps,
                      regenerating=regenerating,
                      converter=converter,
                      **kwargs)


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


def wrap_edges(images: torch.Tensor) -> torch.Tensor:
    """Pads the boundary of all images to simulate a torus

    Args:
        images (torch.Tensor): Images to pad

    Returns:
        torch.Tensor: Padded images
    """
    return F.pad(images, pad=(1, 1, 1, 1), mode='circular', value=0)


def get_living_mask(images: torch.Tensor, channels: List[int]) -> torch.Tensor:
    """Returns the a mask of the living cells in the image

    Args:
        images (torch.Tensor): images to get the living mask
        channels (List[int]): channels where to compute the living mask

    Returns:
        torch.Tensor: Living mask
    """
    if isinstance(channels, int):
        channels = [channels]
    alpha = images[:, channels, :, :]

    neighbors = F.max_pool2d(wrap_edges(alpha), 3, stride=1) > 0.1
    return torch.max(neighbors, dim=1)[0].unsqueeze(1)


def multiple_living_mask(alphas: torch.Tensor):
    """It gives the mask where the CA rules apply in the case where multiple alphas
    are included in the CA

    Args:
        alphas (torch.Tensor):
            The first index refers to the batch, the second to the alphas,
            the third and the fourth to the pixels in the image

    Returns:
        (torch.Tensor) A tensor with bool elements with the same shape on the input tensor
        that represents where each CA rule applies
    """

    # gives the biggest alpha per pixel
    biggest = Reduce('b c w h-> b 1 w h', reduction='max')(alphas)
    
    # the free cells are the ones who have all of the alphas lower than 0.1
    free = biggest < 0.1

    # this is the mask where already one of the alpha is bigger than 0.1, if more than one
    # alpha is bigger than 0.1, than the biggest one wins
    old = (alphas == biggest) & (alphas >= 0.1)
    
    # this is the mask of the cells neighboring each alpha
    neighbor = F.max_pool2d(wrap_edges(alphas), 3, stride=1) >= 0.1
    
    # the cells where the CA can expand are the one who are free and neighboring
    expanding = free & neighbor
    
    # the CA evolves in the cells where it can expand and the ones where is already present
    mask = expanding | old

    return mask #the mask has the same shape of alphas and the values inside ar bool


def n_largest_indexes(array: list, n: int = 1) -> list:
    """returns the indexes of the n largest elements of the array

    url:https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    """
    return sorted(range(len(array)), key=lambda x: array[x])[-n:]


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
    images[:, original_channel] = images[:,original_channel] * (~virus_mask).float()

    return images


class VirusGenerator:
    def __init__(self, n_channels, image_size, n_CAs, CA, virus_rate=0.1, iter_func=ExponentialSampler()):
        self.n_channels = n_channels
        self.image_size = image_size
        self.n_CAs = n_CAs
        self.CA = CA
        self.virus_rate = virus_rate
        self.iter_func = iter_func

        self.model_device = self.CA.device

    def __call__(self, n_images, device):
        start_point = make_seed(n_images, self.n_channels, self.image_size,
                                self.n_CAs, 3, self.model_device)

        batch_size = 32
        i = 0
        while i < n_images:
            start_point[i:i+batch_size] = self.CA.evolve(
                start_point[i:i+batch_size], self.iter_func()[0])
            i += batch_size

        # start_point = add_virus(start_point, -2, -1, self.virus_rate)
        return start_point.to(device)


def multiple_to_single(images: torch.Tensor, n_channels: int, alpha_channel: int) -> torch.Tensor:
    """
    maronn maronn
    """
    return torch.cat((images[:, :3],
                      images[:, alpha_channel:alpha_channel+1],
                      images[:, 3:n_channels]), dim=1)


def single_to_multiple(dx: torch.Tensor, shape, n_channels: int, alpha_channel: int):
    """
    è un miracolo!
    """
    dx_new = torch.zeros(shape)
    dx_new[:, :3] = dx[:, :3]
    dx_new[:, 3:n_channels] = dx[:, 4:]
    dx_new[:, alpha_channel] = dx[:, 3]
    return dx_new
