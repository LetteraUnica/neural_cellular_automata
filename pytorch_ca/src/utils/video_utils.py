from scipy.stats import truncexpon
import torch

from torchvision.io import write_video

from .utils import *
from .image_utils import *


def make_video(CA: "CAModel",
               n_iters: int,
               init_state: torch.Tensor = None,
               regenerating: bool = False,
               fname: str = None,
               fps: int = 10,
               initial_video: torch.Tensor = None,
               converter: callable = None,
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
    # create the initial state in case there is none
    if init_state is None:
        n_channels = CA.n_channels
        init_state = make_seed(1, n_channels-1, 48, alpha_channel=3)
    init_state = init_state.to(CA.device)

    # create the converter if there is none
    if converter==None:
        converter=tensor_to_RGB(CA=CA)
    if type(converter) is not list:
        converter = [converter]
    l = len(converter)
    for c in converter:
        if c.rescaling != converter[0].rescaling:
            raise Exception(
                "the rescaling must be the same for all converters!")

    # set video visualization features
    video_size = init_state.size()[-1] * converter[0].rescaling
    video = [torch.empty((n_iters, 3, video_size, video_size),
                         device="cpu") for _ in range(l)]

    # this manages the kwargs necessary for the regenerating case
    if regenerating:
        target_size = None
        constant_side = None
        if 'target_size' in kwargs:
            target_size = kwargs['target_size']
        if 'constant_side' in kwargs:
            constant_side = kwargs['constant_side']

    # evolution
    with torch.no_grad():
        for i in range(n_iters):
            for k in range(l):
                video[k][i] = converter[k](init_state)
            init_state = CA.forward(init_state)

            if regenerating and i == n_iters//3:
                init_state = make_squares(
                    init_state, target_size, constant_side=constant_side)

    # this concatenates the new video with the old one
    if initial_video is not None:
        if type(initial_video) is not list:
            initial_video = [initial_video]
        if len(initial_video) != l:
            raise Exception(
                "The lenght of the initial_video must be the same of the converter")
        for i in range(l):
            video[i] = torch.cat((initial_video[i], video[i]))

    # this saves the video
    if fname is not None:
        if type(fname) is not list:
            fname = [fname]
        if len(fname) != l:
            raise Exception(
                "The lenght of f_name must be the same of the converter")
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


# TODO test this
def switch_video(old_CA: "CAModel",
                 new_CA: "CAModel",
                 switch_iters: int = 50,
                 n_iters: int = 200,
                 init_state: torch.Tensor = None,
                 regenerating: bool = False,
                 fname: str = None,
                 fps: int = 10,
                 converter: callable = None,
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
        converter (callable, optional):
            function that converts the torch.Tensor of the state to an image.
            Defaults to RGBAtoRGB
    """

    initial_video, initial_state = make_video(
        old_CA, switch_iters, init_state, fps=fps, converter=converter, **kwargs)
    return make_video(new_CA, n_iters, init_state=initial_state,
                      initial_video=initial_video,
                      fname=fname, fps=fps,
                      regenerating=regenerating,
                      converter=converter,
                      **kwargs)
