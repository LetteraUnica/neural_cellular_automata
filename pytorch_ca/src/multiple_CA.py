import torch
from torchvision.io import write_video
import torchvision.transforms as T
from typing import List
import warnings
from einops.layers.torch import Reduce, Rearrange


from .utils import *
from .neural_CA import *


class CustomCA(NeuralCA):
    def __init__(self, n_channels: int = 15,
                 alpha_channel: int = 15,
                 device: torch.device = None,
                 fire_rate: float = 0.5):
        """Initializes the network.

        Args:
            n_channels (int, optional): Number of input channels.
                Defaults to 16.
            alpha_channel (int, optional): Channel to use as alpha.
                Defaults to 16.
            device (torch.device, optional): Device where to store the net.
                Defaults to None.
            fire_rate (float, optional): Probability to reject an update.
                Defaults to 0.5.
        """
        if alpha_channel < n_channels:
            raise Exception(
                "alpha_channel must be greater or equal to n_channels")

        super().__init__(n_channels, device, fire_rate)

        self.alpha_channel = alpha_channel

    def compute_dx(self, x: torch.Tensor, angle: float = 0.,
                   step_size: float = 1.) -> torch.Tensor:
        """Computes a single update dx

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0..
            step_size (float, optional): Step size of the update. Defaults to 1..

        Returns:
            torch.Tensor: dx
        """
        x_new = torch.cat((x[:, :self.n_channels],
                          x[:, self.alpha_channel:self.alpha_channel+1]), dim=1)

        print(x_new.shape, x[:, :self.n_channels].shape, x[:, self.alpha_channel:self.alpha_channel+1].shape)

        # compute update increment
        dx = self.layers(self.perceive(x_new, angle)) * step_size

        dx_new = torch.zeros_like(x)
        dx_new[:, :self.n_channels] = dx[:, :self.n_channels]
        dx_new[:, self.alpha_channel] = dx[:, -1]

        return dx_new

    def get_living_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the a mask of the living cells in the image

        Args:
            images (torch.Tensor): images to get the living mask

        Returns:
            torch.Tensor: Living mask
        """
        alpha = images[:, self.alpha_channel:self.alpha_channel+1, :, :]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0.
            step_size (float, optional): Step size of the update. Defaults to 1.

        Returns:
            torch.Tensor: Next CA state
        """
        pre_life_mask = self.get_living_mask(x)

        x = x + self.compute_dx(x, angle, step_size)

        post_life_mask = self.get_living_mask(x)

        # get alive mask
        life_mask = pre_life_mask & post_life_mask

        # return updated states with alive masking
        return x * life_mask.float()


class MultipleCA(CAModel, TrainCA):
    """Given a list of CA rules, evolves the image pixels using multiple CA rules
    """

    def __init__(self, n_channels=15, n_CAs=2, device=None, fire_rate=0.5):
        """Initializes the model

        Args:

        """
        super().__init__()

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.n_channels = n_channels

        self.fire_rate = fire_rate

        # Stores losses during training
        self.losses = []

        # cellular automatae rules
        self.n_CAs = n_CAs
        self.CAs = [CustomCA(n_channels, n_channels+i, device, fire_rate)
                    for i in range(n_CAs)]

    def CA_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """It gives the mask where the CA rules apply in the case where multiple alphas
        are included in the CA

        Args:
            tensor (torch.Tensor):
                The first index refers to the batch, the second to the alphas,
                the third and the fourth to the pixels in the image

        Returns:
            a tensor with bool elements with the same shape on the input tensor
            that represents where each CA rule applies
        """

        # gives the biggest alpha per pixel
        biggest = Reduce('b c h w -> b 1 h w', reduction='max')(tensor)
        # the free cells are the ones who have all of the alphas lower than 0.1
        free = biggest < 0.1

        # this is the mask where already one of the alpha is bigger than 0.1, if more than one
        # alpha is bigger than 0.1, than the biggest one wins
        old = (tensor[:] == biggest) * (tensor >= 0.1)
        # delete the smallest alphas if the biggest alpha is bigger than 0.1
        tensor = tensor * old
        # this is the mask of the cells neighboring each alpha
        neighbor = F.max_pool2d(wrap_edges(tensor), 3, stride=1) >= 0.1
        # the cells where the CA can expand are the one who are free and neighboring
        expanding = free & neighbor
        # the CA evolves int the cells where it can expand and the ones where is already present
        evolution = expanding + old
        # stocastic filter
        stocastic = torch.rand(
            evolution.size(), device=self.device) < self.fire_rate

        evolution = evolution*stocastic

        return evolution, tensor

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0.
            step_size (float, optional): Step size of the update. Defaults to 1.

        Returns:
            torch.Tensor: Next CA state
        """
        # Ideas: Remove local life masks and only keep a global one?
        # Apply updates all at once or one at a time randomly/sequentially?
        # Currently applies only a global mask and all updates at once

        B, C, H, W = x.size()
        updates = torch.empty(self.n_CAs, B, C, H, W, device=self.device)

        mask, x[:, self.n_channels:] = self.CA_mask(x[:, self.n_channels:])
        for i, CA in enumerate(self.CAs):
            updates[i] = CA.compute_dx(x, angle, step_size)

        updates = Rearrange(updates, 'CA B C W H -> C B CA W H')
        updates[:] = updates[:]*mask
        updates = Rearrange(updates, 'C B CA W H -> CA B C W H')

        x += updates.sum(dim=0)

        return x
