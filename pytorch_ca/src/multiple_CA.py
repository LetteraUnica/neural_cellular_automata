import torch
from torchvision.io import write_video
import torchvision.transforms as T
from typing import List
import warnings
from einops.layers.torch import Reduce
from einops import rearrange


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

        super().__init__(n_channels+1, device, fire_rate)

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
        x_new = torch.cat((x[:, :self.n_channels-1],
                          x[:, self.alpha_channel:self.alpha_channel+1]), dim=1)

        # compute update increment
        dx = self.layers(self.perceive(x_new, angle)) * step_size

        dx_new = torch.zeros_like(x)
        dx_new[:, :self.n_channels-1] = dx[:, :self.n_channels-1]
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

        self.mask_channels = list(range(n_channels, n_channels+n_CAs))

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

        update_mask = multiple_living_mask(x[:, self.n_channels:])
        pre_life_mask = update_mask.max(dim=1)[0].unsqueeze(1)

        tensor = x[:, self.n_channels:]
        biggest = tensor.max(dim=1)[0].unsqueeze(1)
        old = (tensor == biggest) | ((tensor >= 0.1).sum(dim=1) == 0).unsqueeze(1)
        x[:, self.n_channels:] = x[:, self.n_channels:] * old.float()

        B, C, H, W = x.size()
        updates = torch.empty(self.n_CAs, B, C, H, W, device=self.device)
        for i, CA in enumerate(self.CAs):
            updates[i] = CA.compute_dx(
                x, angle, step_size) * update_mask[:, i].float().unsqueeze(1)

        # updates = rearrange(updates, 'CA B C W H -> C B CA W H')
        # updates[:] = updates[:]*mask
        # updates = rearrange(updates, 'C B CA W H -> CA B C W H')

        x += updates.sum(dim=0)

        post_life_mask = get_living_mask(x, self.mask_channels)

        life_mask = pre_life_mask & post_life_mask

        x = x * life_mask.float()

        return x
