import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import os.path
from random import randint
from IPython.display import clear_output
from typing import Callable, Tuple
from abc import abstractmethod
import numpy as np

from .utils import *
from .sample_pool import *
from .CAModel import *



class NeuralCA(CAModel):
    """Implements a neural cellular automata model like described here
    https://distill.pub/2020/growing-ca/
    """

    def __init__(self, n_channels: int = 16,
                 device: torch.device = None,  # ma non è inutile questo argomento?
                 fire_rate: float = 0.5):
        """Initializes the network.

        Args:
            n_channels (int, optional): Number of input channels.
                Defaults to 16.
            device (torch.device, optional): Device where to store the net.
                Defaults to None.
            fire_rate (float, optional): Probability to reject an update.
                Defaults to 0.5.
        """

        super().__init__(n_channels,device,fire_rate)

        # Network layers needed for the update rule
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels*3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, n_channels, 1))

        self.to(device)

        self.alpha_cannel=3

    def perceive(self, images: torch.Tensor, angle: float = 0.) -> torch.Tensor:
        """Returns the perception vector of each cell in an image, or perception matrix

        Args:
            images (torch.Tensor): Images to compute the perception matrix
            angle (float, optional): Angle of the Sobel filters. Defaults to 0.

        Returns:
            torch.Tensor: Perception matrix
        """

        # Filters
        identity = torch.tensor([[0., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]])
        dx = torch.tensor([[-0.125, 0., 0.125],
                           [-0.25, 0., 0.25],
                           [-0.125, 0., 0.125]])
        dy = dx.T

        # Rotation
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        dx, dy = c*dx - s*dy, s*dx + c*dy

        # Create filters batch
        all_filters = torch.stack((identity, dx, dy))
        all_filters_batch = all_filters.repeat(
            self.n_channels, 1, 1).unsqueeze(1)
        all_filters_batch = all_filters_batch.to(self.device)

        # Depthwise convolution over input images
        return F.conv2d(wrap_edges(images), all_filters_batch, groups=self.n_channels)

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
        # compute update increment
        dx = self.layers(self.perceive(x, angle)) * step_size

        # get random-per-cell mask for stochastic update
        update_mask = torch.rand(
            x[:, :1, :, :].size(), device=self.device) < self.fire_rate

        return dx*update_mask.float()

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
        pre_life_mask = get_living_mask(x,3)

        x = x + self.compute_dx(x, angle, step_size)

        post_life_mask = get_living_mask(x,3)

        # get alive mask
        life_mask = pre_life_mask & post_life_mask

        # return updated states with alive masking
        return x * life_mask.float()

    def load(self, fname: str):
        """Loads a (pre-trained) model

        Args:
            fname (str): Path of the model to load
        """

        self.load_state_dict(torch.load(fname))
        print("Successfully loaded model!")

    def save(self, fname: str, overwrite: bool = False):
        """Saves a (trained) model

        Args:
            fname (str): Path where to save the model.
            overwrite (bool, optional): Whether to overwrite the existing file.
                Defaults to False..

        Raises:
            Exception: If the file already exists and
                the overwrite argument is set to False
        """
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        torch.save(self.state_dict(), fname)
        print("Successfully saved model!")
