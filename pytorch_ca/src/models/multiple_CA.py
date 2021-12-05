import torch
from torchvision.io import write_video
import torchvision.transforms as T
from typing import List
import warnings
from einops.layers.torch import Reduce
from einops import rearrange


from ..utils import *
from .neural_CA import *
from .CAModel import *

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
            raise Exception("alpha_channel must be greater or equal to n_channels")

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

        # reshape x in such a way that is good for the NeuralCA class
        x_new = multiple_to_single(x,self.n_channels-1)

        # compute update increment
        dx = super().compute_dx(x_new,angle,step_size)

        # reshape dx in shuch a wat that is good for the MultipleCA class
        dx = single_to_multiple(dx, x.shape, self.n_channels-1, self.alpha_channel)
        
        return dx


class MultipleCA(CAModel):
    """Given a list of CA rules, evolves the image pixels using multiple CA rules
    """

    def __init__(self, n_channels=15, n_CAs=2, device=None, fire_rate=0.5, senescence=None):
        """Initializes the model

        Args:

        """
        super().__init__(n_channels,device,fire_rate)

        # cellular automatae rules
        self.n_CAs = n_CAs
        self.CAs = [CustomCA(n_channels, n_channels+i, device, fire_rate)
                    for i in range(n_CAs)]

        self.alpha_channel = [*range(n_channels,n_channels+n_CAs)]
        self.senescence = senescence

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
        # Apply updates all at once or one at a time randomly/sequentially?
        # Currently applies only a global mask and all updates at once

        #calculate the mask of each channel
        update_mask = multiple_living_mask(x[:, self.n_channels:]).float()
        #calculate the global mask
        pre_life_mask = update_mask.max(dim=1)[0].unsqueeze(1)
            
        #apply the mask to the input tensor
        x[:, self.n_channels:] = x[:, self.n_channels:] * update_mask

        #set to zero every cell that is dead
        x = x * pre_life_mask

        #senescence mechanism
        if self.senescence!=None:
            x = x * (torch.randlike(pre_life_mask)>self.senescence)

        #create the updates tensor
        updates = torch.empty([self.n_CAs, *x.shape], device=self.device)
        for i, CA in enumerate(self.CAs):
            updates[i] = CA.compute_dx(x, angle, step_size)

        update_mask=update_mask/(1e-8 + update_mask.sum(dim=1,keepdim=True))
        update_mask[torch.isnan(update_mask)]=0
        
        #The sum of all updates is the total update
        updates = torch.einsum("Abchw, bAhw -> bchw", updates, update_mask)

        x = x + updates

        return x
