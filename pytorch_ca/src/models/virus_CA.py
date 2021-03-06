import torch

from ..utils import *
from .CAModel import *
from .neural_CA import *


class VirusCA(CAModel):
    """Given two CA rules and a mask, evolves the image pixels using the old_CA
    rule if the mask is == 0 or using the new_CA rule if the mask is == 1.
    """

    def __init__(self, old_CA: NeuralCA, new_CA: NeuralCA, mutation_probability:float = 0.5):
        """Initializes the model

        Args:
            old_CA (CAModel): old_CA model
            new_CA (CAModel): new_CA model
            mutation_probability (float, optional): Probability of the cell to
                become a virus. Defaults to 0.5

        """
        super().__init__()

        if old_CA.device != new_CA.device:
            Exception(f"The two CAs are on different devices: " +
                      f"decaying_CA.device: {old_CA.device} and " +
                      f"regenerating_CA.device: {new_CA.device}")

        self.device = old_CA.device

        self.old_CA = old_CA
        self.new_CA = new_CA
        self.mutation_probability = mutation_probability
        self.initialized=False

    def update_cell_mask(self, x: torch.Tensor):
        """Updates the cell mask randomly with mutation probability equal to
        self.mutation_probability

        Args:
            x (torch.Tensor): Input images, only used to take the shape
        """
        self.new_cells = random_mask(x.size()[0], x.size()[-1], self.mutation_probability, self.device)
        self.old_cells = 1. - self.new_cells

        self.initialized = True


    def update(self, x):
        """Useful fo the train loop"""
        self.update_cell_mask(x)

    def set_cell_mask(self, mutation_mask: torch.Tensor):
        """Updates the cell mask to the given mask

        Args:
            mutation_mask (torch.Tensor): New mask
        """
        self.new_cells = mutation_mask.to(self.device).float()
        self.old_cells = 1. - self.new_cells

        self.initialized = True

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0..
            step_size (float, optional): Step size of the update. Defaults to 1..

        Returns:
            torch.Tensor: Next CA state
        """
        if self.initialized==False:
            self.update_cell_mask(x)

        x_old = self.old_CA(x, angle, step_size)
        x_new = self.new_CA(x, angle, step_size)
        return x_old * self.old_cells + x_new * self.new_cells