import torch
from typing import Tuple

from .utils import *


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


class NCALoss:
    """Custom loss function for the neural CA, simply computes the
        distance of the target image vs the predicted image
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

        losses = self.criterion(x[:, :4], self.target).mean(dim=[1, 2, 3])
        idx_max_loss = torch.argmax(losses)

        return torch.mean(losses), idx_max_loss


class PerturbationLoss:
    """Custom loss function for the neural CA, computes the
        distance of the target image vs the predicted image and adds a
        penalization term
    """

    def __init__(self, target: torch.Tensor, criterion=torch.nn.MSELoss,
                 l: float = 0.):
        """Initializes the loss function by storing the target image and setting
            the criterion

        Args:
            target (torch.Tensor): Target image
            criterion (Loss function, optional): 
                Loss criteria, used to compute the distance between two images.
                Defaults to torch.nn.MSELoss.
            l (float): Regularization factor, useful to penalize the perturbation
        """
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")
        self.l = l

        self._reset_perturbation()

    def _reset_perturbation(self):
        self.perturbation = 0.
        self.N = 0

    def __call__(self, x: torch.Tensor,n:int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss
            n (int): The number of indexes with the max loss to return

        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """
        losses = self.criterion(x[:, :4], self.target).mean(dim=[1, 2, 3])
        idx_max_loss = n_largest_indexes(losses,n)
        loss = torch.mean(losses) + self.l*self.perturbation/self.N

        self._reset_perturbation()

        return loss, idx_max_loss

    def add_perturbation(self, perturbation: torch.Tensor):
        """Adds the perturbation to the loss, in order to penalize it,
            The perturbation can be the output of the neural CA

        Args:
            perturbation (torch.Tensor): Perturbation to add to the loss.
        """

        self.perturbation += torch.mean(perturbation**2)
        self.N += 1
