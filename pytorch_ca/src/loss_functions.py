import torch
from torch import nn
from typing import Tuple, List

from .utils import *


def n_largest_indexes(array: list, n: int = 1) -> list:
    """returns the indexes of the n largest elements of the array

    url:https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    """
    if n==0: return None
    return sorted(range(len(array)), key=lambda x: array[x])[-n:]


class NCALoss:
    """Custom loss function for the neural CA, computes the
        distance of the target image vs the predicted image and adds a
        penalization term
    """

    def __init__(self, target: torch.Tensor, criterion=torch.nn.MSELoss,
                 l: float = 0., alpha_channels: Tuple[int] = [3], n_max_losses: int = 1):
        """Initializes the loss function by storing the target image and setting
            the criterion

        Args:
            target (torch.Tensor): Target image
            criterion (Loss function, optional): 
                Loss criteria, used to compute the distance between two images.
                Defaults to torch.nn.MSELoss.
            l (float): Regularization factor, useful to penalize the perturbation
            n_max_losses (int): The number of indexes with the max loss to return

        """
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")
        self.l = l
        self.alpha_channels = alpha_channels
        self.n_max_losses=n_max_losses

        self._reset_perturbation()

    def _reset_perturbation(self):
        self.perturbation = 0.
        self.N = 0

    def __call__(self, x: torch.Tensor, n_max_losses: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss
            n_max_losses (int): The number of indexes with the max loss to return

        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """
        if n_max_losses==None: n_max_losses=self.n_max_losses

        alpha = torch.sum(x[:, self.alpha_channels], dim=1).unsqueeze(1)
        predicted = torch.cat((x[:, :3], alpha), dim=1)

        losses = self.criterion(predicted, self.target).mean(dim=[1, 2, 3])
        idx_max_loss = n_largest_indexes(losses, n_max_losses)
        loss = torch.mean(losses)

        if self.N != 0:
            loss += self.l*self.perturbation/self.N

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


class MultipleCALoss(NCALoss):
    """Custom loss function for the multiple CA, computes the
        distance of the target image vs the predicted image, adds a
        penalization term and penalizes the number of original cells
    """

    def __call__(self, x, n_max_losses=None):
        original_cells = x[:, self.alpha_channels[0]].sum(dim=[1, 2])
        virus_cells = x[:, self.alpha_channels[1]].sum(dim=[1, 2])
        original_cell_ratio = original_cells / (original_cells+virus_cells)

        loss, idx_max_loss = super().__call__(x, n_max_losses)

        loss = loss + original_cell_ratio.sum()

        return loss, idx_max_loss


class NCADistance(NCALoss):
    def model_distance(self, model1: nn.Module, model2: nn.Module):
        """Computes the distance between the parameters of two models"""
        p1, p2 = ruler.parameters_to_vector(model1), ruler.parameters_to_vector(model2)
        return nn.MSELoss()(p1, p2)

    def __init__(self, model1: nn.Module, model2: nn.Module, target: torch.Tensor,
                 criterion=torch.nn.MSELoss, l: float = 0., alpha_channels: Tuple[int] = [3],
                 n_max_losses : int = 1):
        """Extension of the NCALoss that penalizes the distance between two
        models using the parameter l

        """
        self.model1 = model1
        self.model2 = model2
        super().__init__(target, criterion, l, alpha_channels, n_max_losses)

    def __call__(self, x: torch.Tensor, n_max_losses: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss
            n (int): The number of indexes with the max loss to return

        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """

        loss, idx_max_loss = super().__call__(x, n_max_losses)

        loss += self.l * self.model_distance(self.model1, self.model2)
        return loss, idx_max_loss
