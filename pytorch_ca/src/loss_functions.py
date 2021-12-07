from typing import Callable, Iterable
import torch
from abc import abstractmethod

from src.utils.math.integrators import CachedDiscreteIntegrator
from .utils import *


def n_largest_indexes(array: list, n: int = 1) -> list:
    """returns the indexes of the n largest elements of the array

    url:https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    """
    if n == 0:
        return None
    return sorted(range(len(array)), key=lambda x: array[x])[-n:]


class BaseNCALoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_weight(a: int, b: int) -> float:
        return b - a

    def get_weight_vectorized(self, a: Iterable[int], b: Iterable[int]) -> torch.Tensor:
        return torch.tensor([self.get_weight(ai, bi) for (ai, bi) in zip(a, b)])

    @abstractmethod
    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int]) -> torch.Tensor:
        pass


class NCALoss(BaseNCALoss):
    """Custom loss function for the neural CA, computes the
        distance of the target image vs the predicted image and adds a
        penalization term
    """

    def __init__(self, target: torch.Tensor, criterion=torch.nn.MSELoss,
                 alpha_channels=None):
        """Initializes the loss function by storing the target image and setting
            the criterion

        Args:
            target (torch.Tensor): Target image
            criterion (Loss function, optional): 
                Loss criteria, used to compute the distance between two images.
                Defaults to torch.nn.MSELoss.
        """
        super().__init__()
        
        if alpha_channels is None:
            alpha_channels = [3]
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")
        self.alpha_channels = alpha_channels
    
    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int] = None) -> torch.Tensor:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss

        Returns:
            torch.Tensor: 
                Loss for each of the images in the batch
        """

        alpha = torch.sum(x[:, self.alpha_channels], dim=1).unsqueeze(1)
        predicted = torch.cat((x[:, :3], alpha), dim=1)

        losses = self.criterion(predicted, self.target).mean(dim=[1, 2, 3])

        return losses


class CellRatioLoss(BaseNCALoss):
    """Custom loss function for the multiple CA, computes the
        distance of the target image vs the predicted image, adds a
        penalization term and penalizes the number of original cells
    """

    def __init__(self, alpha_channels=None):
        """Args: The same as the NCALoss and alpha_channels (optional, float): multiplicative constant to regulate
        the importance of the original cell ratio
        """
        super().__init__()

        if alpha_channels is None:
            alpha_channels = [3]
        self.alpha_channels = alpha_channels

    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int] = None) -> torch.Tensor:
        original_cells = x[:, self.alpha_channels[0]].sum(dim=[1, 2])
        virus_cells = x[:, self.alpha_channels[1]].sum(dim=[1, 2])
        original_cell_ratio = original_cells / (original_cells + virus_cells + 1e-8)

        return original_cell_ratio


class NCADistance(BaseNCALoss):
    def __init__(self, model_1: nn.Module, model_2: nn.Module, penalization: float = 0.):
        """Extension of the NCALoss that penalizes the distance between two
        models using the parameter "penalization"
        """
        super().__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.penalization = penalization

    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int] = None) -> torch.Tensor:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss

        Returns:
            Tuple(torch.Tensor, torch.Tensor):
                Average loss of all images in the batch,
                index of the image with maximum loss
        """

        return self.penalization * ruler.distance(self.model_1, self.model_2)


class WeightedLoss(BaseNCALoss):
    """Weights the loss with a function of the number of steps and the epoch"""

    def __init__(self, loss_function: nn.Module, weight_function: Callable[[int], float]) -> None:
        super().__init__()

        self.loss_function = loss_function
        self.weight_function = weight_function
        self.integrator = CachedDiscreteIntegrator(weight_function)

    def get_weight(self, a: int, b: int) -> float:
        return self.integrator.integrate(a, b)
    
    def weight_function_vectorized(self, x):
        try:
            return torch.tensor([self.weight_function(xi) for xi in x])
        except TypeError:
            return self.weight_function(x)  

    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int]) -> torch.Tensor:
        return self.loss_function(x) * self.weight_function_vectorized(evolution_steps).to(x.device)


class CombinedLoss(BaseNCALoss):
    def __init__(self, loss_functions: Iterable[BaseNCALoss], weights: Iterable[float]):
        super().__init__()

        assert len(loss_functions) == len(weights)
        assert True not in [weight < 0 for weight in weights]
        
        self.loss_functions = loss_functions
        self.weights = [weight / sum(weights) for weight in weights]


    def get_weight(self, a: int, b: int) -> float:
        total_weight = 0
        for loss, weight in zip(self.loss_functions, self.weights):
            total_weight += weight * loss.get_weight(a, b)

        return total_weight

    def __call__(self, x: torch.Tensor, evolution_steps: Iterable[int]) -> torch.Tensor:
        losses = self.weights[0] * self.loss_functions[0](x, evolution_steps)

        for i in range(1, len(self.loss_functions)):
            losses += self.weights[i] * self.loss_functions[i](x, evolution_steps)

        return losses
