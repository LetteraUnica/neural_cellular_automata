from typing import Any, Callable, Sequence, Tuple
import torch
from torch import nn

from .utils import *


class NCALoss:
    """Custom loss function for the neural CA, computes the
        distance of the target image vs the predicted image
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
        if alpha_channels is None:
            alpha_channels = [3]
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")
        self.alpha_channels = alpha_channels

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss
        Args:
            x (torch.Tensor): Images to compute the loss
        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """

        alpha = torch.sum(x[:, self.alpha_channels], dim=1, keepdim=True)
        predicted = torch.cat((x[:, :3], alpha), dim=1)

        losses = self.criterion(predicted, self.target).mean(dim=[1, 2, 3])

        return losses


class CellRatioLoss:
    """Penalizes the number of original cells
    """

    def __init__(self, alpha_channels=None, penalization: float = 1.):
        """Args:
            The same as the NCALoss and 
            alpha (optional, float): multiplicative constant to regulate the importance of the original cell ratio
        """

        if alpha_channels is None:
            alpha_channels = [3]
        self.alpha_channels = alpha_channels
        self.penalization = penalization

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor]:
        original_cells = x[:, self.alpha_channels[0]].sum(dim=[1, 2])
        virus_cells = x[:, self.alpha_channels[1]].sum(dim=[1, 2])
        original_cell_ratio = original_cells / (original_cells + virus_cells + 1e-8)

        return self.penalization * original_cell_ratio


class NCADistance:
    def __init__(self, model1: nn.Module, model2: nn.Module, penalization: float = 1.):
        """Penalizes the distance between two models using the parameter penalization
        """
        self.model1 = model1
        self.model2 = model2
        self.penalization = penalization

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Returns the loss and the index of the image with maximum loss
        Args:
            x (torch.Tensor): Images to compute the loss
        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """

        return self.penalization * ruler.distance(self.model1, self.model2)


class cLoss:
    """
    Combines several losses into one loss function that depends on the number of steps
    Most of the work is done by the combination_function, so you need to choose
    the combination_function carefully or use a combination_function generator
    """
    def __init__(self, 
                losses_functions:Sequence[Callable[[torch.Tensor], torch.Tensor]],
                combination_function: Callable[[int, *Any], torch.Tensor]):
        """Args:
            Losses (List[nn.Module]): List of losses to combine
            combination_function (Callable): Function to combine the losses, it takes as input the
                number of steps and the epoch, and it outputs a vector of floats as long as the number of losses
        """
        self.loss_functions = loss_functions
        self.combination_function=combination_function
        if type (combination_function)==list:
            assert len(loss_functions) == len(weight_functions), 'Number of loss functions and weight functions must be the same'
            self.combination_function=combination_function_generator(combination_function)
        
    def __call__(self, x, n_steps=0, n_epoch=0, initial_step=0, *args, **kwargs) -> torch.Tensor:
        losses = torch.stack([loss(x, *args, **kwargs) for loss in self.loss_functions])
        weights=self.combination_function(n_steps,*args, **kwargs).to(x.device)

        if losses.shape==weights.shape:
            return torch.sum(weights*losses,axis=0,requires_grad=False)
        return torch.matmul(weights,losses,requires_grad=False)


class combination_function_generator:
    def __init__(self,weight_functions: Sequence[Callable[[Any], float]]):
        #Each one of them is a function f:R->R
        self.weight_functions = [np.vectorize(weight_function) for weight_function in weight_functions]
        #This are the indefinite integral of the functions above
        self.integrals = [CachedSummer(weight_function) for weight_function in weight_functions]

    def get_normalization(self, end_iteration, start_iteration=0, **kwargs) -> torch.Tensor:
        """Returns the normalization constant for the loss function"""
        #norm of each one of the functions
        constants = [integral.sum_between(start_iteration, end_iteration) for integral in self.integrals]
        return torch.from_numpy(np.array(constants)).sum(dim=0)

    def __call__(self, n_steps, *args, **kwargs) -> torch.Tensor:
        weights = np.array([weight(*args, **kwargs) for weight in self.weight_functions])
        weights = torch.from_numpy(weights).to(x.device)
        #calculates the weights for each loss
        weights = np.array([weight(*args, **kwargs) for weight in self.weight_functions])
        weights = torch.from_numpy(weights).to(x.device)
        #normalizes the losses
        normalization = self.get_normalization(**kwargs).to(x.device)

        return weights / (normalization+1e-8)

class CombinedLoss:
    def __init__(self,
                 loss_functions: Sequence[Callable[[torch.Tensor], torch.Tensor]],
                 weight_functions: Sequence[Callable[[Any], float]]):
        super().__init__()

        assert len(loss_functions) == len(weight_functions), 'Number of loss functions and weight functions must be the same'

        self.loss_functions = loss_functions
        #Each one of them is a function f:R->R
        self.weight_functions = [np.vectorize(weight_function) for weight_function in weight_functions]
        #This are the indefinite integral of the functions above
        self.integrals = [CachedSummer(weight_function) for weight_function in weight_functions]

    def get_normalization(self, end_iteration, start_iteration=0, **kwargs) -> torch.Tensor:
        """Returns the normalization constant for the loss function"""
        #norm of each one of the functions
        constants = [integral.sum_between(start_iteration, end_iteration) for integral in self.integrals]
        return torch.from_numpy(np.array(constants)).sum(dim=0)

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        #calculates the losses for each input image x
        losses = torch.stack([loss(x, *args, **kwargs) for loss in self.loss_functions])
        #calculates the weights for each loss
        weights = np.array([weight(*args, **kwargs) for weight in self.weight_functions])
        weights = torch.from_numpy(weights).to(x.device)
        #normalizes the losses
        normalization = self.get_normalization(**kwargs).to(x.device)

        #this is the value that is returned
        out = torch.sum(losses * weights, dim=0) / (normalization + 1e-8)
        out.requires_grad = False
        return out
