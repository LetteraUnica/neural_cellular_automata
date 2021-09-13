import torch
from torchvision.io import write_video
import torchvision.transforms as T


from .utils import *
from .CAModel import *


def compute_random_mask(size: torch.Size, device: torch.device, probability: float = 0.6):
    return torch.rand(size, device=device) < probability


class VirusCA(CAModel):
    """Given two CA rules and a mask, evolves the image pixels using the old_CA
    rule if the mask is == 0 or using the new_CA rule if the mask is == 1.
    """

    def __init__(self, old_CA: CAModel, new_CA: CAModel, mask: torch.Tensor):
        """Initializes the model

        Args:
            old_CA (CAModel): old_CA model
            new_CA (CAModel): new_CA model
            mask (torch.Tensor): Mask, a tensor with 0's and 1's
        """
        super().__init__()

        if old_CA.device != new_CA.device:
            Exception(f"The two CAs are on different devices: " +
                      f"decaying_CA.device: {old_CA.device} and " +
                      f"regenerating_CA.device: {new_CA.device}")

        self.device = old_CA.device

        self.old_CA = old_CA
        self.new_CA = new_CA
        self.new_cells = mask
        self.losses = []

    def update_cell_masks(self, mutation_mask: torch.Tensor):
        """Updates the cell mask

        Args:
            mutation_mask (torch.Tensor): New mask
        """
        self.new_cells = mutation_mask.to(self.device).float()
        self.old_cells = 1. - self.new_cells

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
        x_old = self.old_CA(x, angle, step_size)
        x_new = self.new_CA(x, angle, step_size)
        return x_old * self.old_cells + x_new * self.new_cells

    def train_CA(self,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor], torch.Tensor],
                 pool: SamplePool,
                 n_epochs: int,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 batch_size: int = 4,
                 skip_update: int = 2,
                 evolution_iters: Tuple[int, int] = (50, 60)):
        """Trains the CA model

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to use, recommended Adam

            criterion (Callable[[torch.Tensor], torch.Tensor]): Loss function to use

            pool (SamplePool): Sample pool from which to extract the images

            n_epochs (int): Number of epochs to perform, 
                this depends on the size of the sample pool

            scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
                 Learning rate scheduler. Defaults to None.

            batch_size (int, optional): Batch size. Defaults to 4.

            skip_update (int, optional): How many batches to process before
                the image with maximum loss is replaced with a new seed.
                Defaults to 2, i.e. substitute the image with maximum loss
                every 2 iterations.

            evolution_iters (Tuple[int, int], optional):
                Minimum and maximum number of evolution iterations to perform.
                Defaults to (50, 60).
        """

        self.old_CA.train()
        self.new_CA.train()

        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                optimizer.zero_grad()

                for k in range(randint(*evolution_iters)):
                    inputs = self.forward(inputs)

                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                if j % skip_update != 0:
                    idx_max_loss = None
                loss.backward()
                optimizer.step()
                pool.update(inputs, indexes, idx_max_loss)

            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)
