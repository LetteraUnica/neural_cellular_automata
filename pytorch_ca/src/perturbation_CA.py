import torch
from torchvision.io import write_video
import torchvision.transforms as T

from .utils import *
from .neural_CA import *


class PerturbationCA(CAModel):
    """Given two CA rules evolves the image pixels using the formula:
    x_t+1 = base_CA(x_t) + new_CA(x_t). To avoid that new_CA overwrites the
    base_CA a L2 loss on the update of new_CA should be used.
    """

    def __init__(self, base_CA: CAModel, new_CA: CAModel):
        """Initializes the model

        Args:
            old_CA (CAModel): base_CA model
            new_CA (CAModel): new_CA model, implements the perturbation
        """
        if base_CA.device != new_CA.device:
            Exception(f"The two CAs are on different devices: " +
                      f"decaying_CA.device: {base_CA.device} and " +
                      f"regenerating_CA.device: {new_CA.device}")

        self.device = base_CA.device
        self.old_CA = base_CA
        self.new_CA = new_CA

        self.new_cells = None
        self.losses = []

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Forward pass, computes the new state:
        x_t+1 = base_CA(x_t) + new_CA(x_t)

        Args:
            x (torch.Tensor): Current state
            angle (float, optional): Angle of the update. Defaults to 0.
            step_size (float, optional): Step size of the update. Defaults to 1.

        Returns:
            torch.Tensor: Next state
        """
        pre_life_mask = self.new_CA.get_living_mask(x)

        dx_new = self.new_CA.compute_dx(x, angle, step_size)
        dx_old = self.old_CA.compute_dx(x, angle, step_size)
        x += dx_new + dx_old

        post_life_mask = self.new_CA.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        self.new_cells = dx_new * life_mask.float()

        return x * life_mask.float()

    def train_CA(self,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor], torch.Tensor],
                 pool: SamplePool,
                 n_epochs: int,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 batch_size: int = 4,
                 skip_update: int = 2,
                 evolution_iters: Tuple[int, int] = (50, 60),
                 kind: str = "growing",
                 **kwargs):
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

            kind (str, optional): 
                Kind of CA to train, can be either one of:
                    growing: Trains a CA that grows into the target image
                    persistent: Trains a CA that grows into the target image
                        and persists
                    regenerating: Trains a CA that grows into the target image
                                  and regenerates any damage that it receives
                Defaults to "growing".
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
                    criterion.add_perturbation(self.new_cells)

                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                if j % skip_update != 0:
                    idx_max_loss = None

                loss.backward()
                optimizer.step()

                # if regenerating, then damage inputs
                if kind == "regenerating":
                    inputs = inputs.detach()
                    try:
                        target_size = kwargs['target_size']
                    except KeyError:
                        target_size = None
                    try:
                        constant_side = kwargs['constant_side']
                    except KeyError:
                        constant_side = None
                    try:
                        skip_damage = kwargs["skip_damage"]
                    except KeyError:
                        skip_damage = 1

                    if j % skip_damage == 0:
                        inputs = make_squares(inputs, target_size=target_size,
                                              constant_side=constant_side)

                if kind != "growing":
                    pool.update(inputs, indexes, idx_max_loss)

            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)
