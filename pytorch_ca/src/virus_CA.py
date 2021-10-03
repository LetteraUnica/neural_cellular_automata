import torch
from torchvision.io import write_video
import torchvision.transforms as T


from .utils import *
from .CAModel import *
from .neural_CA import *


def compute_random_mask(size: torch.Size, device: torch.device, probability: float = 0.6):
    return torch.rand(size, device=device) < probability


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

    def update_cell_mask(self, x: torch.Tensor):
        """Updates the cell mask randomly with mutation probability equal to
        self.mutation_probability

        Args:
            x (torch.Tensor): Input images, only used to take the shape
        """
        self.new_cells = (torch.rand_like(x[:, 0:1, :, :]) < self.mutation_probability).float()
        self.old_cells = 1. - self.new_cells

    def set_cell_mask(self, mutation_mask: torch.Tensor):
        """Updates the cell mask to the given mask

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
                 criterion: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 pool: SamplePool,
                 n_epochs: int,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 batch_size: int = 4,
                 skip_update: int = 2,
                 evolution_iters: Tuple[int, int] = (50, 60),
                 kind: str = "growing",
                 n_max_losses: int = 1,
                 **kwargs):
        """Trains the CA model

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to use, recommended Adam

            criterion (Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]]): Loss function to use

            pool (SamplePool): Sample pool from which to extract the images

            n_epochs (int): Number of epochs to perform, _
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
            n_max_losses(int):
                number of datapoints with the biggest losses to replace.
                Defaults to 1
        """
        self.new_CA.train()

        for i in range(n_epochs):
            epoch_losses = []  # array that stores the loss history

            # take the data
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)  # sample the inputs
                # put them in the current device
                inputs = inputs.to(self.device)
                self.update_cell_mask(inputs)
                optimizer.zero_grad()  # reinitialize the gradient to zero

                # recursive forward-pass
                for k in range(randint(*evolution_iters)):
                    inputs = self.forward(inputs)

                # calculate the loss of the inputs and return the ones with the biggest loss
                loss, idx_max_loss = criterion(inputs, n_max_losses)
                wandb.log({"loss": loss})
                # add current loss to the loss history
                epoch_losses.append(loss.item())

                # look a definition of skip_update
                if j % skip_update != 0:
                    idx_max_loss = None

                # backward-pass
                loss.backward()
                optimizer.step()

                # customization of training for the three processes of growing. persisting and regenerating

                # if regenerating, then damage inputs
                if kind == "regenerating" and j % kwargs["skip_damage"] == 0:
                    inputs = inputs.detach()
                    # damages the inputs by removing square portions
                    inputs = make_squares(inputs)

                # if training is not for growing proccess then re-insert trained/damaged samples into the pool
                if kind != "growing":
                    pool.update(indexes, inputs, idx_max_loss)

            # Wandb logging
            # wandb.log({"mutation_probability": self.mutation_probability})

            # fname = f"Pretrained_models/virus adiabatico {self.mutation_probability}%.pt"
            # self.new_CA.save(fname, overwrite=True)
            # wandb.save(fname)

            # self.mutation_probability -= 0.002

            # update the scheduler if there is one at all
            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)