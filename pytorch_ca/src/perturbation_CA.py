import torch
from torchvision.io import write_video
import torchvision.transforms as T

from .utils import *
from .CAModel import *


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
        super().__init__()

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
        pre_life_mask = get_living_mask(x, 3)

        dx_new = self.new_CA.compute_dx(x, angle, step_size)
        dx_old = self.old_CA.compute_dx(x, angle, step_size)
        x += dx_new + dx_old

        post_life_mask = get_living_mask(x, 3)
        life_mask = pre_life_mask & post_life_mask

        self.new_cells = dx_new * life_mask.float()

        return x * life_mask.float()

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

        self.train()

        for i in range(n_epochs):
            epoch_losses = []  # array that stores the loss history

            # take the data
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)  # sample the inputs
                # put them in the current device
                inputs = inputs.to(self.device)
                optimizer.zero_grad()  # reinitialize the gradient to zero

                # recursive forward-pass
                for k in range(randint(*evolution_iters)):
                    inputs = self.forward(inputs)
                    criterion.add_perturbation(self.new_cells)

                # calculate the loss of the inputs and return the ones with the biggest loss
                loss, idx_max_loss = criterion(inputs, n_max_losses)
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
                    #if we have reset_prob in the kwargs then sometimes the pool resets
                    if 'reset_prob' in kwargs:
                        if np.random.uniform()<kwargs['reset_prob']:
                            pool.reset()
                          

            # update the scheduler if there is one at all
            if scheduler is not None:
                scheduler.step()
            
            # Log epoch losses
            epoch_loss = np.mean(epoch_losses)

            # Stopping criteria
            if np.isnan(epoch_loss) or (epoch_loss > 5 and i > 2): break
            if epoch_loss > 0.25 and i == 40: break

            wandb.log({"loss": epoch_loss})
            self.losses.append(epoch_loss)
            print(f"epoch: {i+1}\navg loss: {epoch_loss}")
            clear_output(wait=True)
