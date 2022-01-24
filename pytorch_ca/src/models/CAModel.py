from typing import Any
import torch
import wandb
from IPython.display import clear_output

from ..loss_functions import *
from ..utils.train import *
from ..sample_pool import *


class CAModel(nn.Module):
    """Base CA class, each CA class inherits from this class
    """

    def __init__(self, n_channels=16, device=None, fire_rate=0.5):
        super(CAModel, self).__init__()

        # useless comment
        self.n_channels = n_channels
        self.alpha_channel = 3

        # defines the device
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Stores losses during training
        self.losses = []

        self.fire_rate = fire_rate

        self.to(self.device)

    @abstractmethod
    def forward(self, x, angle=None, step_size=None) -> torch.Tensor:
        pass

    def evolve(self, x: torch.Tensor, iters: int, angle: float = 0.,
               step_size: float = 1.) -> torch.Tensor:
        """Evolves the input images "x" for "iters" steps

        Args:
            x (torch.Tensor): Previous CA state
            iters (int): Number of steps to perform
            angle (float, optional): Angle of the update. Defaults to 0..
            step_size (float, optional): Step size of the update. Defaults to 1..

        Returns:
            torch.Tensor: dx
        """
        self.eval()
        with torch.no_grad():
            for i in range(iters):
                x = self.forward(x, angle=angle, step_size=step_size)

        return x

    def test_CA(self,
                criterion: Callable[[torch.Tensor], torch.Tensor],
                pool: torch.Tensor,
                evolution_iters: int = 1000) -> torch.Tensor:
        """Evaluates the model over the given images by evolving them
            and computing the loss against the target at each iteration.
            Returns the mean loss at each iteration

        Args:
            criterion (Callable[[torch.Tensor], torch.Tensor]): Loss function
            pool (SamplePool): Sample pool from which to extract the images
            evolution_iters (int, optional): Evolution steps. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            torch.Tensor: tensor of size (evolution_iters) 
                which contains the mean loss at each iteration
        """

        self.eval()

        with torch.no_grad():
            inputs=pool[:]
            inputs = inputs.to(self.device)
            evolutions_per_image = np.zeros(inputs.shape[0])
            loss_per_step = self.loss_eval(inputs, criterion, evolution_iters, evolutions_per_image,epoch=0,log_losses=True)

            #here i remove the outliers
            not_outliers=loss_per_step.mean(dim=[0,1])<loss_per_step.mean(dim=[0,1,2])*5
            loss_per_step = loss_per_step[:,:,not_outliers]

        return loss_per_step.mean(dim=-1).cpu().numpy()

    def train_CA(self,
                 optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, Any], torch.Tensor],
                 pool: SamplePool,
                 n_epochs: int,
                 scheduler: torch.optim.lr_scheduler = None,
                 batch_size: int = 4,
                 skip_update: int = 2,
                 evolution_iters: int = 96,
                 kind: str = "growing",
                 n_max_losses: int = 1,
                 stopping_criterion: StoppingCriteria = DefaultStopping(),
                 **kwargs):
        """Trains the CA model

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to use, recommended Adam

            criterion (Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]]): Loss function to use

            pool (SamplePool): Sample pool from which to extract the images

            n_epochs (int): Number of epochs to perform, _
                this depends on the size of the sample pool

            scheduler (torch.optim.lr_scheduler, optional):
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

            n_max_losses(int,optional):
                number of datapoints with the biggest losses to replace.
                Defaults to 1

            normalize_gradients (bool, optional): Whether to normalize the gradient to norm 1 before applying it.
                Defaults to False.

            stopping_criterion (StoppingCriteria, optional): Stopping criterion to use,
                must inherit from StoppingCriteria and implement the method stop that throws a RuntimeException
        """

        self.train()

        for epoch in range(n_epochs):
            epoch_losses = []  # array that stores the loss history
    
            #in some epochs we do a checkpoint where some operations are performed
            self.checkpoint(epoch)
    
            # take the data
            for j in range(pool.size // batch_size):        
                inputs, indexes = pool.sample(batch_size)  # sample the inputs
                # put them in the current device
                inputs = inputs.to(self.device)
                optimizer.zero_grad()  # reinitialize the gradient to zero

                self.update(inputs)  # This is useful when you update the fixed mask

                # recursive forward-pass
                evolutions_per_image = pool.get_evolutions_per_image(indexes)
                inputs, total_losses = self.loss_eval(inputs, criterion, evolution_iters, evolutions_per_image, epoch)

                # We remove the worst performers, often times they degenerate and ruins everything
                total_loss = torch.mean(total_losses[total_losses<5*total_losses.mean()])
                total_loss.backward()
                optimizer.step()

                # customization of training for the three processes of growing. persisting and regenerating
                # if regenerating, then damage inputs
                if kind == "regenerating" and j % kwargs["skip_damage"] == 0:
                    inputs = inputs.detach()
                    # damages the inputs by removing square portions
                    inputs = make_squares(inputs)

                # look at the definition of skip_update
                if j % skip_update != 0:
                    idx_max_loss = None

                # if training is not for growing process then re-insert trained/damaged samples into the pool
                if kind != "growing":
                    idx_max_loss = n_largest_indexes(total_losses, n_max_losses)
                    pool.update(indexes, inputs, idx_max_loss, evolution_iters)

                epoch_losses.append(total_loss.detach().cpu().item())

            # if we have reset_prob in the kwargs then sometimes the pool resets
            if kind!='growing' and 'reset_prob' in kwargs:
                if np.random.uniform() < kwargs['reset_prob']:
                    pool.reset()

            # update the scheduler if there is one at all
            if scheduler is not None:
                scheduler.step()

            # Log epoch losses
            epoch_loss = np.mean(epoch_losses)

            # Stopping criteria
            stopping_criterion.stop(epoch, epoch_loss)

            wandb.log({"loss": epoch_loss})
            self.losses.append(epoch_loss)
            print(f"epoch: {epoch + 1}\navg loss: {epoch_loss}")
            clear_output(wait=True)


    def loss_eval(self, inputs, criterion, evolution_iters, evolutions_per_image,epoch=0,log_losses=False):
        total_losses = torch.zeros(inputs.size()[0], device=self.device)
        loss_per_step = []

        for n_step in range(evolution_iters):
            inputs = self.forward(inputs)
            # calculate the loss of the inputs and return the ones with the biggest loss
            params = {"start_iteration": evolutions_per_image,
                        "current_iteration": evolutions_per_image + n_step,
                        "end_iteration": evolutions_per_image + evolution_iters - 1,
                        "n_epoch": epoch,
                        "log_losses": log_losses}
            losses = criterion(inputs, **params)
            if log_losses==True:
                loss_per_step.append(losses)
            else:
                total_losses += losses
        
        if log_losses==True:
            return torch.stack(loss_per_step)
        return inputs, total_losses + self.end_step_loss(inputs, **params)

    #These functions are to be defined in the child classes    
    def update(self, x):
        return

    def checkpoint(self, epoch):
        return 

    def end_step_loss(self, x, **params):
        return 0