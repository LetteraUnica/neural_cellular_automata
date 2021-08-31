import torch
from torchvision.io import write_video
import torchvision.transforms as T
from typing import List
import warnings

from .utils import *
from .neural_CA import *


class CustomCA(NeuralCA):
    def __init__(self, mask_channel: int = 16,
                 n_channels: int = 16,
                 device: torch.device = None,
                 fire_rate: float = 0.5):
        """Initializes the network.

        Args:
            mask_channel (int, optional): Channel to use as a mask.
                Defaults to 16.
            n_channels (int, optional): Number of input channels.
                Defaults to 16.
            device (torch.device, optional): Device where to store the net.
                Defaults to None.
            fire_rate (float, optional): Probability to reject an update.
                Defaults to 0.5.
        """
        if mask_channel < n_channels-1:
            Exception("Mask channel must be greater or equal to n_channels-1")

        super().__init__(n_channels, device, fire_rate)

        self.mask_channel = mask_channel-1

    def compute_dx(self, x: torch.Tensor, angle: float = 0.,
                   step_size: float = 1.) -> torch.Tensor:
        """Computes a single update dx

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0..
            step_size (float, optional): Step size of the update. Defaults to 1..

        Returns:
            torch.Tensor: dx
        """

        x_new = torch.cat((x[:, :3],
                           x[:, self.mask_channel:self.mask_channel+1],
                           x[:, 3:self.n_channels-1]), dim=1)

        # compute update increment
        dx = self.layers(self.perceive(x_new, angle)) * step_size

        # get random-per-cell mask for stochastic update
        update_mask = torch.rand(x[:, :1, :, :].size(),
                                 device=self.device) < self.fire_rate

        dx_new = torch.zeros_like(x)
        dx_new[:, :self.n_channels-1] = dx[:, :self.n_channels-1]
        dx_new[:, self.mask_channel] = dx[:, -1]

        return dx_new*update_mask.float()

    def get_living_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the a mask of the living cells in the image

        Args:
            images (torch.Tensor): images to get the living mask

        Returns:
            torch.Tensor: Living mask
        """
        alpha = images[:, self.mask_channel:self.mask_channel+1, :, :]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0.
            step_size (float, optional): Step size of the update. Defaults to 1.

        Returns:
            torch.Tensor: Next CA state
        """
        pre_life_mask = self.get_living_mask(x)

        x = x + self.compute_dx(x, angle, step_size)

        post_life_mask = self.get_living_mask(x)

        # get alive mask
        life_mask = pre_life_mask & post_life_mask

        # return updated states with alive masking
        return x * life_mask.float()


class MultipleCA(CAModel):
    """Given a list of CA rules, evolves the image pixels using multiple CA rules
    """

    def __init__(self, CAs: List[CustomCA]):
        """Initializes the model

        Args:
            CAs (List[CAModel]): List of CAs to use in the evolution
        """

        masks = set()
        for CA in CAs:
            if CA.mask_channel in masks:
                warnings.warn(f"The channel {CA.mask_channel+1} \
                                is already used by another CA")
            masks.add(CA.mask_channel)

        self.mask_channels = list(masks)

        self.CAs = CAs
        self.losses = []

    def evolve_masks(self, x):
        z = x[:, self.mask_channels]
        B, C, H, W = z.size()
        # Creates a mask of the maximum value over all the channels of
        # all the images in the batch
        mask = torch.max(z, dim=1)[0].view(B, 1, H, W) == z

        x[:, self.mask_channels] = z

        return x

    def forward(self, x: torch.Tensor,
                angle: float = 0.,
                step_size: float = 1.) -> torch.Tensor:
        """Single update step of the CA

        Args:
            x (torch.Tensor): Previous CA state
            angle (float, optional): Angle of the update. Defaults to 0.
            step_size (float, optional): Step size of the update. Defaults to 1.

        Returns:
            torch.Tensor: Next CA state
        """

        N = len(self.CAs)
        B, C, H, W = x.size()
        updates = torch.empty((N, B, C, H, W))

        global_pre_life_mask = self.get_life_mask(x)

        # Ideas: Remove local life masks and only keep a global one?
        # Apply updates all at once or one at a time randomly or sequentially?  
        for i, CA in enumerate(self.CAs):
            pre_life_mask = self.get_living_mask(x)

            updates[i] = CA.compute_dx(x, angle, step_size)

            post_life_mask = self.get_living_mask(x+updates[i])
            life_mask = pre_life_mask & post_life_mask

            updates[i] *= life_mask
        
        x += updates.sum(dim=0)

        x = self.evolve_masks(x)

        global_post_life_mask = self.get_life_mask(x)

        global_life_mask = global_pre_life_mask & global_post_life_mask

        return x * global_life_mask.float()

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


class CustomLoss:
    """Custom loss function for the neural CA, simply computes the
        distance of the target image vs the predicted image
    """

    def __init__(self, target: torch.Tensor, criterion=torch.nn.MSELoss,
                 alpha_channel=16):
        """Initializes the loss function by storing the target image

        Args:
            target (torch.Tensor): Target image
            criterion (Loss function, optional): 
                Loss criteria, used to compute the distance between two images.
                Defaults to torch.nn.MSELoss.
        """
        self.target = target.detach().clone()
        self.criterion = criterion(reduction="none")
        self.alpha_channel = alpha_channel - 1

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the loss and the index of the image with maximum loss

        Args:
            x (torch.Tensor): Images to compute the loss

        Returns:
            Tuple(torch.Tensor, torch.Tensor): 
                Average loss of all images in the batch, 
                index of the image with maximum loss
        """

        predicted = torch.cat((x[:, :3],
                              x[:, self.alpha_channel:self.alpha_channel+1]),
                              dim=1)
        losses = self.criterion(predicted, self.target).mean(dim=[1, 2, 3])
        idx_max_loss = torch.argmax(losses)

        return torch.mean(losses), idx_max_loss
