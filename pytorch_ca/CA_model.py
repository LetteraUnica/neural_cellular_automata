import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload
from random import randint
from IPython.display import clear_output

from scripts import *


class CAModel(nn.Module):
    """Neural cellular automata model
    
       Parameters
       ----------
       device		device on which to perform computations
       n_channels	number of channels of the image
       fire_rate	parameter needed for the stochastic cell update
       losses		stores losses during training	
    """
    def __init__(self, n_channels=16, device=None, fire_rate=0.5):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.n_channels = n_channels
        
        self.fire_rate = fire_rate
        
        self.losses = []

	# network layers needed for the update rule
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels*3, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, n_channels, 1))
        
    
    def wrap_edges(self, x):
    	"""Sets boundary conditions to simulate a torus"""
        return F.pad(x, (1,1,1,1), 'circular', 0)

    
    def get_living_mask(self, x):
    	"""Gets portion of living cells in the image"""
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1


    def perceive(self, x, angle=0.):
    	"""Perception function, returnin the depthwise 2D convolution on the 16-channels images"""
        identity = torch.tensor([[0.,0.,0.],
                                 [0.,1.,0.],
                                 [0.,0.,0.]])
        dx = torch.tensor([[-0.125,0.,0.125],
                           [-0.25 ,0.,0.25 ],
                           [-0.125,0.,0.125]])
        dy = dx.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        dx, dy = c*dx - s*dy, s*dx + c*dy
        
        # create filters batch
        all_filters = torch.stack((identity, dx, dy))
        all_filters_batch = all_filters.repeat(self.n_channels,1,1).unsqueeze(1)
        all_filters_batch = all_filters_batch.to(self.device)
        # output depthwise convolutions over inputs x
        return F.conv2d(self.wrap_edges(x), all_filters_batch, groups=self.n_channels)
    

    def compute_dx(self, x, angle=0., step_size=1.):
        dx = self.layers(self.perceive(x, angle)) * step_size
        update_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.fire_rate

        return dx*update_mask.float()


    def forward(self, x, angle=0., step_size=1.):
    	"""Forward pass of the training procedure"""
        pre_life_mask = self.get_living_mask(x)
        
        # compute update increment
        dx = self.layers(self.perceive(x, angle)) * step_size
        
        # get random-per-cell mask for stochastic update
        update_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.fire_rate
        x += dx*update_mask.float()
        
        post_life_mask = self.get_living_mask(x)
        
        # get alive mask
        life_mask = pre_life_mask & post_life_mask

	# return updated states with alive masking
        return x * life_mask.float()
    
    
    def make_video(self, n_iters, video_size, regenerating = False, fname="video.mkv", rescaling=8, init_state=None, fps=10, **kwargs):
    	"""Outputs the video of the evolution of the Cellular Automaton"""
        if init_state is None:
            init_state = make_seed(1, self.n_channels, video_size)

        init_state = init_state.to(self.device)
        
        # set video visualization features
        video_size *= rescaling
        video = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        rescaler = T.Resize((video_size, video_size), interpolation=T.InterpolationMode.NEAREST)
        
        # evolution
        with torch.no_grad():
            for i in range(n_iters):
                video[i] = FloattoRGB(rescaler(init_state))[0].permute(1,2,0).cpu()
                init_state = self.forward(init_state)
                
                if regenerating:
                    if i == n_iters//3:
                        try: 
                            target_size = kwargs['target_size']
                        except KeyError:
                            target_size = None
                        try: 
                            constant_side = kwargs['constant_side']
                        except KeyError:
                            constant_side = None
                        init_state = make_squares(init_state, target_size=target_size, constant_side=constant_side)
        
        write_video(fname, video, fps=fps)
        
    
    def evolve(self, x, iters, angle=0., step_size=1.):
    	"""Performs the evolution of input images x towards the target pattern"""
        self.eval()
        with torch.no_grad():
            for i in range(iters):
                x = self.forward(x, angle=angle, step_size=step_size)
    
        return x
    

    def train_CA(self, optimizer, criterion, pool, n_epochs, scheduler=None, batch_size=4, evolution_iters=55, kind="growing", **kwargs):
    	"""Training routine for the Cellular Automaton"""
        self.train()

        for i in range(n_epochs):
            epoch_losses = []
            for j in range(pool.size // batch_size):
                inputs, indexes = pool.sample(batch_size)
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
            
            	# recursive forward-pass
                for k in range(evolution_iters + randint(-5, 5)):
                    inputs = self.forward(inputs)
                
                loss, idx_max_loss = criterion(inputs)
                epoch_losses.append(loss.item())
                if j%2 != 0:
                    idx_max_loss = None
                    
                # backward-pass
                loss.backward()
                optimizer.step()
                
                # customization of training for the three processes of growing. persisting and regenerating
                
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
                    inputs = make_squares(inputs, target_size=target_size, constant_side=constant_side)
                    
                # if training is not for growing proccess then re-insert trained/damaged samples into the pool
                if kind != "growing":
                    pool.update(inputs, indexes, idx_max_loss)

            if scheduler is not None:
                scheduler.step()

            self.losses.append(np.mean(epoch_losses))
            print(f"epoch: {i+1}\navg loss: {np.mean(epoch_losses)}")
            clear_output(wait=True)


    def eval_CA(self, criterion, image_size, eval_samples=128, evolution_iters=1000, batch_size=32):
    	"""Performs evaluation of the model over sample seed states""" 
        self.eval()
        self.evolution_losses = torch.zeros((evolution_iters), device="cpu")
        n = 0
        with torch.no_grad():
            for i in range(eval_samples // batch_size):
                inputs = make_seed(batch_size, self.n_channels, image_size, device=self.device)
                for j in range(evolution_iters):
                    inputs = self.forward(inputs)
                    loss, _ = criterion(inputs)
                    self.evolution_losses[j] = (n*self.evolution_losses[j] + batch_size*loss.cpu()) / (n+batch_size)
                n += batch_size
        pl.loglog(self.evolution_losses)
        return self.evolution_losses


    def load(self, fname):
    	"""Loads (pre-trained) model"""
        self.load_state_dict(torch.load(fname))
        print("Successfully loaded model!")
        
        
    def save(self, fname, overwrite=False):
        """Saves (trained) model"""
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        torch.save(self.state_dict(), fname)
        print("Successfully saved model!")