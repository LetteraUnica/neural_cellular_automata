from hashlib import new
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pylab as pl
from random import randint

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# Utility functions

def RGBAtoFloat(images):
    """Converts images in 0-1 range"""
    return torch.clip(images.float() / 255, 0., 1.)

def FloattoRGBA(images):
    """Converts images in 0-255 range"""
    return torch.clip((images * 255), 0, 255).type(torch.uint8)

def FloattoRGB(images):
    """Converts a 0-1 float image with an alpha channel into RGB"""
    if len(images.size()) < 4:
        images = torch.unsqueeze(images, 0)
    return torch.clip(images[:,:3,:,:] * images[:,3,:,:] * 255 + (1-images[:,3,:,:])*255, 0, 255).type(torch.uint8)

def FloattoTrain(images, apply_tanh=False):
    """Converts a 0-1 float image to [-1,1] range"""
    images = images * 2 - 1
    if apply_tanh:
        images = torch.tanh(images*2)/np.tanh(2)

    return images

def TraintoFloat(images, inverse_tanh=False):
    """Converts a [-1,1] float image to 0-1 range"""
    images = torch.clip(images, -1., 1.)
    if inverse_tanh:
        images = torch.atanh(images * np.tanh(2)) / 2
    return images * 0.5 + 0.5

def FloattoGrayscale(image, cmap="viridis"):
    if len(image.size()) > 2:
        Exception(f"images must be a 1d or 2d tensor, got {image.shape} instead")
        return
    with torch.no_grad():
        scale = torch.max(image) - torch.min(image)
        if scale < 1e-6:
            image = torch.zeros_like(image)
        else: 
            image = (image - torch.min(image)) / scale
    viridis = cm.get_cmap(cmap)
    return torch.tensor(viridis(image)).permute(2,0,1)
    
def center_crop(images, size):
    """Center crops an image"""
    return T.CenterCrop(size)(images)

def pad(images, padding, fill=0.):
    "Pads the tensor images"
    return T.Pad(padding//2, fill=fill)(images)

def imshow(image, apply_center_crop=False):
    """Prints an image"""
    if apply_center_crop:
        image = center_crop(image)
    pl.imshow(np.asarray(image.cpu().permute(1,2,0)[:,:,:4]))


def make_seed(n_images, n_channels, image_size, device="cpu"):
    """Makes the seed to start the CA, i.e. a black dot"""
    start_point = torch.zeros((n_images, n_channels, image_size, image_size), device=device)
    start_point[:, 3, image_size//2, image_size//2] = 1.
    return start_point

def moving_average(x, w):
    """Computes moving average of x""" 
    return np.convolve(x, np.ones(w), 'valid') / w

def side(size, constant_side=False):
    """Return size of the side to be used to erase square portion of the images"""
    if constant_side:
        return size//2
    return randint(size//6, size//2)

def make_squares(images, target_size=None, side=side, constant_side=False):
    """Sets square portion of input images to zero"""
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        y = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        images[i, :, x-side(target_size, constant_side)//2:x+side(target_size, constant_side)//2, y-side(target_size, constant_side)//2:y+side(target_size, constant_side)//2] = 0.

    return images
    
def make_poligon(images, target_size=None, side=side):
    """Sets random poligonal portion of input images to zero"""
    if target_size is None:
        target_size = images.size()[-1]
    for i in range(images.size()[0]):
        x1 = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        x2 = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        y1 = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        y2 = randint(target_size//2-target_size//4, target_size//2+target_size//4)
        images[i, :, x1-side(target_size)//2:x2+side(target_size)//2, y1-side(target_size)//2:y2+side(target_size)//2] = 0.
    
    return images

# Custom loss function
class loss_fn:
    """Custom l2 or l1 loss function"""
    def __init__(self, target, order=2):
        self.order = order
        self.target = target
        
    def __call__(self, x):
        losses = torch.mean(torch.abs(x[:, :4, :, :] - self.target)**self.order, [1,2,3])
        idx_max_loss = torch.argmax(losses)
        loss = torch.mean(losses)
        return loss, idx_max_loss

# Sample pool dataset
class SamplePool(Dataset):
    """Samples the training images
    
    Parameters
    ----------
    images	tensor of samples to be trained 
    size	size of the pool
    n_channels	number of image channels
    image_size	size of the input image  
    """
    def __init__(self, pool_size, n_channels, image_size, transform = None, device="cpu"):
        self.images = make_seed(pool_size, n_channels, image_size, device)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

        self.device = torch.device(device)
    
    def __len__(self):
        return self.size

    
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), idx

    
    def transform(self, transformation):
        self.images = transformation(self.images)

    
    def sample(self, batch_size):
        """Extracts random batch of size batch_size frome the pool"""
        idx = np.random.choice(self.size, batch_size, False)
        return self.transform(self.images[idx]), idx
    

    def update(self, new_images, idx, idx_max_loss=None):
        """Raplaces images at indeces idx of the pool with new_images"""
        self.images[idx] = new_images.detach().to(self.device)
        if idx_max_loss is not None:
            self.images[idx[idx_max_loss]] = make_seed(1, self.n_channels, self.image_size)[0]
            