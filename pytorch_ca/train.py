import torch
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
from random import randint
from IPython.display import clear_output
import numpy as np
import pylab as pl
import wandb
import av

from src import *

N_CHANNELS = 16        # Number of CA state channels
TARGET_PADDING = 8     # Number of pixels used to pad the target image border
TARGET_SIZE = 40       # Size of the target emoji
IMAGE_SIZE = TARGET_PADDING+TARGET_SIZE
POOL_SIZE = 2000
CELL_FIRE_RATE = 0.5
N_ITERS = 50


torch.backends.cudnn.benchmark = True # Speeds up things

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config={
    'gamma':0.3,
    'lr':5e-4,
    'batch_size': 120,
    'n_epochs':60,
    'step_size':1,
    'image':'t-rex_small',
    'scheduler':'step',
    'milestones':[20,40],
    'weight_decay':2e-2,
    'grad_noise':0.0,
    'starting_weights':'Pretrained_models/firework_growing.pt'
    }


growing=NeuralCA(N_CHANNELS,device)
if config['starting_weights'] != None:
  growing.load_state_dict(torch.load(config['starting_weights'], map_location=device))
  growing.to(device)


wandb.login()
#wandb.init(project='growing', entity="neural_ca", config=default_config, mode="disabled")
wandb.init(project='growing', entity="neural_ca", config=config)
config=wandb.config
print(config)
wandb.watch(growing, log_freq=32)

# Imports the target emoji
target = read_image("images/"+config['image']+".png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((TARGET_SIZE, TARGET_SIZE))(target)
target = RGBAtoFloat(target)
target = target.to(device)


# Starting state

def generator(n, device):
    return make_seed(n, N_CHANNELS-1, IMAGE_SIZE, alpha_channel=3, device=device)

pool = SamplePool(POOL_SIZE, generator)
#imshow(pool[0])


#torch.backends.cudnn.benchmark = True # Speeds up training
optimizer = torch.optim.Adam(growing.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
criterion = NCALoss(pad(target, TARGET_PADDING), torch.nn.MSELoss)
if config['scheduler']=='step':
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
else:
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

growing.grad_noise=config['grad_noise']

growing.train_CA(
    optimizer,
    criterion,
    pool,
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    scheduler=scheduler,
    skip_update=1,
    kind="growing")


converter=tensor_to_RGB(function="RGBA",CA=growing)
n_steps=300
seed=make_seed(1,N_CHANNELS-1,IMAGE_SIZE,n_CAs=1,alpha_channel=3,device=device)
fname="RGBA.mp4"
video,init_state=make_video(growing,n_steps,seed,converter=converter,fname=fname)

wandb.log({'video' : wandb.Video('RGBA.mp4',fps=10,format='mp4')})

growing.save(f"model.pt",overwrite=True)
wandb.save('model.pt')