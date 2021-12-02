import torch
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
from random import randint
from IPython.display import clear_output
import moviepy.editor as mvp
import numpy as np
import pylab as pl
import wandb
import av

from src import *


torch.backends.cudnn.benchmark = True # Speeds up things

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config={
    'gamma': 0.1,
    'lr':2e-3,
    'batch_size': 25,
    'n_epochs':100,
    'step_size':1,
    'image':'butterfly',
    'scheduler':'step',
    'milestones':[50,100],
    'weight_decay':0,
    'starting_weights': None,
    'evolution_iters':[64,96],
    'target_size':40,
    'target_padding':8,
    'pool_size':1000,
    'optimizer':'Adam',
    }

N_CHANNELS = 16        # Number of CA state channels
IMAGE_SIZE = 2*config['target_padding']+config['target_size']
CELL_FIRE_RATE = 0.5
N_ITERS = 50

growing=NeuralCA(N_CHANNELS,device)
for p in growing.parameters():
  print(p.shape)
  if p.shape[0]==16:
    p.data.fill_(0)
if config['starting_weights'] != None:
  growing.load_state_dict(torch.load(config['starting_weights'], map_location=device))
  growing.to(device)

#wandb.init(project='growing', entity="neural_ca", config=default_config, mode="disabled")
wandb.init(project='growing', entity="neural_ca", config=config)
config=wandb.config
print(config)
wandb.watch(growing, log_freq=32)

# Imports the target emoji
target = read_image("images/"+config['image']+".png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((config['target_size'], config['target_size']))(target)
target = pad(target,config['target_padding'])
target = RGBAtoFloat(target)
target = target.to(device)

# Starting state

def generator(n, device):
    return make_seed(n, N_CHANNELS-1, IMAGE_SIZE, alpha_channel=3, device=device)

pool = SamplePool(config['pool_size'], generator)
#imshow(pool[0])

if config['optimizer']=='Adam':
  optimizer = torch.optim.Adam(growing.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
if config['optimizer']=='SGD':
    optimizer = torch.optim.SGD(growing.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

if config['scheduler']=='step':
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
if config['scheduler']=='exponential':
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

criterion = NCALoss(pad(target, config['target_padding']), torch.nn.MSELoss)

growing.train_CA(
    optimizer,
    criterion,
    pool,
    batch_size=config['batch_size'],
    n_epochs=config['n_epochs'],
    scheduler=scheduler,
    skip_update=1,
    kind="growing",
    evolution_iters=config['evolution_iters'],
    normalize_gradients=True)

converter=tensor_to_RGB(function="RGBA",CA=growing)
n_steps=300
seed=make_seed(1,N_CHANNELS-1,IMAGE_SIZE,n_CAs=1,alpha_channel=3,device=device)
fname="RGBA.mp4"
make_video(growing,n_steps,seed,converter=converter,fname=fname)


growing.save(f"model.pt",overwrite=True)
wandb.save('model.pt')
wandb.log({'video' : wandb.Video('RGBA.mp4',fps=10,format='mp4')})