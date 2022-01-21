import torch
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
from random import randint
from IPython.display import clear_output
import numpy as np
import pylab as pl
import wandb
import av
import moviepy.editor as mvp


from src import *

N_CHANNELS = 15        # Number of CA state channels
TARGET_PADDING = 8     # Number of pixels used to pad the target image border
TARGET_SIZE = 40       # Size of the target emoji
CELL_FIRE_RATE = 0.5

torch.backends.cudnn.benchmark = True # Speeds up things

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config={
    'percentage':0.5,
    'gamma':1-1e-2,
    'lr1':2e-3,
    'lr2':2e-3,
    'batch_size': 50,
    'pool_size': 200,
    'n_epochs':50,
    'n_max_loss_ratio':8,
    'step_size':52,
    'evolution_iters':50,
    'target_size':40,
    'target_padding':16,
    'tau':1/100,
    'kind':'persist',
    'image':'firework',
    'start_appling_loss':100,
    'start_appling_kill_loss':100,
    'speed_appling_loss':2,
    'speed_appling_kill_loss':2,
    'kill_multiplier':1e-4,
    'reset_prob':0,
    'growing_file':'Pretrained_models/firework/Virus/firework_growing_64_96.pt',
    'virus_file':None
    }

IMAGE_SIZE = config['target_padding']+config['target_size']


#import the models
model = MultipleCA(N_CHANNELS, n_CAs=2, device=device)

vanishing='Pretrained_models/firework_vanishing.pt'
growing='Pretrained_models/firework/Virus/firework_growing_64_96.pt'
virus='Pretrained_models/firework/Virus/firework_virus_ckp_50%.pt'
last='model.pt'

if config['growing_file']!=None:
  model.CAs[0].load_state_dict(torch.load(config['growing_file'], map_location=device))
if config['virus_file']!=None:
  model.CAs[1].load_state_dict(torch.load(config['virus_file'], map_location=device))

for param in model.CAs[0].parameters():
    param.requires_grad = False

model.to(device)


wandb.login(key='c1d1eea97dcd6cda86b8a1b8cf18600174e3a38c')
#wandb.init(project='quick_virus', entity="neural_ca", config=default_config, mode="disabled")
wandb.init(project='quick_virus', entity="neural_ca", config=config)
config=wandb.config
print(config)
#wandb.watch(model, log_freq=32)


def prova():return [randint(70,90)]
generator=VirusGenerator(N_CHANNELS,IMAGE_SIZE,2,model,config['percentage'],iter_func=prova)

pool = SamplePool(config['pool_size'], generator,indexes_max_loss_size=8)


# Imports the target emoji
target = read_image("images/"+config['image']+".png", ImageReadMode.RGB_ALPHA).float()
target = T.Resize((config['target_size'], config['target_size']))(target)
target = RGBAtoFloat(target)
target = target.to(device)

class sigmoid:
    def __init__(self,sigma,x_0):
        self.sigma = sigma
        self.x_0 = x_0

    def sigmoid(self,x):
        x=(x-self.x_0)/self.sigma
        return 1/(1+np.exp(-x))

    def __call__(self,current_iteration,start_iteration,end_iteration,*args,**kwargs):
        return self.sigmoid(current_iteration)/(end_iteration-start_iteration)

def WeightFunction(current_iteration,start_iteration,end_iteration,*args,**kwargs):
    image_weight=sigmoid(config['speed_appling_loss'],config['start_appling_loss'])
    Nold_weight=sigmoid(config['speed_appling_kill_loss'],config['start_appling_kill_loss'])

    m1=image_weight(current_iteration,start_iteration,end_iteration)
    m2=Nold_weight(current_iteration,start_iteration,end_iteration)*config['kill_multiplier']

    N=1+config['kill_multiplier']
    return torch.tensor([m1/N,m2/N],device=device,requires_grad=False).float()


#set up the training 
params=model.CAs[1].parameters()
optimizer = torch.optim.Adam(params, lr=config['lr1'])
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,config['lr1'],config['lr1']+config['lr2'],config['step_size'],gamma=config['gamma'], cycle_momentum=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=config['gamma'])


image_loss = NCALoss(pad(target, config['target_padding']), alpha_channels=[15, 16])
percentage_loss = OldCellLoss(alpha_channel=-2)


criterion = CombinedLoss([image_loss, percentage_loss], WeightFunction)


def checkpoint(epoch):
    if epoch%10==0:
      with torch.no_grad():
          pool = SamplePool(20, generator)
          evolution_iters=500
          loss_per_step = model.test_CA(criterion,pool, evolution_iters)
          pl.plot(loss_per_step[:,0]*evolution_iters,label='MSE loss')
          pl.plot(loss_per_step[:,1]*evolution_iters,label='$N_{old}$ loss')
          pl.xlabel('$n_{steps}$')
          pl.ylabel('loss density')
          pl.yscale('log')
          pl.legend()
          pl.savefig("epoch"+str(epoch)+".png")
          pl.close()
          model.CAs[1].save(f"epoch"+str(epoch)+".pt",overwrite=True)
          wandb.log({"loss_graph":wandb.Image("epoch"+str(epoch)+".png")})
          wandb.save("epoch"+str(epoch)+".pt")


model.checkpoint=checkpoint


#The actual training part
model.train_CA(
    optimizer,
    criterion,
    pool,
    scheduler=scheduler,
    skip_update=1,
    n_max_losses=config['batch_size'] // config['n_max_loss_ratio'],
    skip_damage=2,
    **config)


converter=[tensor_to_RGB(function="RGBA",CA=model),
           tensor_to_RGB(function=[-2,-1],CA=model)]

seed=make_seed(1,N_CHANNELS,IMAGE_SIZE,n_CAs=2,alpha_channel=-2,device=device)
fname=["virus_50.mp4","virus_50_alpha.mp4"]
video,init_state=make_video(model,60,seed,converter=converter)
init_state=add_virus(init_state,-2,-1,config['percentage'])
_=make_video(model,300,init_state,fname=fname,initial_video=video,converter=converter, regenerating=False)


model.CAs[1].save(f"model.pt",overwrite=True)

wandb.save('model.pt')
for name in fname:
  wandb.log({name : wandb.Video(name,fps=10,format='mp4')})