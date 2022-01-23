import torch
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
from random import randint
from IPython.display import clear_output
import moviepy.editor as mvp
import wandb

import numpy as np
import pylab as pl

from src import *

import sys
sys.setrecursionlimit(2000)
torch.backends.cudnn.benchmark = True # Speeds up things


config = {
    'image':'firework.png',
    'growing_file':'Pretrained_models/firework/Virus/firework_growing_64_96.pt',
    'virus_file':'Pretrained_models/firework/Virus/firework_virus_ckp_50%.pt',

    'target_size':40,
    'target_padding':8,
    'n_channels':15,

    'pool_size': 512,
    'generator_iters_min':64,
    'generator_iters_max':96,
    'virus_rate':0.45,

    'start_appling_loss':64,
    'speed_appling_loss':1,
    'start_appling_kill_loss':84,
    'speed_appling_kill_loss':1,
    'kill_multiplier':1e-3,
    'perturbation_multiplier':1e-3,

    'lr':1e-3,
    'batch_size':20,
    'n_epochs':20,
    'n_max_loss_ratio':8,
    'evolution_iters':96,
    'kind':'persist',
    'skip_update':2,
    'skip_damage':1,
    'reset_prob':0,

    'step_size':10,
    'gamma':0.3,

    'trained_model_name': 'model'
    }

config["n_max_losses"] = max(1, config["batch_size"] // config["n_max_loss_ratio"])
config["image_size"] = config["target_size"] + config["target_padding"]


# Imports the target emoji
target = read_image("images/" + config["image"], ImageReadMode.RGB_ALPHA).float()
target = T.Resize((config["target_size"], config["target_size"]))(target)
target = RGBAtoFloat(target)


# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = target.to(device)

model = MultipleCA(config["n_channels"], n_CAs=2, device=device)
model.CAs[0].load(config["growing_file"])
if config['virus_file']!=None: model.CAs[1].load(config["virus_file"])
else: model.CAs[1].load(config["growing_file"])

for param in model.CAs[0].parameters():
    param.requires_grad = False

generator = VirusGenerator(config["n_channels"], config["image_size"], 2, model, virus_rate=config["virus_rate"],
                           iter_func=ExponentialSampler(min=config["generator_iters_min"], max=config["generator_iters_max"]))
pool = SamplePool(config["pool_size"], generator)


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


losses = [NCALoss(pad(target, config["target_padding"]), alpha_channels=[-1, -2]),
          OldCellLoss()]

criterion = CombinedLoss(losses, WeightFunction)

def end_step_loss(inputs, **params):
    return NCADistance(model.CAs[0], model.CAs[1])(inputs)*config['perturbation_multiplier']

#model.end_step_loss=end_step_loss


def checkpoint(epoch):
    if epoch%10==0:
      with torch.no_grad():
          pool = SamplePool(20, generator)
          evolution_iters=500
          loss_per_step = model.test_CA(criterion,pool, evolution_iters)
          MSE = loss_per_step[:,0]
          N_old = loss_per_step[:,1]
          pl.close()
          pl.plot(MSE*evolution_iters,label="MSE")
          pl.plot(N_old*evolution_iters,label="$N_{old}$")
          pl.yscale('log')
          pl.xlabel("$n_{steps}$")
          pl.ylabel("loss density")
          pl.legend()
          pl.savefig("epoch"+str(epoch)+".png")
          pl.show()
          model.CAs[1].save(f"epoch"+str(epoch)+".pt",overwrite=True)
          wandb.save("epoch"+str(epoch)+".pt")
          wandb.log({"loss_graph":wandb.Image("epoch"+str(epoch)+".png")})


model.checkpoint=checkpoint


params = model.CAs[1].parameters()

optimizer = torch.optim.Adam(params, lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["step_size"], config["gamma"])

model.train_CA(optimizer, criterion, pool, scheduler=scheduler, **config)

model.CAs[1].save(config["trained_model_name"] + ".pt")

wandb.save(config["trained_model_name"] + ".pt")


# Log pool samples
imgs = []

for i in range(16):
    img = pool.sample(1)[0].detach().cpu()
    img = state_to_image(img, [-1, -2])[0]
    img = np.asarray(img.permute(1, 2, 0)[:, :, :4])
    img = wandb.Image(img)
    imgs.append(img)

wandb.log({"pool samples": imgs})


converter=[tensor_to_RGB(function="RGBA",CA=model),
           tensor_to_RGB(function=[-2,-1],CA=model)]

seed=make_seed(1,config["n_channels"],config["image_size"],n_CAs=2,alpha_channel=-2,device=device)
fname=["virus.mp4","virus_alpha.mp4"]
video,init_state=make_video(model,60,seed,converter=converter)
init_state=add_virus(init_state,-2,-1,config['virus_rate'])
_=make_video(model,300,init_state,fname=fname,initial_video=video,converter=converter, regenerating=False)

for name in ["virus_alpha.mp4", "virus.mp4"]:
    wandb.log({name: wandb.Video(name, fps=10)})