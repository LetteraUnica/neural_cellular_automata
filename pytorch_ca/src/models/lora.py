import torch
from torch import nn

from torch.nn import functional as F

class LoraConvLayer(nn.Module):
    def __init__(self, conv_layer:nn.Conv2d, rank:int=1):
        """
            This class takes nn.Conv2d module, makes it non-trainable and adds a trainable lora layer initialized to zero 
        """

        super().__init__()
        assert isinstance(conv_layer,nn.Conv2d)

        self.conv_layer=conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.device=conv_layer.device

        assert type(rank)==1, f"rank must be an integer, got {type(rank)} instead"
        assert rank>0, f"rank must be positive, got {rank} instead"
        assert rank<min(self.in_channels,self.out_channels), f"why do you even use a rank that gives you more parameters than you started with lol"
        
        self.factor_matrix1=nn.Parameter(torch.zeros(self.in_channels,rank))
        self.factor_matrix2=nn.Parameter(torch.zeros(rank, self.out_channels))

        self.to(conv_layer.device)
        conv_layer.no_grad()

    def forward(self, images):

        weight=conv_layer

        F.conv2d(images,weight,bias)