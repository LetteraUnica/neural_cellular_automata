import torch
from torch import nn

from torch.nn import functional as F


"""
TODO:

Make make sure that the original weights remain unaltered during training
"""


class LoraConvLayer(nn.Module):
    def __init__(self, conv_layer:nn.Conv2d, rank:int=1):
        """This class takes nn.Conv2d module, makes it non-trainable and adds a trainable lora layer initialized to zero. 
        All in all in about 50% slower than the conv_layer 

        When initialized, it makes sure that the parameters of the conv_layer are not altered during training

        Args:
            conv_layer (nn.Conv2d): nn.Conv2d module to be wrapped
            rank (int, optional): rank of the lora layer. Defaults to 1.        
        """

        super().__init__()
        assert isinstance(conv_layer,nn.Conv2d)

        self.conv_layer=conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size

        assert type(rank)==int, f"rank must be an integer, got {type(rank)} instead"
        assert rank>0, f"rank must be positive, got {rank} instead"
        assert rank<min(self.in_channels,self.out_channels), f"why do you even use a rank that gives you more parameters than you started with lol"

        #TODO: Improve initialization 
        self.factor_matrix1=nn.Parameter(torch.randn(self.out_channels,rank))
        self.factor_matrix2=nn.Parameter(torch.randn(rank, self.in_channels))
        self.factor_bias=nn.Parameter(torch.randn(rank))

        conv_layer.weight.requires_grad=False
        conv_layer.bias.requires_grad=False

        self.device=conv_layer.weight.device
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        self.lora_weight = (self.factor_matrix1 @ self.factor_matrix2)[...,None,None]
        self.lora_bias = self.factor_matrix1 @ self.factor_bias        
        
        weight = self.conv_layer.weight + self.lora_weight
        bias = self.conv_layer.bias + self.lora_bias


        return F.conv2d(images,weight,bias)