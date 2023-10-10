import torch
from torch import nn

from torch.nn import functional as F

from pytorch_ca.src.models.neural_CA import NeuralCA

"""
TODO:

Make make sure that the original weights remain unaltered during training
"""


class LoraConvLayer(nn.Module):
    def __init__(self, conv_layer:nn.Conv2d, rank:int=1):
        """
            This class takes nn.Conv2d module, makes it non-trainable and adds a trainable lora layer initialized to zero

            all in all in about 50% slower than the conv_layer 
        """

        super().__init__()
        assert isinstance(conv_layer,nn.Conv2d)

        self.conv_layer=conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        #self.device=conv_layer.device

        assert type(rank)==int, f"rank must be an integer, got {type(rank)} instead"
        assert rank>0, f"rank must be positive, got {rank} instead"
        assert rank<min(self.in_channels,self.out_channels), f"why do you even use a rank that gives you more parameters than you started with lol"
        
        self.factor_matrix1=nn.Parameter(torch.zeros(self.out_channels,rank))
        self.factor_matrix2=nn.Parameter(torch.zeros(rank, self.in_channels))
        self.factor_bias=nn.Parameter(torch.zeros(rank))

        #self.to(conv_layer.device)
        conv_layer.weight.requires_grad=False
        conv_layer.bias.requires_grad=False

    def forward(self, images):

        weight=self.conv_layer.weight + (self.factor_matrix1@self.factor_matrix2)[...,None,None]
        bias=self.conv_layer.bias + (self.factor_matrix1@self.factor_bias)


        F.conv2d(images,weight,bias)



class LoraNeuralCA(NeuralCA):

    def __init__(self, NeuralCA:NeuralCA, rank:int):
        super().__init__(NeuralCA.n_channels,NeuralCA.device,NeuralCA.fire_rate)

        # Network layers needed for the update rule
        self.layers=nn.Sequential(
            LoraConvLayer(NeuralCA.layers[0]),
            nn.ReLU(),
            LoraConvLayer(NeuralCA.layers[2])
        )

