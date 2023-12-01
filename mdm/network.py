import math
import numpy as np

import torch
import torch.nn as nn

from ipdb import set_trace as debug

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

        self.norms = nn.ModuleList(
            # [nn.BatchNorm1d(hidden_dim) for _ in range(num_res_blocks)]
            [nn.GroupNorm(32, hidden_dim) for _ in range(num_res_blocks)]
        )

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / np.sqrt(2)
        return h

class ToyPolicy(torch.nn.Module):
    def __init__(self, hidden_dim=256, time_embed_dim=128, data_dim=2, zero_out_last_layer=False):
        super(ToyPolicy,self).__init__()


        self.time_embed_dim = time_embed_dim
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=3)

        self.norm = nn.GroupNorm(32, hid)

        self.out_module = nn.Sequential(
            nn.Linear(hid,hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )
        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self,x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out   = self.out_module(self.norm(x_out+t_out))

        return out
