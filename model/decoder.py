import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class GoalDecoder(nn.Module):
    def __init__(self, enc_hidden_size, goal_latent_size, out_dim=2, hidden_layers=[128, 64],
                 layer_norm_eps=1e-4, act_fn='relu'):
        super().__init__()

        self.linear1 = nn.Linear(enc_hidden_size+goal_latent_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        self.decoder = nn.Linear(hidden_layers[0], out_dim)

        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x


class SpatialDecoder(nn.Module):
    def __init__(self, enc_hidden_size, out_dim=2, hidden_layers=[128, 64],
                 layer_norm_eps=1e-4, act_fn='relu'):
        super().__init__()

        self.linear1 = nn.Linear(enc_hidden_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        self.decoder = nn.Linear(hidden_layers[0], out_dim)

        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x

