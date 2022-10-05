
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class MTPDecoder(nn.Module):
    """
    Masked Trajectory Prediction
    """
    def __init__(self, hidden_size, out_dim=2, hidden_layers=[128, 64],
                 layer_norm_eps=1e-4, act_fn='relu'):
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        # self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.act_fn2 = ACT2FN[act_fn]
        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_layers[0], out_dim)
        # self.bias = nn.Parameter(torch.zeros(out_dim))
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        # x = self.act_fn2(self.linear2(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x

# Social Interaction
class SIPDecoder(nn.Module):
    """
    Social Interaction Prediction
    2-class classification model : is_near, is_not_near
    """
    def __init__(self, hidden_size, num_class=3):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# class GoalDecoder(nn.Module):
#     def __init__(self, hidden_size, goal_latent_size, out_dim=2, layer_norm_eps=1e-4, act_fn='relu'):
#         super().__init__()
#
#         # self.linear1 = nn.Linear(hidden_size, 128)
#         # self.act_fn1 = ACT2FN[act_fn]
#         # self.linear2 = nn.Linear(128, 64)
#         # self.act_fn2 = ACT2FN[act_fn]
#         # self.layer_norm = nn.LayerNorm(64, eps=layer_norm_eps)
#         # self.linear3 = nn.Linear(64, out_dim, bias=False)
#         self.linear1 = nn.Linear(hidden_size+goal_latent_size, hidden_size)
#         self.act_fn = ACT2FN[act_fn]
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.linear2 = nn.Linear(hidden_size, out_dim, bias=False)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.act_fn(x)
#         x = self.layer_norm(x)
#         x = self.linear2(x)
#         return x
#         # x = self.linear1(x)
#         # x = self.act_fn1(x)
#         # return self.decoder(x)

class GoalDecoder(nn.Module):
    def __init__(self, enc_hidden_size, goal_latent_size, out_dim=2, hidden_layers=[128, 64],
                 layer_norm_eps=1e-4, act_fn='relu'):
        super().__init__()

        # self.linear1 = nn.Linear(enc_hidden_size+goal_latent_size, out_dim)
        # self.act_fn1 = ACT2FN[act_fn]

        self.linear1 = nn.Linear(enc_hidden_size+goal_latent_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        # self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.act_fn2 = ACT2FN[act_fn]
        self.decoder = nn.Linear(hidden_layers[0], out_dim)

        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        # x = self.act_fn2(self.linear2(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x


# class SpatialDecoder(nn.Module):
#     def __init__(self, hidden_size, out_dim=2, layer_norm_eps=1e-4, act_fn='relu'):
#         super().__init__()
#         self.linear1 = nn.Linear(hidden_size, hidden_size)
#         self.act_fn = ACT2FN[act_fn]
#         self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.linear2 = nn.Linear(hidden_size, out_dim, bias=False)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.act_fn(x)
#         x = self.layer_norm(x)
#         x = self.linear2(x)
#         return x

class SpatialDecoder(nn.Module):
    def __init__(self, enc_hidden_size, out_dim=2, hidden_layers=[128, 64],
                 layer_norm_eps=1e-4, act_fn='relu'):
        super().__init__()

        self.linear1 = nn.Linear(enc_hidden_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        # self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.act_fn2 = ACT2FN[act_fn]
        self.decoder = nn.Linear(hidden_layers[0], out_dim)

        self.LayerNorm = nn.LayerNorm(hidden_layers[0], eps=layer_norm_eps)
        self.decoder.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        # x = self.act_fn2(self.linear2(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x


#
# # Social Interaction
# class WalkingBertSIPredictionHead(nn.Module):
#     """
#     2-class classification model : is_near, is_not_near
#     """
#
#     def __init__(self, config):
#         """
#         :param hidden: BERT model output size
#         """
#         super().__init__()
#         self.linear = nn.Linear(config.hidden_size, 3)
#         self.softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, x):
#         return self.softmax(self.linear(x))
