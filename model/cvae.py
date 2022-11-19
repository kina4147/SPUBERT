import torch.nn as nn
from transformers.activations import ACT2FN

class GoalRecognitionNet(nn.Module):
    def __init__(self, enc_hidden_size, goal_hidden_size, goal_latent_size, hidden_layers=[128, 64], layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
        super().__init__()
        self.linear1 = nn.Linear(enc_hidden_size+goal_hidden_size, hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.act_fn2 = ACT2FN[act_fn]
        self.linear3 = nn.Linear(hidden_layers[1], goal_latent_size*2)

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.act_fn2(self.linear2(x))
        x = self.linear3(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, spatial_dim=2, goal_hidden_size=64, hidden_layers=[32, 64], layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
        super().__init__()
        self.linear1 = nn.Linear(spatial_dim, goal_hidden_size) #hidden_layers[0])
        self.act_fn1 = ACT2FN[act_fn]
        self.LayerNorm = nn.LayerNorm(goal_hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.act_fn1(self.linear1(x))
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x
