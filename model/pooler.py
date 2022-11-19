from torch import nn
from transformers.activations import ACT2FN


class FutureTrajPooler(nn.Module):
    def __init__(self, hidden_size, obs_len=8, pred_len=12, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[act_fn]
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x[:, 1+self.obs_len:1+self.obs_len+self.pred_len, :]
        x = self.act_fn(self.linear(x))
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class GoalPooler(nn.Module):
    def __init__(self, hidden_size, obs_len, pred_len, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[act_fn]
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x[:, self.obs_len+self.pred_len, :]
        x = self.act_fn(self.linear(x))
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x
