import torch
from torch import nn

class FutureTrajPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x[:, (self.obs_len+1):(self.obs_len+self.pred_len+1), :]
        # x = self.act_fn(self.linear(x))
        return x


class SIPPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x[:, (self.obs_len+self.pred_len+1)::(self.obs_len+1)]
        return x


class MTPPooler(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, num_nbr=4):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pool_ids = torch.ones((self.obs_len+1)*(num_nbr+1)+self.pred_len, dtype=bool)
        self.pool_ids[0] = False
        self.pool_ids[(self.obs_len+self.pred_len+1)::(self.obs_len+1)] = False

    def forward(self, x):
        x = x[:, self.pool_ids, :]
        return x