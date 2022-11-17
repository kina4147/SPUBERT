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


# class TrajHiddenGRUEncoder(nn.Module):
#     def __init__(self, mgp_hidden_size=2, hidden_size=64, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
#         super().__init__()
#         bidirectional = True
#         if bidirectional:
#             self.gru_enc = nn.GRU(input_size=mgp_hidden_size, hidden_size=int(hidden_size/2),
#                                         batch_first=True,
#                                         bidirectional=bidirectional)  # Batch * Time * Feature
#         else:
#             self.gru_enc = nn.GRU(input_size=mgp_hidden_size, hidden_size=hidden_size,
#                                         batch_first=True,
#                                         bidirectional=bidirectional)  # Batch * Time * Feature
#
#         self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, traj):
#         _, traj_h = self.gru_enc(traj)
#         traj_h = traj_h.permute(1, 0, 2)
#         traj_h = traj_h.reshape(-1, traj_h.shape[1] * traj_h.shape[2])
#         traj_h = self.LayerNorm(traj_h)
#         traj_h = self.dropout(traj_h)
#         return traj_h
#
#
# class TrajGRUEncoder(nn.Module):
#     def __init__(self, input_dim=2, hidden_size=64, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu', init=False):
#         super().__init__()
#         bidirectional = True
#         self.init = init
#         if bidirectional:
#             self.gru_enc = nn.GRU(input_size=input_dim, hidden_size=int(hidden_size/2),
#                                         batch_first=True,
#                                         bidirectional=bidirectional)  # Batch * Time * Feature
#             if init:
#                 self.init_pos_enc = nn.Linear(input_dim, int(hidden_size/2))
#                 self.act_fn = ACT2FN[act_fn]
#         else:
#             self.gru_enc = nn.GRU(input_size=input_dim, hidden_size=hidden_size,
#                                         batch_first=True,
#                                         bidirectional=bidirectional)  # Batch * Time * Feature
#             if init:
#                 self.init_pos_enc = nn.Linear(input_dim, hidden_size)
#                 self.act_fn = ACT2FN[act_fn]
#
#         self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, traj, init_pos=None):
#         if self.init:
#             init_pos = self.init_pos_enc(init_pos) # self.act_fn(self.init_pos_enc(init_pos))
#             init_pos = torch.stack([init_pos, torch.zeros_like(init_pos, device=init_pos.device)], dim=0)
#             _, traj_h = self.gru_enc(traj, init_pos)
#         else:
#             _, traj_h = self.gru_enc(traj)
#         traj_h = traj_h.permute(1, 0, 2)
#         traj_h = traj_h.reshape(-1, traj_h.shape[1] * traj_h.shape[2])
#         traj_h = self.LayerNorm(traj_h)
#         traj_h = self.dropout(traj_h)
#         return traj_h
#
# class TrajMLPEncoder(nn.Module):
#     def __init__(self, input_dim=2, hidden_size=64, traj_len=12, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
#         super().__init__()
#         self.linear1 = nn.Linear(traj_len*input_dim, int(traj_len/2) * hidden_size)
#         self.act_fn1 = ACT2FN[act_fn]
#         self.linear2 = nn.Linear(int(traj_len/2) * hidden_size, hidden_size)
#         self.act_fn2 = ACT2FN[act_fn]
#         self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, traj_h):
#         traj_h = traj_h.reshape(-1, traj_h.shape[1] * traj_h.shape[2])
#         traj_h = self.act_fn1(self.linear1(traj_h))
#         traj_h = self.act_fn2(self.linear2(traj_h))
#         traj_h = self.LayerNorm(traj_h)
#         traj_h = self.dropout(traj_h)
#         return traj_h

# class MGPEncoder(nn.Module):
#     def __init__(self, hidden_size=256, hidden_layers=[256, 256], layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
#         super().__init__()
#         self.linear1 = nn.Linear(hidden_size, hidden_layers[0])
#         self.act_fn1 = ACT2FN[act_fn]
#         self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
#         self.act_fn2 = ACT2FN[act_fn]
#         self.linear3 = nn.Linear(hidden_layers[1], hidden_size)
#         self.act_fn3 = ACT2FN[act_fn]
#         # self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, x):
#         x = self.act_fn1(self.linear1(x))
#         x = self.act_fn2(self.linear2(x))
#         x = self.linear3(x)
#         # x = self.LayerNorm(x)
#         x = self.dropout(x)
#         return x


# class GoalPriorNet(nn.Module):
#     def __init__(self, enc_hidden_size, goal_latent_size, act_fn='relu'):
#         super().__init__()
#         self.module = nn.Sequential(nn.Linear(enc_hidden_size, 128),
#                                     nn.ReLU(), # ACT2FN[act_fn],
#                                     nn.Linear(128, 64),
#                                     nn.ReLU(), # ACT2FN[act_fn],
#                                     nn.Linear(64, goal_latent_size*2))
#
#     def forward(self, x):
#         return self.module(x)

# class GoalRecognitionNet(nn.Module):
#     def __init__(self, enc_hidden_size, goal_hidden_size, goal_latent_size, act_fn='relu'):
#         super().__init__()
#         self.module = nn.Sequential(nn.Linear(enc_hidden_size + goal_hidden_size, 128),
#                                     nn.ReLU(), # ACT2FN[act_fn],
#                                     nn.Linear(128, 64),
#                                     nn.ReLU(), # ACT2FN[act_fn],
#                                     nn.Linear(64, goal_latent_size*2))
#     def forward(self, x):
#         return self.module(x)