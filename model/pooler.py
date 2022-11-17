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

# class SIPPooler(nn.Module):
#     def __init__(self, obs_len=8, pred_len=12, num_nbr=4):
#         super().__init__()
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.seq_input_len = (obs_len+1) * (num_nbr+1) + pred_len
#         self.pool_ids = torch.zeros(self.seq_input_len, dtype=bool)
#         self.pool_ids[(self.obs_len+self.pred_len+1)::(self.obs_len+1)] = True
#
#     def forward(self, x):
#         x = x[:, :self.seq_input_len, :]
#         x = x[:, self.pool_ids, :]
#         return x
#
#
# class MTPPooler(nn.Module):
#     def __init__(self, obs_len=8, pred_len=12, num_nbr=4):
#         super().__init__()
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.seq_input_len = (obs_len+1) * (num_nbr+1) + pred_len
#         self.pool_ids = torch.ones(self.seq_input_len, dtype=bool)
#         self.pool_ids[0] = False
#         self.pool_ids[(self.obs_len+self.pred_len+1)::(self.obs_len+1)] = False
#
#     def forward(self, x):
#         x = x[:, :self.seq_input_len, :]
#         x = x[:, self.pool_ids, :]
#         return x
#
# class PastTrajPooler(nn.Module):
#     def __init__(self, hidden_size, obs_len=8, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
#         super().__init__()
#         self.obs_len = obs_len
#         self.linear = nn.Linear(hidden_size, hidden_size)
#         self.act_fn = ACT2FN[act_fn]
#         # self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         # self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, x):
#         x = x[:, 1:self.obs_len+1, :]
#         x = self.act_fn(self.linear(x))
#         # x = self.LayerNorm(x)
#         # x = self.dropout(x)
#         return x
#
#
#
# class FullTrajPooler(nn.Module):
#     def __init__(self, hidden_size, obs_len, pred_len, layer_norm_eps=1e-4, dropout_prob=0.25, act_fn='relu'):
#         super().__init__()
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.linear = nn.Linear(hidden_size, hidden_size)
#         self.act_fn = ACT2FN[act_fn]
#         # self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         # self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, x):
#         x = x[:, 1+self.obs_len:1+self.obs_len+self.pred_len, :]
#         x = self.act_fn(self.linear(x))
#         # x = self.LayerNorm(x)
#         # x = self.dropout(x)
#         return x
# class SBertTargetTrajFineTuningPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.obs_len = config.obs_len
#         self.pred_len = config.pred_len
#         self.full_loss = config.full_loss
#
#     def forward(self, x):
#         if self.full_loss:
#             x = x[:, 1:1+self.obs_len+self.pred_len, :] # for pre-training
#         else:
#             x = x[:, 1+self.obs_len:1+self.obs_len+self.pred_len, :]  # for pre-training
#         return x

# class SBertTargetTrajPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.obs_len = config.obs_len
#         self.pred_len = config.pred_len
#         self.full_traj = config.full_loss or config.mask_loss
#
#     def forward(self, x):
#         if self.full_traj:
#             x = x[:, 1:1+self.obs_len+self.pred_len, :] # for pre-training
#         else:
#             x = x[:, 1+self.obs_len:1+self.obs_len+self.pred_len, :]  # for pre-training
#         return x

#
# class SBertNeighborTrajPreTrainingPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         if config.neighbor_prediction:
#             self.pool_ids = torch.ones((config.obs_len+config.pred_len+1)*(config.num_nbr+1), dtype=bool)
#             self.pool_ids[:config.obs_len + config.pred_len + 2] = False
#             self.pool_ids[config.obs_len+config.pred_len+1::(config.obs_len+config.pred_len+1)] = False
#         else:
#             self.pool_ids = torch.ones((config.obs_len + 1) * (config.num_nbr + 1) + config.pred_len, dtype=bool)
#             self.pool_ids[:config.obs_len+config.pred_len+2] = False
#             self.pool_ids[config.obs_len+config.pred_len+1::(config.obs_len+1)] = False
#
#     def forward(self, x):
#         x = x[:, self.pool_ids, :]
#         return x
#
#
# class SBertNeighborTrajFineTuningPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         if config.neighbor_prediction:
#             if config.full_loss:
#                 self.pool_ids = torch.ones((config.obs_len+config.pred_len+1)*(config.num_nbr+1), dtype=bool)
#                 self.pool_ids[:config.obs_len + config.pred_len + 2] = False
#                 self.pool_ids[config.obs_len+config.pred_len+1::(config.obs_len+config.pred_len+1)] = False
#             else:
#                 self.pool_ids = [False]*(config.obs_len+config.pred_len+1)+\
#                                 ([False]*(config.obs_len+1)+[True]*config.pred_len)*config.num_nbr
#                 self.pool_ids = torch.tensor(self.pool_ids)
#         else:
#             self.pool_ids = torch.ones((config.obs_len + 1) * (config.num_nbr + 1) + config.pred_len, dtype=bool)
#             self.pool_ids[:config.obs_len+config.pred_len+2] = False
#             self.pool_ids[config.obs_len+config.pred_len+1::(config.obs_len+1)] = False
#
#     def forward(self, x):
#         x = x[:, self.pool_ids, :]
#         return x
#
# class SBertSocialPreTrainingPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.target_length = config.obs_len + config.pred_len
#         if config.neighbor_prediction:
#             self.neighbor_length = config.obs_len + config.pred_len
#         else:
#             self.neighbor_length = config.obs_len
#
#     def forward(self, x):
#         x = x[:, self.target_length+1::(self.neighbor_length+1)]
#         return x
#
#
# class GoalPooler(nn.Module):
#     def __init__(self, cfgs):
#         super().__init__()
#         self.target_pooler = SBertTargetTrajPreTrainingPooler(cfgs)
#         if cfgs.neighbor_observation:
#             self.neighbor_pooler = SBertNeighborTrajPreTrainingPooler(cfgs)
#         else:
#             self.neighbor_pooler = None
#         if cfgs.social_interaction:
#             self.social_pooler = SBertSocialPreTrainingPooler(cfgs)
#         else:
#             self.social_pooler = None
#
#     def forward(self, hidden_state):
#         social_output = None
#         neighbor_output = None
#         target_output = self.target_pooler(hidden_state)
#         if self.social_pooler:
#             social_output = self.social_pooler(hidden_state)
#         if self.neighbor_pooler:
#             neighbor_output = self.neighbor_pooler(hidden_state)
#         return target_output, neighbor_output, social_output
#


# class TrajPooler(nn.Module):
#     def __init__(self, cfgs):
#         super().__init__()
#         self.target_pooler = SBertTargetTrajPreTrainingPooler(cfgs)
#         if cfgs.neighbor_observation:
#             self.neighbor_pooler = SBertNeighborTrajPreTrainingPooler(cfgs)
#         else:
#             self.neighbor_pooler = None
#         if cfgs.social_interaction:
#             self.social_pooler = SBertSocialPreTrainingPooler(cfgs)
#         else:
#             self.social_pooler = None
#
#     def forward(self, hidden_state):
#         social_output = None
#         neighbor_output = None
#         target_output = self.target_pooler(hidden_state)
#         if self.social_pooler:
#             social_output = self.social_pooler(hidden_state)
#         if self.neighbor_pooler:
#             neighbor_output = self.neighbor_pooler(hidden_state)
#         return target_output, neighbor_output, social_output
