import torch
import torch.nn as nn
import numpy as np



def bom_loss(pred_goals, pred_trajs, gt_goals, gt_trajs, k_sample, output_dim=2):
    '''
    CVAE loss use best-of-many
    Params:
        pred_goal: (B, K, Sdim)
        pred_traj: (B, K, Seq, Sdim)
        gt_goals: (B, Sdim)
        gt_trajs: (B, Seq, Sdim)
        best_of_many: whether use best of many loss or not
    Returns:

    '''
    gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1) # (B, T, S) => (B, K, T, S)
    gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1) # (B, S) => (B, K, S)

    goal_rmse = torch.sqrt(torch.sum((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2, dim=-1))
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1))
    traj_fde_rmse = traj_rmse[:, :, -1] # (B, K, T) => (B, K)
    traj_ade_rmse = traj_rmse.mean(dim=-1) # (B, K, T) => (B, K)

    best_gde = torch.min(goal_rmse, dim=-1)[0].sum()
    best_fde = torch.min(traj_fde_rmse, dim=-1)[0].sum()
    best_ade = torch.min(traj_ade_rmse, dim=-1)[0].sum()
    return best_gde, best_ade, best_fde


class ADELoss(nn.Module):
    def __init__(self):
        super(ADELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, input, target):
        loss = torch.sqrt(torch.sum(((input - target) ** 2.0) + self.eps, dim=2)).mean()
        return loss


class FDELoss(nn.Module):
    def __init__(self):
        super(FDELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, input, target):
        loss = torch.sqrt(torch.sum(((input - target) ** 2.0) + self.eps, dim=1)).mean()
        return loss


class MGPCVAELoss(nn.Module):
    def __init__(self):
        super(MGPCVAELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, pred_goals, gt_goals, k_sample, output_dim=2, best=True):
        '''
        CVAE loss use best-of-many
        Params:
            pred_goal: (B, K, Sdim)
            pred_traj: (B, T, Seq, Sdim)
            pred_goal: (B, K, Sdim)
            pred_traj: (B, K, Seq, Sdim)
            gt_goals: (B, Sdim)
            gt_trajs: (B, Seq, Sdim)
            best_of_many: whether use best of many loss or not
        Returns:

        '''
        gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1) # (B, S) => (B, K, S)
        goal_rmse = torch.sqrt(torch.sum(((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2) + self.eps, dim=-1)) # (B, K, S) => (B, K)
        best_idx = torch.argmin(goal_rmse, dim=-1)
        if best:
            goal_loss = goal_rmse[range(len(best_idx)), best_idx].mean()
        else:
            goal_loss = goal_rmse.mean()

        return goal_loss, best_idx

