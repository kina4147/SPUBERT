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

# class MaskedADELoss(nn.Module):
#     def __init__(self):
#         super(MaskedADELoss, self).__init__()
#         self.eps = 1e-8
#
#     def forward(self, input, target, mask, weight=None):
#         l2_sum = torch.sqrt(torch.sum(((input - target) ** 2.0) * mask + self.eps, dim=2)).sum()
#         num_mask = torch.sum(mask)/2 #(torch.sum(mask, dim=2).sum(dim=1).sum() / 2) ** 2
#         result = l2_sum / num_mask
#         return result

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


# class TGPCVAELoss(nn.Module):
#     def __init__(self):
#         super(TGPCVAELoss, self).__init__()
#         self.eps = 1e-8
#
#     def forward(self, best_goal_idx, pred_goals, pred_trajs, gt_trajs, k_sample, output_dim=2):
#         gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1) # (B, T, S) => (B, K, T, S)
#         fde_rmse = torch.sqrt(torch.sum(((pred_trajs[:, :, -1, :output_dim] - pred_goals[:, :, :output_dim]) ** 2) + self.eps, dim=-1))
#         ade_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1) + self.eps, dim=-1).sum(dim=-1)
#         fde_loss = fde_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#         ade_loss = ade_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#         return ade_loss, fde_loss
#
#
# class MGPTGPFDELoss(nn.Module):
#     def __init__(self):
#         super(MGPTGPFDELoss, self).__init__()
#         self.eps = 1e-8
#
#     def forward(self, best_goal_idx, pred_goals, pred_trajs):
#         best_pred_goals = pred_goals[range(len(best_goal_idx)), best_goal_idx]
#         fde_loss = torch.sqrt(torch.sum((pred_trajs[:, -1, :] - best_pred_goals) ** 2, dim=-1) + self.eps, dim=-1).mean()
#         return fde_loss
#
# class InfoVAELoss(nn.Module):
#     def __init__(self):
#         super(InfoVAELoss, self).__init__()
#
#     def compute_kernel(self, x, y):
#         x_size = x.shape[0]
#         y_size = y.shape[0]
#         dim = x.shape[1]
#         tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
#         tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
#         return torch.exp(-torch.mean((tiled_x - tiled_y).float() ** 2, dim=2) / dim)
#
#     def forward(self, z, latent_dim, device):
#         # z: Batch * Sample * Lat Dim
#         z_true = torch.randn(1000, latent_dim).to(device)
#         # print(z.shape, z_true.shape)
#         z_true_kernel = compute_kernel(z_true, z)
#         z_kernel = compute_kernel(z, z)
#         z_true_z_kernel = compute_kernel(z_true, z)
#         # mmd = torch.mean(z_true_kernel) + torch.mean(z_kernel) - 2 * torch.mean(z_true_z_kernel)
#         return torch.mean(z_true_kernel) + torch.mean(z_kernel) - 2 * torch.mean(z_true_z_kernel)
#
# class BetaTCVAELoss(nn.Module):
#     def __init__(self):
#         super(BetaTCVAELoss, self).__init__()
#
#
#     def forward(self, z, z_mean, z_logvar, w_tc=1.0):
#         """Estimate of total correlation on a batch.
#         Borrowed from https://github.com/google-research/disentanglement_lib/
#         Args:
#           z: [batch_size, num_latents]-tensor with sampled representation.
#           z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
#           z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
#         Returns:
#           Total correlation estimated on a batch.
#         """
#         log_qz_prob = gaussian_log_density(z.unsqueeze(dim=1),
#                                            z_mean.unsqueeze(dim=0),
#                                            z_logvar.unsqueeze(dim=0))
#         log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)
#
#         log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()
#
#         return (w_tc - 1)*(log_qz - log_qz_product).mean()
#
#
#     def gaussian_log_density(self, samples, mean, log_var):
#         """ Estimate the log density of a Gaussian distribution
#         Borrowed from https://github.com/google-research/disentanglement_lib/
#         :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
#         :param mean: batched means of Gaussian densities
#         :param log_var: batches means of log_vars
#         :return:
#         """
#         import math
#         pi = torch.tensor(math.pi, requires_grad=False)
#         normalization = torch.log(2. * pi)
#         inv_sigma = torch.exp(-log_var)
#         tmp = samples - mean
#         return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
#
#
# def reparametrize(mu, logvar):
#     std = logvar.mul(0.5).exp_()
#     eps = std.data.new(std.size()).normal_()
#     return eps.mul(std).add_(mu)
#
#
# def info_vae_loss(z_p, z_r):
#     # z: Batch * Sample * Lat Dim
#     # print(z.shape, z_true.shape)
#     z_p_kernel = compute_kernel(z_p, z_p)
#     z_r_kernel = compute_kernel(z_r, z_r)
#     cross_kernel = compute_kernel(z_r, z_p)
#     # mmd = torch.mean(z_true_kernel) + torch.mean(z_kernel) - 2 * torch.mean(z_true_z_kernel)
#     return torch.mean(z_p_kernel) + torch.mean(z_r_kernel) - 2 * torch.mean(cross_kernel)
#
#
# def compute_kernel(x, y):
#     x_size = x.shape[0]
#     y_size = y.shape[0]
#     dim = x.shape[1]
#     tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
#     tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
#     return torch.exp(-torch.mean((tiled_x - tiled_y).float()**2,dim=2)/dim)
#
#
# # def factor_vae_loss():
#
# def beta_tcvae_loss(qz, qz_mean, qz_logvar, pz, pz_mean, pz_logvar, w_mi=1.0, w_tc=1.0, w_dw_kl=1.0):
#     """Estimate of total correlation on a batch.
#     Borrowed from https://github.com/google-research/disentanglement_lib/
#     Args:
#       z: [batch_size, num_latents]-tensor with sampled representation.
#       z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
#       z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
#     Returns:
#       Total correlation estimated on a batch.
#     """
#
#     log_qzCx = gaussian_log_density(qz, qz_mean, qz_logvar).sum(dim=1, keepdim=False)
#     zeros = torch.zeros_like(qz)
#     log_qz_prob = gaussian_log_density(qz.unsqueeze(dim=1),
#                                        qz_mean.unsqueeze(dim=0),
#                                        qz_logvar.unsqueeze(dim=0))
#
#     log_qz_prod = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)
#     log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()
#
#     log_pz_prob = gaussian_log_density(pz.unsqueeze(dim=1),
#                                        pz_mean.unsqueeze(dim=0),
#                                        pz_logvar.unsqueeze(dim=0))
#     log_pz = log_pz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()
#
#     # index-code mutual information loss: I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
#     mi_loss = log_qzCx - log_qz
#     # total correlation loss: TC[z] = KL[q(z)||\prod_i z_i]
#     tc_loss = log_qz - log_qz_prod
#     # dimension-wise KL loss: KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
#     dw_kl_loss = log_qz_prod - log_pz
#
#     return (w_mi*mi_loss + w_tc*tc_loss + w_dw_kl*dw_kl_loss).mean()
#
#
# def beta_tcvae_loss_normal(z, z_mean, z_logvar, w_mi=1.0, w_tc=1.0, w_dw_kl=1.0):
#     """Estimate of total correlation on a batch.
#     Borrowed from https://github.com/google-research/disentanglement_lib/
#     Args:
#       z: [batch_size, num_latents]-tensor with sampled representation.
#       z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
#       z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
#     Returns:
#       Total correlation estimated on a batch.
#     """
#
#     log_qzCx = gaussian_log_density(z, z_mean, z_logvar).sum(dim=1, keepdim=False)
#     zeros = torch.zeros_like(z)
#     log_pz = gaussian_log_density(z, zeros, zeros).sum(dim=1, keepdim=False)
#
#     log_qz_prob = gaussian_log_density(z.unsqueeze(dim=1),
#                                        z_mean.unsqueeze(dim=0),
#                                        z_logvar.unsqueeze(dim=0))
#
#     log_qz_prod = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)
#     log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()
#
#     # index-code mutual information loss: I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
#     mi_loss = log_qzCx - log_qz
#     # total correlation loss: TC[z] = KL[q(z)||\prod_i z_i]
#     tc_loss = log_qz - log_qz_prod
#     # dimension-wise KL loss: KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
#     dw_kl_loss = log_qz_prod - log_pz
#
#     return (w_mi*mi_loss + w_tc*tc_loss + w_dw_kl*dw_kl_loss).mean()
#
#
# def gaussian_log_density(samples, mean, log_var):
#     """ Estimate the log density of a Gaussian distribution
#     Borrowed from https://github.com/google-research/disentanglement_lib/
#     :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
#     :param mean: batched means of Gaussian densities
#     :param log_var: batches means of log_vars
#     :return:
#     """
#     import math
#     pi = torch.tensor(math.pi, requires_grad=False)
#     normalization = torch.log(2. * pi)
#     inv_sigma = torch.exp(-log_var)
#     tmp = samples - mean
#     return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
#
# def matrix_gaussian_log_density(samples, mean, log_var):
#     """Calculates log density of a Gaussian for all combination of bacth pairs of
#     `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
#     instead of (batch_size, dim) in the usual log density.
#     Parameters
#     ----------
#     x: torch.Tensor
#         Value at which to compute the density. Shape: (batch_size, dim).
#     mu: torch.Tensor
#         Mean. Shape: (batch_size, dim).
#     logvar: torch.Tensor
#         Log variance. Shape: (batch_size, dim).
#     batch_size: int
#         number of training images in the batch
#     """
#     batch_size, dim = samples.shape
#     x = samples.view(batch_size, 1, dim)
#     mu = mean.view(1, batch_size, dim)
#     logvar = log_var.view(1, batch_size, dim)
#     return gaussian_log_density(x, mu, logvar)
#
#
# def cal_idx_from_pos(pos, min_pos, max_idx, res):
#     pos = torch.floor((pos - min_pos) / res).type(torch.LongTensor).to(pos.device)
#     # pos = torch.where((0<=pos)&(pos<max_idx), pos)
#     valid = ((0 <= pos) & (pos < max_idx)).to(pos.device)
#     return pos, valid
#
#
# def mgp_variance_loss(pred_goals):
#     '''
#     :param pred_goals: B * K * S
#     :return: loss
#     '''
#
#     mgp_var = 1.0 - 1.0/torch.var(pred_goals, dim=1).mean()
#     mgp_var = (mgp_var > 0).float() * mgp_var
#     return mgp_var
#
# def traj_collision_loss(pred_trajs, envs, envs_params):
#     '''
#     :param pred_trajs: (B, K, Seq, S)
#     :param envs:
#     :param envs_params:
#     :return: loss
#     '''
#     # num_traj = 0
#     num_col_traj = 0
#     # if pred_trajs.dim() == 3: # (B, Seq, S)
#     #     pred_trajs = pred_trajs.unsqueeze(1) # => (B, K, Seq, S)
#     num_traj = float(pred_trajs.size(dim=0) * pred_trajs.size(dim=1))
#     for bidx, btraj in enumerate(pred_trajs):
#         # for kidx, ktraj in enumerate(btraj):
#         x_ids, x_valid = cal_idx_from_pos(btraj[:, 0], envs_params[bidx][0], envs_params[bidx][2], envs_params[bidx][4])
#         y_ids, y_valid = cal_idx_from_pos(btraj[:, 1], envs_params[bidx][1], envs_params[bidx][3], envs_params[bidx][4])
#         valid = (x_valid & y_valid).to(pred_trajs.device)
#         vals = envs[bidx][y_ids[valid], x_ids[valid]]
#         num_col_traj = torch.sum((vals!=envs_params[bidx][5])&(vals!=envs_params[bidx][6])).double()
#         # if torch.any((vals!=envs_params[bidx][5])&(vals!=envs_params[bidx][6])):
#         #     num_col_traj += 1.0
#
#     return torch.div(num_col_traj, num_traj)
#
#
# def goal_collision_loss(pred_goals, envs, envs_params):
#     '''
#     :param pred_goals: (B, K, S)
#     :param envs:
#     :param envs_params:
#     :return: loss
#     '''
#     num_goal = 0
#     num_col_goal = 0
#     for bidx, bgoal in enumerate(pred_goals):
#         num_goal += bgoal.size(dim=0)
#         bgoal = bgoal.reshape(-1, 2)
#         x_ids, x_valid = cal_idx_from_pos(bgoal[:, 0], envs_params[bidx][0], envs_params[bidx][2], envs_params[bidx][4])
#         y_ids, y_valid = cal_idx_from_pos(bgoal[:, 1], envs_params[bidx][1], envs_params[bidx][3], envs_params[bidx][4])
#         valid = (x_valid & y_valid).to(pred_goals.device)
#         vals = envs[bidx][y_ids[valid], x_ids[valid]]
#         num_col_goal += torch.sum(vals > envs_params[bidx][5]).double()
#         # num_col_goal += torch.sum((vals!=envs_params[bidx][5])&(vals!=envs_params[bidx][6])).double()
#     return torch.div(num_col_goal, num_goal)
#
# def pos_collision_loss(pred_trajs, envs, envs_params):
#     '''
#     :param pred_trajs: (B, K, Seq, S)
#     :param envs:
#     :param envs_params:
#     :return: loss
#     '''
#     num_pos = 0
#     num_col_pos = 0
#     # if pred_trajs.
#     for bidx, btraj in enumerate(pred_trajs):
#         btraj = btraj.reshape(-1, 2)
#         x_ids, x_valid = cal_idx_from_pos(btraj[:, 0], envs_params[bidx][0], envs_params[bidx][2], envs_params[bidx][4])
#         y_ids, y_valid = cal_idx_from_pos(btraj[:, 1], envs_params[bidx][1], envs_params[bidx][3], envs_params[bidx][4])
#         valid = (x_valid & y_valid).to(pred_trajs.device)
#         num_pos += torch.sum(valid)
#         vals = envs[bidx][y_ids[valid], x_ids[valid]]
#         num_col_pos += torch.sum(vals > envs_params[bidx][5]).double()
#
#     return torch.div(num_col_pos, num_pos)
#
#
# def single_env_collision_loss(pred_traj, envs, envs_params):
#     '''
#     :param pred_trajs: (B, K, Seq, S)
#     :param map: grid_map
#     :return:
#     '''
#     num_traj = 0
#     num_col = 0
#     # Trajs: (K, Seq, S)
#     # Goals: (K, S)
#     pred_traj = pred_traj.reshape(-1, 2)
#     x_ids, x_valid = cal_idx_from_pos(pred_traj[:, 0], envs_params[0], envs_params[2], envs_params[4])
#     y_ids, y_valid = cal_idx_from_pos(pred_traj[:, 1], envs_params[1], envs_params[3], envs_params[4])
#     valid = (x_valid & y_valid).to(pred_traj.device)
#     num_traj += valid.size(dim=0)
#     vals = envs[y_ids[valid], x_ids[valid]]
#     num_col += torch.sum(vals != envs_params[5]).double()
#
#     return torch.div(num_col, num_traj)
#
#
# def mgp_cvae_loss(pred_goals, gt_goals, k_sample, goal_dim=2, output_dim=2, best_of_many=True):
#     '''
#     CVAE loss use best-of-many
#     Params:
#         pred_goal: (B, K, Sdim)
#         pred_traj: (B, T, Seq, Sdim)
#         pred_goal: (B, K, Sdim)
#         pred_traj: (B, K, Seq, Sdim)
#         gt_goals: (B, Sdim)
#         gt_trajs: (B, Seq, Sdim)
#         best_of_many: whether use best of many loss or not
#     Returns:
#
#     '''
#     gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1) # (B, S) => (B, K, S)
#     # select bom based on  goal_rmse
#     goal_rmse = torch.sqrt(torch.sum((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2, dim=-1)) # (B, K, S) => (B, K)
#
#     # if best_of_many:
#     best_idx = torch.argmin(goal_rmse, dim=-1)
#     # goal_rmse = torch.sqrt(torch.sum((pred_goals[:, :, :goal_dim] - gt_goals[:, :, :goal_dim]) ** 2, dim=-1))
#     loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
#     return loss_goal, best_idx
#     # else:
#     #     loss_goal = goal_rmse.mean()
#     #     return loss_goal
#
# def tgp_cvae_loss(best_goal_idx, pred_goals, pred_trajs, gt_goals, gt_trajs, k_sample, goal_dim=2, output_dim=2, best_of_many=True):
#     gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1) # (B, T, S) => (B, K, T, S)
#
#     fde_rmse = torch.sqrt(torch.sum((pred_trajs[:, :, -1, :output_dim] - pred_goals[:, :, :output_dim]) ** 2, dim=-1))
#     traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1)).sum(dim=-1)
#
#     loss_fde_traj = fde_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#     loss_ade_traj = traj_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#
#     return loss_ade_traj, loss_fde_traj
#
#
# def mgp_tgp_fde_loss(pred_goals, pred_trajs):
#     fde_loss = torch.sqrt(torch.sum((pred_trajs[:, -1, :] - pred_goals) ** 2, dim=-1)).mean()
#     return fde_loss
#
#
# def cvae_loss(pred_goals, pred_trajs, gt_goals, gt_trajs, k_sample, goal_dim=2, output_dim=2, best_of_many=True):
#     '''
#     CVAE loss use best-of-many
#     Params:
#         pred_goal: (B, K, Sdim)
#         pred_traj: (B, K, Seq, Sdim)
#         gt_goals: (B, Sdim)
#         gt_trajs: (B, Seq, Sdim)
#         best_of_many: whether use best of many loss or not
#     Returns:
#
#     '''
#     # K = pred_goal.shape[1]
#     gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1) # (B, T, S) => (B, K, T, S)
#     gt_goals = gt_goals.unsqueeze(1).repeat(1, k_sample, 1) # (B, S) => (B, K, S)
#
#     # select bom based on  goal_rmse
#     fde_rmse = torch.sqrt(torch.sum((pred_trajs[:, :, -1, :output_dim] - pred_goals[:, :, :output_dim]) ** 2, dim=-1))
#     goal_rmse = torch.sqrt(torch.sum((pred_goals[:, :, :output_dim] - gt_goals[:, :, :output_dim]) ** 2, dim=-1))
#     traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1)).sum(dim=-1)
#     # traj_fde_rmse = traj_rmse[:, :, -1] # (B, K, T) => (B, K)
#     # traj_ade_rmse = traj_rmse.mean(dim=-1) # (B, K, T) => (B, K)
#
#     if best_of_many:
#         # best_traj_idx = torch.argmin(traj_ade_rmse, dim=-1)
#         best_goal_idx = torch.argmin(goal_rmse, dim=-1)
#         loss_fde_traj = fde_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#         loss_goal = goal_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#         loss_ade_traj = traj_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#         # loss_fde_traj = traj_fde_rmse[range(len(best_goal_idx)), best_goal_idx].mean()
#     else:
#         loss_goal = goal_rmse.mean()
#         loss_ade_traj = traj_rmse.mean()
#         loss_fde_traj = fde_rmse.mean()
#
#     return loss_goal, loss_ade_traj, loss_fde_traj
#
#
# def bom_traj_loss(pred_trajs, gt_trajs):
#     '''
#     pred: (B, K, T, S)
#     target: (B, T, S)
#     '''
#     batch_size, k_sample, _, _ = pred_trajs.shape
#     gt_trajs = gt_trajs.unsqueeze(1).repeat(1, k_sample, 1, 1) # (B, T, S) => (B, K, T, S)
#     traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1)) # (B, K, T)
#     traj_fde_rmse = traj_rmse[:, :, -1] # (B, K, T) => (B, K)
#     traj_ade_mean = traj_rmse.mean(dim=-1) # (B, K, T) => (B, K)
#     best_idx = torch.argmin(traj_ade_mean, dim=-1)
#     loss_ade_traj = traj_ade_mean[range(len(best_idx)), best_idx].mean()
#     loss_fde_traj = traj_fde_rmse[range(len(best_idx)), best_idx].mean()
#
#     return loss_ade_traj, loss_fde_traj
#
#
# def traj_loss(pred_trajs, gt_trajs):
#     '''
#     pred: (B, T, dim)
#     target: (B, T, dim)
#     '''
#     # gt_trajs [pad / msk / val] # msk only evaluation
#     # pred_trajs
#
#     traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1))
#     # traj_rmse = torch.linalg.norm(pred_trajs - gt_trajs, dim=-1) # pytorch 1.10
#     loss_ade_traj = traj_rmse.mean()
#     loss_fde_traj = traj_rmse[:, -1].mean()
#     return loss_ade_traj, loss_fde_traj
#
#
# def attn_traj_loss(pred_trajs, gt_trajs, attn_mask, obs_len=8, pred_len=12):
#     '''
#     :param pred_trajs: (B, T, S)
#     :param gt_trajs: (B, T, S)
#     :param attn_mask: (B, T)
#     :return:
#     '''
#     tgt_attn_mask = attn_mask[:, obs_len+1:obs_len+pred_len+1]
#     tgt_attn_ids = tgt_attn_mask == 1
#     pred_trajs = pred_trajs[tgt_attn_ids, :]
#     gt_trajs = gt_trajs[tgt_attn_ids, :]
#     traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs) ** 2, dim=-1))
#     loss_ade_traj = traj_rmse.mean()
#     return loss_ade_traj
#
#
# def fol_rmse(x_true, x_pred):
#     '''
#     Params:
#         x_pred: (batch, T, pred_dim) or (batch, T, K, pred_dim)
#         x_true: (batch, T, pred_dim) or (batch, T, K, pred_dim)
#     Returns:
#         rmse: scalar, rmse = \sum_{i=1:batch_size}()
#     '''
#
#     L2_diff = torch.sqrt(torch.sum((x_pred - x_true) ** 2, dim=-1))  #
#     L2_diff = torch.sum(L2_diff, dim=-1).mean()
#
#     return L2_diff
#
#
# # sum(ADE) / (pred_length * sum(batch_size))
# def ADError(pred_traj, gt_traj):
#     # batch x seq x pos
#     loss = gt_traj - pred_traj
#     loss = loss**2
#     loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
#     return torch.sum(loss)
#
# # sum(FDE) / sum(batch_size)
# def FDError(pred_final_pos, gt_final_pos):
#     # batch x pos
#     loss = (gt_final_pos - pred_final_pos).squeeze(dim=1)
#     loss = loss**2
#     loss = torch.sqrt(loss.sum(dim=1))
#     return torch.sum(loss)
