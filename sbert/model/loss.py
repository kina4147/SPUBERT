import torch
import torch.nn as nn
import numpy as np

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, mask):
        diff2 = ((torch.flatten(input) - torch.flatten(target)) ** 2.0) * torch.flatten(mask)
        if self.reduction is 'mean':
            result = torch.sum(diff2) / len(input)
        elif self.reduction is 'normal':
            result = torch.sum(diff2) / torch.sum(mask)
        else:
            result = torch.sum(diff2)
        return result


class MaskedADELoss(nn.Module):
    def __init__(self):
        super(MaskedADELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, input, target, mask, weight=None):
        l2_sum = torch.sqrt(torch.sum(((input - target) ** 2.0) * mask + self.eps, dim=2)).sum()
        num_mask = torch.sum(mask)/2 #(torch.sum(mask, dim=2).sum(dim=1).sum() / 2) ** 2
        result = l2_sum / num_mask
        return result

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
        loss = torch.sqrt(torch.sum(((input[:, -1] - target[:, -1]) ** 2.0) + self.eps, dim=1)).mean()
        return loss
    
    
class ADEFDELoss(nn.Module):
    def __init__(self):
        super(ADEFDELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, input, target, mask, weight=None):
        ade = torch.sqrt(torch.sum(((input - target) ** 2.0) * mask + self.eps, dim=2)).sum()
        num_mask = torch.sum(mask)/2
        ade_result = ade / num_mask
        fde = torch.sqrt(torch.sum(((input[:, -1] - target[:, -1]) ** 2.0) + self.eps, dim=1)).sum()
        fde_result = fde / len(input)
        return ade_result + fde_result


class WeightedADELoss(nn.Module):
    def __init__(self):
        super(WeightedADELoss, self).__init__()
        self.eps = 1e-8

    def forward(self, input, target, mask, weight):
        l2_loss = torch.sqrt(torch.sum(((input - target) ** 2.0) * mask + self.eps, dim=2))
        l2_sum = (l2_loss * weight).sum()
        weight_sum = weight.sum() * mask.shape[0]
        # num_mask = torch.sum(mask)/2
        return l2_sum / weight_sum

class CollisionLoss(nn.Module):
    def __init__(self):
        super(CollisionLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, output, tgt_label, tgt_msk, nbr_label, nbr_msk):
        tgt_l2_loss = torch.sqrt(torch.sum(((tgt_label - output) ** 2.0)*tgt_msk + self.eps, dim=2)).sum()
        # batch * nbr * pred_len * dim if nbr has nan value? msk!!
        # nbr_msk batch * nbr * pred_len * dim
        nbr_l2_loss = torch.sqrt(torch.sum(((nbr_label - output) ** 2.0)*nbr_msk + self.eps, dim=3)).sum()
        tgt_l2_loss = tgt_l2_loss / (torch.sum(tgt_msk)/2)
        nbr_l2_loss = nbr_l2_loss / (torch.sum(nbr_msk)/2)
        col_loss = tgt_l2_loss / nbr_l2_loss
        return col_loss

# class ADELoss(nn.Module):
#     def __init__(self):
#         super(ADELoss, self).__init__()
#
#     def forward(self, input, target, mask):
#         l2_sum = torch.sum(((input - target) ** 2.0) * mask, dim=2).sum(dim=1).sum()
#         num_mask = (torch.sum(mask, dim=2).sum(dim=1).sum() / 2) ** 2
#         result = l2_sum / num_mask
#         return result


# sum(ADE) / (pred_length * sum(batch_size))
def ADError(pred_traj, gt_traj):
    # batch x seq x pos
    loss = gt_traj - pred_traj
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    return torch.sum(loss)

# sum(FDE) / sum(batch_size)
def FDError(pred_final_pos, gt_final_pos):
    # batch x pos
    loss = (gt_final_pos - pred_final_pos)
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))
    return torch.sum(loss)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, all_path='checkpoint.pt', bert_path='checkpoint.pt', trace_func=print, parallel=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.all_path = all_path
        self.bert_path = bert_path
        self.trace_func = trace_func
        self.parallel = parallel

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        if self.parallel:
            model.module.save_pretrained(self.all_path)
            model.module.bert.save_pretrained(self.bert_path)
        else:
            model.save_pretrained(self.all_path)
            model.bert.save_pretrained(self.bert_path)
        self.val_loss_min = val_loss

