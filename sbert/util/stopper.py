import numpy as np
import torch
import os


def save_model(path, model, model_name='ckpt.pth', parallel=False):
    # if parallel:
    #     torch.save(model.module.state_dict(), os.path.join(path, model_name + '.pth'))
    # else:
    torch.save(model.state_dict(), os.path.join(path, model_name + '.pth'))

# def load_model(parallel=False):


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, ckpt_path='./output', verbose=False, delta=0, parallel=False):
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
        self.ckpt_path = ckpt_path
        self.parallel = parallel

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return True # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        if self.parallel:
            torch.save(model.module.state_dict(), os.path.join(self.ckpt_path, 'ckpt.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(self.ckpt_path, 'ckpt.pth'))
        # if self.parallel:
        #     model.module.save_pretrained(self.ckpt_path)
        # else:
        #     model.save_pretrained(self.ckpt_path)
        self.val_loss_min = val_loss
