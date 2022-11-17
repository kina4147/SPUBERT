import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, ckpt_path='./output', verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ade = None
        self.best_fde = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.ckpt_path = ckpt_path

    def __call__(self, ade, fde, epoch):
        if self.best_ade is None:
            self.best_ade = ade
            self.best_fde = fde
            self.best_epoch = epoch
            if self.verbose:
                print(f'[ES] EarlyStopper Initialization: Best score is ADE:{self.best_ade:.6f}, FDE:{self.best_fde:.6f}, EPOCH:{self.best_epoch}.')
            return True
        elif ade > self.best_ade + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[ES] EarlyStopping counter: {self.counter} out of {self.patience}.')
                print(f'Best Score (ADE:{self.best_ade:.6f}, FDE:{self.best_fde:.6f}, EPOCH:{self.best_epoch})')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.counter = 0
            if self.verbose:
                print(f'[ES] Loss has been decreased ({self.best_ade:.6f} => {ade:.6f}).')
                print(f'Best Score (ADE:{ade:.6f}, FDE:{fde:.6f}, EPOCH:{epoch})')
            self.best_ade = ade
            self.best_fde = fde
            self.best_epoch = epoch
            return True

    def save_model(self, path, model, model_name='ckpt.pth'):
        if self.verbose:
            print('[ES] Saving model...')
        torch.save(model.state_dict(), os.path.join(path, model_name + '.pth'))