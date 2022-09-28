import torch
import numpy as np

class ParamScheduler:
    def __init__(self, name="", start_val=1.0, end_val=100.0, start_epoch=0, last_epoch=100, num_gpu=0):
        self.name = name
        self.epoch = -start_epoch
        self.last_epoch = last_epoch - start_epoch
        self.param = start_val
        self.start_val = start_val
        self.end_val = end_val
        self.num_gpu = num_gpu

    def step(self):
        self.epoch += 1

    def get_param(self):
        if self.last_epoch == 0:
            if self.num_gpu == 0:
                return torch.tensor(self.end_val)
            else:
                return torch.ones([self.num_gpu])*self.end_val
        if self.num_gpu == 0:
            if self.epoch < 0:
                return torch.tensor(self.start_val)
            elif self.epoch > self.last_epoch:
                return torch.tensor(self.end_val)
            else:
                return torch.tensor(self.start_val + (self.end_val - self.start_val) * self.epoch/self.last_epoch)
        else:
            if self.epoch < 0:
                return torch.ones([self.num_gpu])*self.start_val
            elif self.epoch > self.last_epoch:
                return torch.ones([self.num_gpu])*self.end_val
            else:
                return torch.ones([self.num_gpu])*(self.start_val + (self.end_val - self.start_val) * self.epoch/self.last_epoch)

# import matplotlib.pyplot as plt
class CyclicScherduler:
    def __init__(self, name="", start_val=0, end_val=1.0, epoch=100, num_cycle=2, num_gpu=0, ratio=0.5, func=None, monotonic=False, reverse=False):
        self.name = name
        self.epoch = 0
        self.num_gpu = num_gpu

        if num_cycle == 0: # contant
            self.weights = np.full(epoch, end_val)
        # elif num_cycle == 1: # monotonic
        #     self.weights = end_val*func(start_val, 1.0, epoch, num_cycle, ratio)
        #     start_epoch = int(epoch / num_cycle)
        #     self.weights[start_epoch:] = end_val
        else: # cyclical
            self.weights = end_val*func(start_val, 1.0, epoch, num_cycle, ratio)
            if monotonic:
                start_epoch = int(epoch / num_cycle)
                self.weights[start_epoch:] = end_val
        if reverse:
            self.weights = self.weights[::-1]

    def step(self):
        self.epoch += 1

    def get_param(self, train=True):
        if train:
            if self.num_gpu == 0:
                return torch.tensor(self.weights[self.epoch])
            else:
                return torch.ones([self.num_gpu]) * self.weights[self.epoch]
        else:
            if self.num_gpu == 0:
                return torch.tensor(1.0)
            else:
                return torch.ones([self.num_gpu])

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


def frange(start, stop, step, n_epoch):
    L = np.ones(n_epoch)
    v, i = start, 0
    while v <= stop:
        L[i] = v
        v += step
        i += 1
    return L