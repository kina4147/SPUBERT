
import numpy as np
import matplotlib.pyplot as plt
from .util import *


def viz_scene_seq(trajs, bg_img):
    for idx, traj in enumerate(trajs):
        plt.imshow(bg_img)
        colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(traj)))
        traj = to_scene(traj)
        for seq_traj in traj:
            plt.scatter(seq_traj[:, 0], seq_traj[:, 1], c=colors,  edgecolors='k')
            plt.pause(0.1)
        plt.clf()
    plt.show()

def viz_trajs(trajs):
    plt.clf()
    for idx, traj in enumerate(trajs):
        plt.scatter(traj[:, :, 0], traj[:, :, 1])
    plt.show()

def viz_scene(trajs, bg_img):
    plt.clf()
    plt.imshow(bg_img)
    for idx, traj in enumerate(trajs):
        plt.scatter(traj[:, :, 0], traj[:, :, 1])
    plt.show()


def viz_sequence(trajs, bg_img):
    plt.clf()
    plt.imshow(bg_img)
    plt.scatter(trajs[:, :, 0], trajs[:, :, 1])
    plt.show()

