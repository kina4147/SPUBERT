import math
import copy
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from SocialBERT.sbertplus.dataset.grid_map_numpy import RectangularGridMap

# index removal


def separate_trajs(trajs, obs_len=8, pred_len=12, num_nbr=4, pad_val=-20):
    sep_trajs = []
    sep_trajs.append(trajs[1:obs_len+pred_len+1, :])
    nbr_trajs = trajs[1+obs_len+pred_len:]
    for nbr_idx in range(num_nbr):
        nbr_traj = nbr_trajs[1+(obs_len+1)*nbr_idx:(obs_len+1)*(nbr_idx+1)]
        nbr_traj[nbr_traj == pad_val] = np.nan
        if not (nbr_traj == np.nan).all():
            sep_trajs.append(nbr_traj)

    return sep_trajs

def viz_input_trajectory(input_trajs, obs_len=8, pred_len=12, num_nbr=4, view_range=20.0, view_angle=np.pi/3, social_range=2.0, pad_val=-20, ax=None,
              nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown'],
              zorder=0, viz_interest_area=False, arrow=False):
    trajs = copy.deepcopy(input_trajs)
    if ax is None:
        fig, ax = plt.subplots()

    sep_trajs = separate_trajs(trajs, obs_len, pred_len, num_nbr, pad_val)
    tgt_traj = sep_trajs[0]
    ax.plot(tgt_traj[:obs_len, 0], tgt_traj[:obs_len, 1], c='red', zorder=zorder)
    ax.scatter(tgt_traj[:obs_len-1, 0], tgt_traj[:obs_len-1, 1], c='red', edgecolors='k', zorder=zorder+2)
    ax.scatter(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], c='red', edgecolors='k', marker='o', zorder=zorder+2)
    dx = tgt_traj[obs_len - 1, 0] - tgt_traj[obs_len - 2, 0]
    dy = tgt_traj[obs_len - 1, 1] - tgt_traj[obs_len - 2, 1]
    dist = math.sqrt(dx * dx + dy * dy)
    dist_scale = 0.4
    arrow_width = 0.1
    dx = dist_scale * dx / dist
    dy = dist_scale * dy / dist
    if arrow:
        ax.arrow(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], dx, dy, color='k',
                 width=arrow_width, head_width=dist_scale, head_length=dist_scale, zorder=zorder)

    if viz_interest_area:
        ax.add_patch(mpl.patches.Wedge((0, 0), r=view_range, theta1=-view_angle*90/np.pi, theta2=view_angle*90/np.pi, alpha=0.5, color='k', fill=False, linestyle='--', zorder=zorder))
        ax.add_patch(mpl.patches.Circle((0, 0), radius=social_range, alpha=0.5, color='k', fill=False, linestyle='--', zorder=zorder))

    for idx, traj in enumerate(sep_trajs[1:]):
        ax.plot(traj[:obs_len, 0], traj[:obs_len, 1], c=nbr_colors[idx], zorder=zorder)
        ax.scatter(traj[:obs_len-1, 0], traj[:obs_len-1, 1], c=nbr_colors[idx], edgecolors='k', zorder=zorder+1)
        ax.scatter(traj[obs_len-1, 0], traj[obs_len-1, 1], c=nbr_colors[idx], edgecolors='k', marker='o', zorder=zorder+1)
        dx = traj[obs_len - 1, 0] - traj[obs_len - 2, 0]
        dy = traj[obs_len-1, 1]-traj[obs_len-2, 1]
        dist = math.sqrt(dx*dx + dy*dy)
        dx = dist_scale * dx / dist
        dy = dist_scale * dy / dist
        if arrow:
            ax.arrow(traj[obs_len-1, 0], traj[obs_len-1, 1], dx, dy, color='k',
                 width=arrow_width, head_width=dist_scale, head_length=dist_scale, zorder=zorder)

def viz_gt_trajs(input_traj, alpha=0.2, dot=False, color="yellow", ax=None, zorder=0):
    traj = copy.deepcopy(input_traj)
    if ax is None:
        fig, ax = plt.subplots()
    if dot:
        ax.scatter(traj[:, 0], traj[:, 1], alpha=alpha, c='white', edgecolors=color, marker='o', linewidth=1.5, zorder=zorder+1)
        ax.plot(traj[:, 0], traj[:, 1], c=color, alpha=alpha, zorder=zorder)
    else:
        ax.plot(traj[:, 0], traj[:, 1], c=color, alpha=alpha, zorder=zorder)


def viz_pred_trajs(input_traj, alpha=0.2, dot=False, color="red", linewidth=None, ax=None, zorder=0):
    traj = copy.deepcopy(input_traj)
    if ax is None:
        fig, ax = plt.subplots()
    if dot:
        ax.scatter(traj[:, 0], traj[:, 1], c=color, alpha=alpha, edgecolors='k', zorder=zorder+1)
        ax.plot(traj[:, 0], traj[:, 1], c=color, alpha=alpha, zorder=zorder)
    # ax.scatter(traj[:, 0], traj[:, 1], c='yellow', alpha=alpha, edgecolors='k', zorder=zorder)
    else:
        ax.plot(traj[:, 0], traj[:, 1], alpha=alpha, c='k', linewidth=linewidth, zorder=zorder)


def viz_k_pred_trajs(input_trajs, alpha=0.2, color="blue", dot=False, ax=None, zorder=0):
    trajs = copy.deepcopy(input_trajs)
    if ax is None:
        fig, ax = plt.subplots()
    for traj in trajs:
        if dot:
            ax.scatter(traj[:, 0], traj[:, 1], c=color, edgecolors='k', alpha=alpha, zorder=zorder+1)
        else:
            ax.plot(traj[:, 0], traj[:, 1], c=color, alpha=alpha, zorder=zorder)


def viz_goal_samples(input_samples, cntr_x=0, cntr_y=0, range=20.0, resol=1.0, alpha=0.5, ax=None, zorder=0):
    samples = copy.deepcopy(input_samples)
    num_rows = int(2.0 * range / resol)
    num_cols = int(2.0 * range / resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    grid_map = RectangularGridMap(width=num_cols, height=num_rows, resolution=resol, center_x=cntr_x, center_y=cntr_y)
    grid_map.update_val_from_xy_pos(samples[:, 0], samples[:, 1], 0.1)
    grid_map.plot_grid_map_in_space(zorder=zorder, alpha=alpha, ax=ax)


def viz_gt_goal(input_goal, ax=None, color='yellow', scale=None, zorder=0):
    goal = copy.deepcopy(input_goal)
    if ax is None:
        fig, ax = plt.subplots()
    if scale is None:
        ax.scatter(goal[0], goal[1], c=color, marker='X', edgecolor='k', zorder=zorder)
    else:
        s = scale * mpl.rcParams['lines.markersize'] ** 2
        ax.scatter(goal[0], goal[1], c=color, marker='X', s=s, edgecolor='k', zorder=zorder)

def viz_pred_goal(input_goals, alpha=0.5, color='blue', scale=None, ax=None, zorder=0):
    goals = copy.deepcopy(input_goals)
    if ax is None:
        fig, ax = plt.subplots()
    if scale is None:
        ax.scatter(goals[0], goals[1], c=color, edgecolor='k', marker='X', alpha=alpha, zorder=zorder)
    else:
        s = scale * mpl.rcParams['lines.markersize'] ** 2
        ax.scatter(goals[0], goals[1], c=color, marker='X', s=s, edgecolor='k', alpha=alpha, zorder=zorder)


def viz_k_pred_goals(input_goals, alpha=0.5, color='blue', ax=None, zorder=0):
    goals = copy.deepcopy(input_goals)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(goals[:, 0], goals[:, 1], c=color, edgecolor='k', marker='X', alpha=alpha, zorder=zorder)



def viz_goal_attention(input_trajs, input_gt_goals, input_pred_goals, input_traj_attentions, self_attn=True, env_seq_len=None, layer_id=0, head_id=0,
                       obs_len=8, pred_len=12, num_nbr=2, pad_val=-20, ofs_x=3, ofs_y=-3, ax=None, zorder=0,
                       nbr_cmaps=['Blues', 'Greens', 'Oranges', 'Reds', 'Purples'],
                       nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown']):
    trajs = copy.deepcopy(input_trajs)
    gt_goals = copy.deepcopy(input_gt_goals)
    pred_goals = copy.deepcopy(input_pred_goals)
    traj_attentions = copy.deepcopy(input_traj_attentions)


    if ax is None:
        fig, ax = plt.subplots()

    attn = traj_attentions[head_id].numpy()
    sep_trajs = separate_trajs(trajs, obs_len, pred_len, num_nbr, pad_val)
    tgt_traj = sep_trajs[0]

    # attention viz
    traj_seq_len = (obs_len + 1) * (num_nbr + 1) + pred_len

    tgt_ids = np.zeros(traj_seq_len+env_seq_len, dtype=bool)
    tgt_ids[1:1+obs_len] = True
    tgt_ids[1+obs_len+pred_len-1] = True

    tgt_goal_ids = np.zeros(traj_seq_len+env_seq_len, dtype=bool)
    tgt_goal_ids[1+obs_len+pred_len-1] = True

    nbr_ids = np.ones(traj_seq_len+env_seq_len, dtype=bool)
    nbr_ids[:1+obs_len+pred_len] = False # Target Trajectory
    nbr_ids[1+obs_len+pred_len::(1+obs_len)] = False # SEP
    nbr_ids[traj_seq_len:] = False # Environment

    max_attn = False

    tgt_traj[obs_len+pred_len-1, :] = pred_goals[:2]
    if max_attn:
        src_tgt_traj = tgt_traj[tgt_ids[1:obs_len+pred_len+1]]
        self_attn = attn[tgt_ids, :]
        self_attn = self_attn[:, tgt_ids]
        self_attn = self_attn.max(axis=0)
        self_attn = self_attn.squeeze()
        self_attn[self_attn < 0.01] = 0.01
        tgt_traj[tgt_traj == pad_val] = np.nan
        # scale = self_attn * 3000
        # ax.scatter(src_tgt_traj[:, 0], src_tgt_traj[:, 1], marker='o', c='red', s=scale)
        darkness = self_attn
        ax.scatter(src_tgt_traj[:, 0], src_tgt_traj[:, 1], marker='o', c=darkness, cmap="Reds", zorder=zorder+1)

        # NBR
        nbr_attn = attn[tgt_ids, :]
        nbr_attn = nbr_attn[:, nbr_ids]
        nbr_attn = nbr_attn.max(axis=0)
        nbr_attn = nbr_attn.squeeze()
        nbr_attn[nbr_attn < 0.01] = 0.01
        for nbr_idx, nbr_traj in enumerate(sep_trajs[1:]):
            nbr_traj[nbr_traj == pad_val] = np.nan
            # scale = nbr_attn[nbr_idx*obs_len:(nbr_idx+1)*obs_len] * 3000
            # ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=nbr_colors[nbr_idx], s=scale)
            darkness = nbr_attn[nbr_idx*obs_len:(nbr_idx+1)*obs_len]
            ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=darkness, cmap=nbr_cmaps[nbr_idx], zorder=zorder)
            #nbr_cmaps[nbr_idx], s=scale)
            # ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=nbr_cmaps[nbr_idx], s=scale, alpha)
            # ax.scatter(nbr_traj[obs_len-1, 0], nbr_traj[obs_len-1, 1], marker='o', c=darkness[obs_len-1], cmap=nbr_cmaps[nbr_idx], edgecolor='k')#, s=scale[obs_len-1])

    else:
        if self_attn:
            self_attn = attn[tgt_ids, :]
            self_attn = self_attn[:, tgt_ids]
            self_attn = self_attn.squeeze()
            self_attn[self_attn < 0.01] = 0.01
            tgt_traj[tgt_traj == pad_val] = np.nan
            scale = self_attn * 3000
            # ax.scatter(tgt_traj[:obs_len, 0], tgt_traj[:obs_len, 1], marker='o', c='red', s=scale[:obs_len])
            src_tgt_traj = tgt_traj[tgt_ids[1:obs_len+pred_len+1]]
            dst_tgt_traj = src_tgt_traj + np.array([ofs_x, ofs_y])
            # ax.scatter(src_tgt_traj[obs_len, 0], src_tgt_traj[obs_len, 1], marker='o', c='red')
            for src_tpidx, src_tgt_pos in enumerate(src_tgt_traj):
                for dst_tpidx, dst_tgt_pos in enumerate(dst_tgt_traj):
                    ax.plot([src_tgt_pos[0], dst_tgt_pos[0]], [src_tgt_pos[1], dst_tgt_pos[1]], alpha=self_attn[src_tpidx, dst_tpidx], c='red', zorder=5)
            ax.scatter(dst_tgt_traj[:, 0], dst_tgt_traj[:, 1], edgecolors='grey', c='lightgrey', marker='o', zorder=21)
            # ax.scatter(tgt_traj[obs_len+pred_len-1, 0], tgt_traj[obs_len+pred_len-1, 1], marker='o', c='red', s=scale[obs_len])

        nbr_attn = attn[tgt_ids, :]
        nbr_attn = nbr_attn[:, nbr_ids]
        nbr_attn = nbr_attn.squeeze()
        src_tgt_traj = tgt_traj[tgt_ids[1:1+obs_len+pred_len]]
        ax.scatter(tgt_traj[obs_len+pred_len-1, 0], tgt_traj[obs_len+pred_len-1, 1], marker='o', c='red') # , s=scale[obs_len])
        for src_tpidx, src_tgt_pos in enumerate(src_tgt_traj):
            for nbr_idx, nbr_traj in enumerate(sep_trajs[1:]):
                for nbr_pos_idx, nbr_pos in enumerate(nbr_traj):
                    if nbr_pos[0] == pad_val:
                        continue
                    nbr_attn_idx = nbr_idx*obs_len + nbr_pos_idx
                    ax.plot([src_tgt_pos[0], nbr_pos[0]], [src_tgt_pos[1], nbr_pos[1]], alpha=nbr_attn[src_tpidx, nbr_attn_idx], c=nbr_colors[nbr_idx], zorder=5)


def viz_trajectory_attention(input_trajs, input_gt_trajs, input_pred_trajs, input_traj_attentions, self_attn=True, layer_id=0, head_id=0,
                             obs_len=8, pred_len=12, num_nbr=2, pad_val=-20, ofs_x=3, ofs_y=-3, ax=None, order=0,
                               nbr_cmaps=['Blues', 'Greens', 'Oranges', 'Reds', 'Purples'],
                               nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown']):
    trajs = copy.deepcopy(input_trajs)
    gt_trajs = copy.deepcopy(input_gt_trajs)
    pred_trajs = copy.deepcopy(input_pred_trajs)
    traj_attentions = copy.deepcopy(input_traj_attentions)
    if ax is None:
        fig, ax = plt.subplots()

    attn = traj_attentions[head_id].numpy()
    sep_trajs = separate_trajs(trajs, obs_len, pred_len, num_nbr, pad_val)
    tgt_traj = sep_trajs[0]

    # attention viz
    traj_seq_len = (obs_len + 1) * (num_nbr + 1) + pred_len

    tgt_ids = np.zeros(traj_seq_len, dtype=bool)
    tgt_ids[1:1 + obs_len + pred_len] = True

    nbr_ids = np.ones(traj_seq_len, dtype=bool)
    nbr_ids[:1+obs_len + pred_len] = False  # Target Trajectory
    nbr_ids[1 + obs_len + pred_len::(1 + obs_len)] = False  # SEP
    nbr_ids[traj_seq_len:] = False  # Environment
    tgt_traj[obs_len:obs_len+pred_len] = pred_trajs # gt_trajs
    max_attn = False
    if max_attn:
        src_tgt_traj = tgt_traj[tgt_ids[:obs_len+pred_len]]
        self_attn = attn[tgt_ids, :]
        self_attn = self_attn[:, tgt_ids]
        self_attn = self_attn.max(axis=0)
        self_attn = self_attn.squeeze()
        self_attn[self_attn < 0.01] = 0.01
        tgt_traj[tgt_traj == pad_val] = np.nan
        # scale = self_attn * 3000
        # ax.scatter(src_tgt_traj[:, 0], src_tgt_traj[:, 1], marker='o', c='red', s=scale)
        darkness = self_attn
        ax.scatter(src_tgt_traj[:, 0], src_tgt_traj[:, 1], marker='o', c=darkness, cmap="Reds")

        # NBR
        nbr_attn = attn[tgt_ids, :]
        nbr_attn = nbr_attn[:, nbr_ids]
        nbr_attn = nbr_attn.max(axis=0)
        nbr_attn = nbr_attn.squeeze()
        nbr_attn[nbr_attn < 0.01] = 0.01
        for nbr_idx, nbr_traj in enumerate(sep_trajs[1:]):
            nbr_traj[nbr_traj == pad_val] = np.nan
            # scale = nbr_attn[nbr_idx*obs_len:(nbr_idx+1)*obs_len] * 3000
            # ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=nbr_colors[nbr_idx], s=scale)
            darkness = nbr_attn[nbr_idx * obs_len:(nbr_idx + 1) * obs_len]
            ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=darkness,
                       cmap=nbr_cmaps[nbr_idx])
            # nbr_cmaps[nbr_idx], s=scale)
            # ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=nbr_cmaps[nbr_idx], s=scale, alpha)
            # ax.scatter(nbr_traj[obs_len-1, 0], nbr_traj[obs_len-1, 1], marker='o', c=darkness[obs_len-1], cmap=nbr_cmaps[nbr_idx], edgecolor='k')#, s=scale[obs_len-1])

    else:
        if self_attn:
            self_attn = attn[tgt_ids, :]
            self_attn = self_attn[:, tgt_ids]
            self_attn = self_attn.squeeze()
            self_attn[self_attn < 0.01] = 0.01
            tgt_traj[tgt_traj == pad_val] = np.nan
            scale = self_attn * 3000
            # ax.scatter(tgt_traj[:obs_len, 0], tgt_traj[:obs_len, 1], marker='o', c='red', s=scale[:obs_len])
            # ax.scatter(tgt_traj[:obs_len+pred_len, 0], tgt_traj[:obs_len+pred_len, 1], marker='o',
            #            c='red', s=scale[obs_len])
            src_tgt_traj = tgt_traj[tgt_ids[:obs_len+pred_len]]
            dst_tgt_traj = src_tgt_traj + np.array([ofs_x, ofs_y])
            for src_tpidx, src_tgt_pos in enumerate(src_tgt_traj):
                for dst_tpidx, dst_tgt_pos in enumerate(dst_tgt_traj):
                    ax.plot([src_tgt_pos[0], dst_tgt_pos[0]], [src_tgt_pos[1], dst_tgt_pos[1]],
                            alpha=self_attn[src_tpidx, dst_tpidx], c='red', zorder=5)
            ax.scatter(dst_tgt_traj[:, 0], dst_tgt_traj[:, 1], edgecolors='grey', c='lightgrey', marker='o',
                       zorder=21)
            # ax.scatter(tgt_traj[obs_len+pred_len-1, 0], tgt_traj[obs_len+pred_len-1, 1], marker='o', c='red', s=scale[obs_len])

        nbr_attn = attn[tgt_ids, :]
        nbr_attn = nbr_attn[:, nbr_ids]
        nbr_attn = nbr_attn.squeeze()
        src_tgt_traj = tgt_traj[tgt_ids[:obs_len+pred_len]]
        for src_tpidx, src_tgt_pos in enumerate(src_tgt_traj):
            for nbr_idx, nbr_traj in enumerate(sep_trajs[1:]):
                for nbr_pos_idx, nbr_pos in enumerate(nbr_traj):
                    if nbr_pos[0] == pad_val:
                        continue
                    nbr_attn_idx = nbr_idx * obs_len + nbr_pos_idx
                    ax.plot([src_tgt_pos[0], nbr_pos[0]], [src_tgt_pos[1], nbr_pos[1]],
                            alpha=nbr_attn[src_tpidx, nbr_attn_idx], c=nbr_colors[nbr_idx], zorder=5)

def viz_input_scene(env_patches, obs_len=8, pred_len=12, num_nbr=2, patch_size=32, resol=0.1, alpha=0.5, ax=None, order=0):

    if ax is None:
        fig, ax = plt.subplots()
    env_seq_len, patch_vec_len = env_patches.shape
    num_patch = np.sqrt(env_seq_len)
    patch_size = np.sqrt(patch_vec_len)
    width = height = int(num_patch*patch_size)
    dst_map = RectangularGridMap(width, height, resol, 0, 0)
    viz_attn = np.ones((width, height))
    traj_seq_len = (obs_len + 1) * (num_nbr + 1) + pred_len
    tgt_ids = np.zeros(traj_seq_len+env_seq_len, dtype=bool)
    tgt_ids[1:1+obs_len+pred_len] = True
    for pidx, patch in enumerate(env_patches):
        px = pidx % num_patch
        py = pidx // num_patch
        for i, p in enumerate(patch):
            ix = int(i % patch_size + px*patch_size)
            iy = int(i // patch_size + py*patch_size)
            dst_map.set_value_from_xy_index(ix, iy, p)

    dst_map.plot_grid_map_in_space(alpha=alpha, ax=ax)



def viz_scene_attention(env_patches, env_attentions, layer_id=0, attn_head_id=0, obs_len=8, pred_len=12,
                        num_nbr=2, patch_size=32, resol=0.1, alpha=0.5, ax=None, order=0):
    '''


    :param env_patches:
    :param env_attentions: target pedestrian trajectory and environmental patches
    :param patch_size:
    :param ax:
    :return:
    '''
    if ax is None:
        fig, ax = plt.subplots()
    # print(env_attentions.shape)
    attn = env_attentions[attn_head_id].numpy()
    env_seq_len, patch_vec_len = env_patches.shape
    num_patch = np.sqrt(env_seq_len)
    patch_size = np.sqrt(patch_vec_len)
    width = height = int(num_patch*patch_size)
    dst_map = RectangularGridMap(width, height, resol, 0, 0)
    viz_attn = np.ones((width, height))
    traj_seq_len = (obs_len + 1) * (num_nbr + 1) + pred_len
    tgt_ids = np.zeros(traj_seq_len+env_seq_len, dtype=bool)
    tgt_ids[1:1+obs_len+pred_len] = True
    for pidx, patch in enumerate(env_patches):
        px = pidx % num_patch
        py = pidx // num_patch
        ptc_idx = pidx + traj_seq_len
        # if tgt2env:
        env_attn = attn[tgt_ids, :]
        env_attn = env_attn[:, ptc_idx]
        # else:
        #     env_attn = attn[ptc_idx, :]
        #     env_attn = env_attn[tgt_ids]
        env_attn = max(env_attn)
        for i, p in enumerate(patch):
            ix = int(i % patch_size + px*patch_size)
            iy = int(i // patch_size + py*patch_size)
            viz_attn[iy, ix] = env_attn
            dst_map.set_value_from_xy_index(ix, iy, p)

    # print(alpha.shape, dst_map.width, dst_map.height)
    dst_map.plot_grid_map_in_space(emph=viz_attn, alpha=alpha, ax=ax)


# def viz_goals(goal_mus, goal_stds, tgt_x, tgt_y, view_range, resol):
#     num_rows = 2.0 * view_range / resol
#     num_cols = 2.0 * view_range / resol
#     num_rows = int(num_rows) if (num_rows % 2) == 0 else int(num_rows+1)
#     num_cols = int(num_cols) if (num_cols % 2) == 0 else int(num_cols+1)
#     grid_map = GridMap(width=num_cols, height=num_rows, resolution=resol, center_x=tgt_x, center_y=tgt_y)
#     gxs, gys = grid_map.get_all_positions()
#     for x, y in zip(gxs, gys):
#         # Search minimum distance
#         max_pdf = 0 # float("inf")
#         for mu, std in zip(goal_mus, goal_stds):
#             d = math.hypot(mu[0] - x, mu[1] - y)
#             pdf = (1.0 - norm.cdf(d, 0.0, std))
#             if max_pdf < pdf:
#                 max_pdf = pdf
#         grid_map.set_value_from_xy_pos(x, y, max_pdf)
#
#     grid_map.plot_grid_map()

def viz_goal_on_target(goal_mus, goal_stds, tgt_x, tgt_y, view_range, resol):
    num_rows = 2.0 * view_range / resol
    num_cols = 2.0 * view_range / resol
    num_rows = int(num_rows) if (num_rows % 2) == 0 else int(num_rows+1)
    num_cols = int(num_cols) if (num_cols % 2) == 0 else int(num_cols+1)
    grid_map = GridMap(width=num_cols, height=num_rows, resolution=resol, center_x=tgt_x, center_y=tgt_y)
    gxs, gys = grid_map.get_all_positions()
    for x, y in zip(gxs, gys):
        # Search minimum distance
        max_pdf = 0 # float("inf")
        for mu, std in zip(goal_mus, goal_stds):
            d = math.hypot(mu[0] - x, mu[1] - y)
            pdf = (1.0 - norm.cdf(d, 0.0, std))
            if max_pdf < pdf:
                max_pdf = pdf
        grid_map.set_value_from_xy_pos(x, y, max_pdf)

    grid_map.plot_grid_map()


def viz_trajectory_max_attention(trajs, pred_trajs, traj_attentions, self_attn=True, env_seq_len=None, layer_id=0, head_id=0,
                                 obs_len=8, pred_len=12, num_nbr=2, pad_val=-20, ofs_x=3, ofs_y=-3, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    attn = traj_attentions[head_id].numpy()
    sep_trajs = separate_trajs(trajs, obs_len, pred_len, num_nbr, pad_val)
    tgt_traj = sep_trajs[0]
    nbr_colors = nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown'] # 'magenta', 'yellow', 'purple', 'pink', 'cyan', 'brown']
    nbr_cmaps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
                 # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 # 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    # attention viz
    traj_seq_len = (obs_len + 1) * (num_nbr + 1) + pred_len
    tgt_ids = np.zeros(traj_seq_len+env_seq_len, dtype=bool)
    tgt_ids[1:1+obs_len+pred_len] = True
    nbr_ids = np.ones(traj_seq_len+env_seq_len, dtype=bool)
    nbr_ids[:1+obs_len+pred_len] = False # Target Trajectory
    nbr_ids[1+obs_len+pred_len::(1+obs_len)] = False # SEP
    nbr_ids[traj_seq_len:] = False # Environment
    tgt_traj[obs_len:] = pred_trajs
    if self_attn:
        attn = attn[tgt_ids, :]
        attn = attn[:, tgt_ids]
        max_attn = attn.max(axis=0)
        max_attn[max_attn < 0.01] = 0.01
        tgt_traj[tgt_traj == pad_val] = np.nan
        scale = max_attn * 3000
        ax.scatter(tgt_traj[:, 0], tgt_traj[:, 1], marker='o', c='red', s=scale)
        ax.scatter(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], marker='o', c='red', edgecolor='k', s=scale[obs_len-1])
    else:
        ax.scatter(tgt_traj[:, 0], tgt_traj[:, 1], c='red')
        ax.scatter(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], c='red', marker='o', edgecolors='k')
        attn = attn[tgt_ids, :]
        attn = attn[:, nbr_ids]
        # TGT * NBR
        max_attn = attn.max(axis=0)
        max_attn[max_attn < 0.01] = 0.01
        for nbr_idx, nbr_traj in enumerate(sep_trajs[1:]):
            nbr_traj[nbr_traj == pad_val] = np.nan
            # scale = max_attn[nbr_idx*obs_len:(nbr_idx+1)*obs_len] * 3000
            darkness = max_attn[nbr_idx*obs_len:(nbr_idx+1)*obs_len]
            # ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=nbr_cmaps[nbr_idx], s=scale, alpha)
            ax.scatter(nbr_traj[:obs_len, 0], nbr_traj[:obs_len, 1], marker='o', c=darkness, cmap=nbr_cmaps[nbr_idx])#nbr_cmaps[nbr_idx], s=scale)
            ax.scatter(nbr_traj[obs_len-1, 0], nbr_traj[obs_len-1, 1], marker='o', c=darkness[obs_len-1], cmap=nbr_cmaps[nbr_idx], edgecolor='k')#, s=scale[obs_len-1])


def viz_mask_trajs(trajs, pred_trajs, gt_pred_trajs, obs_len=8, pred_len=12, num_nbr=4, view_range=20.0, view_angle=np.pi/3,
                   social_range=2.0, pad_val=-20, msk_val=20, ax=None, zorder=0, scale=1):
    sep_trajs = separate_trajs(trajs, obs_len, pred_len, num_nbr, pad_val)
    tgt_traj = sep_trajs[0]
    # ax.scatter(tgt_traj[:obs_len-1, 0], tgt_traj[:obs_len-1, 1], c='red')
    # ax.scatter(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], c='red', marker='o', edgecolors='k')
    # ds = ax.rcParams['lines.markersize'] ** 2
    s = scale * mpl.rcParams['lines.markersize'] ** 2
    tgt_pred_trajs = tgt_traj[obs_len:]
    msk_ids = tgt_pred_trajs[:, 0] == msk_val
    pad_ids = tgt_pred_trajs[:, 0] == pad_val
    no_pad_ids = ~pad_ids
    ax.scatter(gt_pred_trajs[msk_ids, 0], gt_pred_trajs[msk_ids, 1], c='yellow', marker='o', s=s, edgecolors='k', zorder=zorder+1)
    ax.scatter(gt_pred_trajs[pad_ids, 0], gt_pred_trajs[pad_ids, 1], c='grey', marker='o', s=s, edgecolors='k', zorder=zorder)
    ax.scatter(gt_pred_trajs[pred_len-1, 0], gt_pred_trajs[pred_len-1, 1], c='red', marker='o', s=s, zorder=zorder)
    ax.scatter(pred_trajs[no_pad_ids, 0], pred_trajs[no_pad_ids, 1], c='blue', edgecolors='k', zorder=zorder+2)
    # ax.plot(pred_trajs[no_pad_ids, 0], pred_trajs[no_pad_ids, 1], c='k')

    # plt.arrow(tgt_traj[obs_len-1, 0], tgt_traj[obs_len-1, 1], tgt_traj[obs_len-1, 0]-tgt_traj[obs_len-2, 0],
    #           tgt_traj[obs_len-1, 1]-tgt_traj[obs_len-2, 1], color='k')
    # ax.add_patch(mpl.patches.Wedge((0, 0), r=view_range, theta1=-view_angle*180/np.pi, theta2=view_angle*180/np.pi, alpha=0.5, color='k', fill=False, linestyle='--', zorder=1))
    # ax.add_patch(mpl.patches.Circle((0, 0), radius=social_range, alpha=0.5, color='k', fill=False, linestyle='--', zorder=1))
    # for idx, traj in enumerate(sep_trajs[1:]):
    #     plt.scatter(traj[:obs_len-1, 0], traj[:obs_len-1, 1], c='green')
    #     plt.scatter(traj[obs_len-1, 0], traj[obs_len-1, 1], c='green', marker='o', edgecolors='k')


def init_viz(title=None):
    if title:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(title)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots()
    return ax


def viz_save(x_min=None, x_max=None, y_min=None, y_max=None, axs=None, desc="", path="./output"):
    if axs:
        for ax in axs:
            ax.axis("equal")
            if x_min:
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
            ax.axis("off")
            title = ax.get_title()
            ax.set_title("")
            ax.figure.savefig(path + "/" + title + "_" + desc + ".pdf", bbox_inches='tight', format='pdf')
            plt.close(ax.figure)
    else:
        if x_min:
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
        plt.axis('off')
        plt.savefig(path + "/" + title + "_" + desc + ".pdf", bbox_inches='tight', format='pdf')
        plt.close()


def viz_show(x_min=None, x_max=None, y_min=None, y_max=None, axs=None):
    if axs:
        for ax in axs:
            ax.axis("equal")
            if x_min:
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])
            # ax.axis("off")
    else:
        if x_min:
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
        # plt.axis('off')
    plt.show()

def main():
    print("start!!")
    # ox = (np.random.rand(1000) - 0.5) * 20.0
    # oy = (np.random.rand(1000) - 0.5) * 20.0
    # samples = np.stack((ox, oy), axis=-1)

    # n_dist = torch.distributions.Normal(torch.tensor([[4.0, 3, 4], [4.0, 3, 4]]), torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.2, 0.3]]))


    n_dist = torch.distributions.Normal(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 1.0]))
    samples = n_dist.sample(torch.Size([1000]))
    print(samples)
    # print(n_dist.sample_n(3).shape)
    # print(n_dist.sample_n(3))
    #
    # a = torch.zeros(2, 4)
    # b = torch.diag(torch.randn(2, 4))*0.1
    # print(a, b)
    # p_dist = torch.distributions.MultivariateNormal(a, b)
    # print(p_dist.mean, p_dist.variance)
    # goal_latent = p_dist.sample_n(10)
    #
    # print(goal_latent)
    viz_goal_samples(samples, cntr_x=0, cntr_y=0, range=20.0, resol=1)
    # visz_goal_on_target(goal_mus, goal_stds, tgt_x=0, tgt_y=0, view_range=20.0, resol=1)
    print("done!!")
    plt.show()


if __name__ == '__main__':
    main()