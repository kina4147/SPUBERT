
import math
import random
import copy
import numpy as np
from PIL import Image
from SocialBERT.sbertplus.dataset.grid_map_numpy import RectangularGridMap
import matplotlib.pyplot as plt

# pixel coordinate
# position coordinate

def px2pos(homo_inv):
    pass


def pos2px(homo):
    pass


def rad2deg(rad):
    return rad*180/np.pi


def deg2rad(deg):
    return deg*np.pi/180

def to_scene(trajs):
    return trajs.transpose(1, 0, 2)


def to_trajectory(scene):
    return scene.transpose(1, 0, 2)


def shift_xy(xy, center):
    xy = xy + center[np.newaxis, :]
    return xy


def shift_trajs(trajs, center):
    trajs[:, :, 0:2] = trajs[:, :, 0:2] + center[np.newaxis, np.newaxis, 0:2]
    return trajs

# numpy copy
def extract_patch_from_map(src_map, patch_size, init_val=0, binary=None):
    num_width_patch = src_map.width // patch_size
    num_height_patch = src_map.height // patch_size
    patches = []
    attn_mask = np.ones(num_height_patch*num_width_patch)
    pidx = 0
    for i in range(num_height_patch):
        for j in range(num_width_patch):
            patch = src_map.grid_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if np.all(patch == init_val):
                attn_mask[pidx] = 0
            # binary map
            if binary is not None:
                tmp_patch = copy.deepcopy(patch)
                # patch[tmp_path == init_val] = 2
                patch[(tmp_patch != init_val) & (tmp_patch <= binary)] = 1
                patch[tmp_patch > binary] = 2
            patches.append(patch.flatten())
            pidx += 1

    return patches, attn_mask

#
def convert_patches_to_map(patches, patch_size, resol=0.1):
    num_patch, patch_len = patches.shape
    num_side_patch = np.sqrt(num_patch)
    patch_size = np.sqrt(patch_len)
    width = height = int(num_side_patch*patch_size)
    dst_map = RectangularGridMap(width, height, resol, 0, 0)
    for pidx, patch in enumerate(patches):
        px = pidx % num_side_patch
        py = pidx // num_side_patch
        s_ids = np.arange(patch_len)
        sxs = (s_ids % patch_size + px*patch_size).astype(np.int64)
        sys = (s_ids // patch_size + py*patch_size).astype(np.int64)
        dst_map.set_value_from_xy_index(sxs, sys, patch.flatten())
    return dst_map


def expand_map_with_pad(src_map, pad_val, pad_size):
    width = src_map.width+2*pad_size
    height = src_map.height+2*pad_size
    dst_map = RectangularGridMap(width=width, height=height, resolution=src_map.resolution, center_x=src_map.center_x, center_y=src_map.center_y, init_val=pad_val)

    x_ids, y_ids = src_map.get_all_indices()
    vals = src_map.get_value_from_xy_index(x_ids, y_ids)
    dst_map.set_value_from_xy_index(x_ids+pad_size, y_ids+pad_size, vals)
    return dst_map


def extract_map(src_map, range, resol, trans=None, rot=None):
    num_rows = int((2.0 * range) / resol)
    num_cols = int((2.0 * range) / resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    dst_map = RectangularGridMap(num_cols, num_rows, resol, 0, 0)
    xs, ys = dst_map.get_all_positions()
    if np.all(trans is not None) and rot is not None:
        xys = np.stack((xs, ys), axis=-1)
        rxys = rotate_xy(xys, theta=-rot)
        rxys = shift_xy(rxys, center=-trans)
        vals, valid = src_map.get_value_from_xy_pos(rxys[:, 0], rxys[:, 1])
        # rxys = rxys[valid]
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals)
    else:
        xys = np.stack((xs, ys), axis=-1)
        vals, valid = src_map.get_value_from_xy_pos(xys[:, 0], xys[:, 1])
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals)

    return dst_map


def rotate_xy(xy, theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])
    return xy.dot(r.transpose())


def rotate_trajs(trajs, theta, traj_dim=2):
    # ctrajs = copy.copy(trajs)
    ct = math.cos(theta)
    st = math.sin(theta)
    if traj_dim == 4:
        r = np.array([[ct, st, 0, 0], [-st, ct, 0, 0], [0, 0, ct, st], [0, 0, -st, ct]])
        rtrajs = []
        for idx, traj in enumerate(trajs):
            rtrajs.append(traj.dot(r.transpose()))
    else:
        r = np.array([[ct, st], [-st, ct]])
        rtrajs = []
        for idx, traj in enumerate(trajs):
            rtrajs.append(traj.dot(r.transpose()))
    return np.array(rtrajs) # np.einsum('ptc,ci->pti', trajs, r)


def aug_random_flip(trajs, env=None, prob_thres=0.5):
    prob = random.random()
    # if prob < prob_thres/2:
    #     trajs[:, :, 1] = -trajs[:, :, 1]
    #     if env:
    #         env.flip_x()
    # elif prob < prob_thres:
    #     trajs[:, :, 0] = -trajs[:, :, 0]
    #     if env:
    #         env.flip_y()
    if prob > prob_thres:
        trajs[:, :, 0] = -trajs[:, :, 0]
        if env:
            env.flip_y()
    return trajs, env


def aug_random_scale(trajs, view_range=20.0, obs_len=8, min_scale=0.8, max_scale=1.2, prob_thres=0.5):
    rnd = random.random()
    if rnd < prob_thres:
        scale = min_scale + (max_scale-min_scale)*(rnd-prob_thres)/prob_thres
        tgt_traj = copy.deepcopy(trajs[0])
        tgt_traj = tgt_traj * scale
        tgt_obs_traj = tgt_traj[:obs_len]
        dist = np.linalg.norm(tgt_obs_traj, axis=-1)
        dist = dist[~np.isnan(dist)]
        if np.all(dist < view_range):
            trajs = trajs * scale
    return trajs


def aug_random_rotation(trajs, env=None, prob_thres=0.5):
    # theta = np.random.choice(np.arange(0, 360, 15)) * np.pi / 180
    # prob = random.random()
    # if prob < prob_thres:
    #     theta = prob/prob_thres * 2.0*np.pi
    #     if env:
    #         env.rotate(theta)
    #     return rotate_trajs(trajs=trajs, theta=theta), env
    # else:
    #     return trajs, env
    prob = random.random()
    theta = prob * 2.0*np.pi
    if env:
        env.rotate(theta)
    return rotate_trajs(trajs=trajs, theta=theta), env

def neighbor_filtering(trajs, num_nbr, obs_len, pred_len, input_dim=2, view_range=20.0, view_angle=np.pi/3.0, social_range=2.0):
    """
      At the last frame in observation frames
      After Transform, Filtering, only for observation frames,
      For observation frames
    """
    # data in observation time
    nbr_trajs = copy.deepcopy(trajs[1:, :obs_len+pred_len])
    nbr_trajs = nbr_trajs[~np.all(np.isnan(nbr_trajs[:, :obs_len, 0]), axis=1)]
    ###################
    # Neighbor Filtering
    ###################
    if len(nbr_trajs) > 0:
        # Remove position over view_range (After translation to target center) for obervation
        nbr_dist = np.sqrt(np.sum(np.square(nbr_trajs[:, :, 0:2]), axis=2))
        nbr_dist[np.isnan(nbr_dist)] = view_range
        out_range = nbr_dist >= view_range
        nbr_trajs[out_range, :] = np.nan

        # Interactable pedestrians (After alignment to target orientation)
        nbr_curr_pos = copy.deepcopy(nbr_trajs[:, obs_len-1])
        nbr_curr_dist = copy.deepcopy(nbr_dist[:, obs_len-1])

        out_social_traj = nbr_curr_dist > social_range # out of interest
        out_sight_traj_angle = np.abs(np.arctan2(nbr_curr_pos[:, 1], nbr_curr_pos[:, 0]))
        out_sight_traj_angle[np.isnan(out_sight_traj_angle)] = view_angle
        out_sight_traj = out_sight_traj_angle > view_angle/2.0
        out_interest_traj = out_social_traj & out_sight_traj
        nbr_curr_pos[out_interest_traj, :] = np.nan
        valid_nbr_trajs = np.array([not np.isnan(pos[0]) for pos in nbr_curr_pos])

        nbr_trajs_in = nbr_trajs[valid_nbr_trajs]
        # nbr_trajs_out = nbr_trajs[~valid_nbr_trajs]

        # Neighbor Importance Filtering
        ## Nearest
        nbr_curr_dist_in = nbr_curr_dist[valid_nbr_trajs]
        # nbr_curr_dist_out = nbr_curr_dist[~valid_nbr_trajs]
        nearest_nbr_in = np.argsort(nbr_curr_dist_in)
        # nearest_nbr_out = np.argsort(nbr_curr_dist_out)

        ## Interactable
        # in_interact = get_interactability(trajs[0, :obs_len], nbr_trajs_in, obs_len)
        # nearest_nbr_in = np.argsort(in_interact)[::-1]
        # out_interact = get_interactability(trajs[0, :obs_len], nbr_trajs_out, obs_len)
        # nearest_nbr_out = np.argsort(out_interact)

        # Maximum number of neighbor
        nbr_trajs_in = nbr_trajs_in[nearest_nbr_in]
        # nbr_trajs_out = nbr_trajs_out[nearest_nbr_out]
        # nbr_trajs = np.concatenate([nbr_trajs_in, nbr_trajs_out])
        nbr_trajs = nbr_trajs_in
        num_nbr = min(num_nbr, len(nbr_trajs))
        nbr_trajs = nbr_trajs[:num_nbr, :, :]

        # Random ordering
        rand_order = np.arange(len(nbr_trajs))
        np.random.shuffle(rand_order)
        nbr_trajs = nbr_trajs[rand_order]
        #####################

        out_trajs = np.full((num_nbr + 1, obs_len + pred_len, input_dim), np.nan)
        out_trajs[0] = trajs[0]
        out_trajs[1:, :obs_len+pred_len] = nbr_trajs
        # all nan removal
        out_trajs = out_trajs[~np.all(np.isnan(out_trajs[:, :obs_len, 0]), axis=1)]
        # if out_trajs.ndim == 2:
        #     out_trajs = np.expand_dims(out_trajs, axis=0)
    else:
        # out_trajs = np.full((num_nbr + 1, obs_len + pred_len, input_dim), np.nan)
        out_trajs = np.expand_dims(trajs[0], axis=0)

    return out_trajs

def neighbor_trajs_filtering(obs_trajs, tgt_pred_traj, num_nbr, obs_st_trajs=None, obs_len=8, pred_len=12, input_dim=6, view_range=20.0, view_angle=np.pi/3.0, social_range=2.0):
    """
      At the last frame in observation frames
      After Transform, Filtering, only for observation frames,
      For observation frames
    """
    # trajs = copy.deepcopy(obs_trajs[:, :, 0:2])
    center = -obs_trajs[0, obs_len-1, 0:2] ## Last Observation
    obs_trajs = shift_trajs(obs_trajs, center)
    # print(obs_trajs)
    tgt_pred_traj[:, 0:2] = tgt_pred_traj[:, 0:2] + center[np.newaxis, 0:2]
    nbr_obs_trajs = obs_trajs[1:]

    ## Rotate
    trajs = copy.deepcopy(obs_trajs[:, :, 0:2])
    last1_obs = trajs[0, -1]
    last2_obs = trajs[0, -2]
    diff = np.array([last1_obs[0] - last2_obs[0], last1_obs[1] - last2_obs[1]])
    theta = np.arctan2(diff[1], diff[0])
    trajs = rotate_trajs(trajs, theta)

    # fig3, ax3 = plt.subplots()
    # nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown']
    # for idx, traj in enumerate(trajs):
    #     ax3.scatter(traj[:obs_len-1, 0], traj[:obs_len-1, 1], c=nbr_colors[idx])
    #     ax3.scatter(traj[obs_len-1, 0], traj[obs_len-1, 1], c=nbr_colors[idx], edgecolor="k")
    # ax3.add_patch(mpl.patches.Wedge((0, 0), r=view_range, theta1=-view_angle * 90 / np.pi, theta2=view_angle * 90 / np.pi,
    #                       alpha=0.5, color='k', fill=False, linestyle='--', zorder=0))
    # ax3.add_patch(mpl.patches.Circle((0, 0), radius=social_range, alpha=0.5, color='k', fill=False, linestyle='--',
    #                                 zorder=0))
    # ax3.set_xlim([-view_range/3, view_range/3])
    # ax3.set_ylim([-view_range/3, view_range/3])
    # plt.show()
    ###################
    # Neighbor Filtering
    ###################
    # data in observation time
    nbr_trajs = trajs[1:]
    if len(nbr_trajs) > 0:
        # Remove position over view_range (After translation to target center) for obervation
        nbr_dist = np.sqrt(np.sum(np.square(nbr_trajs[:, :, 0:2]), axis=2))
        nbr_dist[np.isnan(nbr_dist)] = view_range
        out_range = nbr_dist >= view_range
        nbr_trajs[out_range, :] = np.nan
        nbr_obs_trajs[out_range, :] = np.nan

        # Interactable pedestrians (After alignment to target orientation)
        nbr_curr_pos = nbr_trajs[:, obs_len-1]
        nbr_curr_dist = nbr_dist[:, obs_len-1]
        out_social_traj = nbr_curr_dist > social_range # out of interest
        out_sight_traj_angle = np.abs(np.arctan2(nbr_curr_pos[:, 1], nbr_curr_pos[:, 0]))
        out_sight_traj_angle[np.isnan(out_sight_traj_angle)] = view_angle
        out_sight_traj = out_sight_traj_angle > view_angle/2.0
        out_interest_traj = out_social_traj & out_sight_traj
        nbr_curr_pos[out_interest_traj, :] = np.nan
        valid_nbr_trajs = np.array([not np.isnan(pos[0]) for pos in nbr_curr_pos])
        # arrangement

        nbr_obs_trajs_in = nbr_obs_trajs[valid_nbr_trajs]
        nbr_curr_dist_in = nbr_curr_dist[valid_nbr_trajs]
        nbr_obs_trajs_out = nbr_obs_trajs[~valid_nbr_trajs]
        nbr_curr_dist_out = nbr_curr_dist[~valid_nbr_trajs]

        # Maximum number of neighbor
        nearest_nbr_in = np.argsort(nbr_curr_dist_in)
        nearest_nbr_out = np.argsort(nbr_curr_dist_out)
        nbr_obs_trajs_in = nbr_obs_trajs_in[nearest_nbr_in]
        nbr_obs_trajs_out = nbr_obs_trajs_out[nearest_nbr_out]
        nbr_obs_trajs = np.concatenate([nbr_obs_trajs_in, nbr_obs_trajs_out])

        # print(nbr_obs_trajs.shape)
        num_nbr = min(num_nbr, len(nbr_obs_trajs))
        nbr_obs_trajs = nbr_obs_trajs[:num_nbr, :, :]

        # Random ordering
        rand_order = np.arange(len(nbr_obs_trajs))
        np.random.shuffle(rand_order)
        nbr_obs_trajs = nbr_obs_trajs[rand_order]
        ##################

        out_trajs = np.full((num_nbr+1, obs_len, input_dim), np.nan)
        if obs_st_trajs is not None:
            nbr_obs_st_trajs = obs_st_trajs[1:]
            nbr_obs_st_trajs[out_range, :] = np.nan
            nbr_obs_st_trajs_in = nbr_obs_st_trajs[valid_nbr_trajs]
            nbr_obs_st_trajs_out = nbr_obs_st_trajs[~valid_nbr_trajs]
            nbr_obs_st_trajs_in = nbr_obs_st_trajs_in[nearest_nbr_in]
            nbr_obs_st_trajs_out = nbr_obs_st_trajs_out[nearest_nbr_out]
            # if len(nbr_obs_st_trajs_out) > 0:
            nbr_obs_st_trajs = np.concatenate([nbr_obs_st_trajs_in, nbr_obs_st_trajs_out])
            # else:
            #     nbr_obs_st_trajs = nbr_obs_st_trajs_in
            nbr_obs_st_trajs = nbr_obs_st_trajs[:num_nbr, :, :]
            nbr_obs_st_trajs = nbr_obs_st_trajs[rand_order]
            out_trajs[0] = obs_st_trajs[0]
            out_trajs[1:, :, :] = nbr_obs_st_trajs
        else:
            out_trajs[0] = obs_trajs[0]
            out_trajs[1:, :, :] = nbr_obs_trajs
    else:
        out_trajs = np.full((num_nbr+1, obs_len, input_dim), np.nan)
        if obs_st_trajs is not None:
            out_trajs[0] = obs_st_trajs[0]
        else:
            out_trajs[0] = obs_trajs[0]

    return out_trajs, tgt_pred_traj


def is_target_outbound(tgt_traj, obs_len=8, traj_bound=20.0):
    tgt_obs_traj = tgt_traj[:obs_len]
    center_pos = tgt_obs_traj[obs_len-1]
    tgt_obs_traj = tgt_obs_traj - center_pos[np.newaxis, :]
    # dist = np.sqrt(np.sum(np.square(tgt_obs_traj), axis=1))
    dist = np.linalg.norm(tgt_obs_traj, axis=-1)
    dist = dist[~np.isnan(dist)]
    return np.any(dist >= traj_bound)


def translation_to_target(obs_trajs):
    center = -obs_trajs[0, -1, 0:2] ## Last Observation
    nbr_obs_trajs = shift_trajs(obs_trajs, center)
    return obs_trajs


def transform_to_target(trajs, obs_len=8, traj_dim=2):
    ## Center
    center = -trajs[0, obs_len-1, 0:2] ## Last Observation
    trajs = shift_trajs(trajs, center)
    if traj_dim > 2:
        theta = np.arctan2(trajs[0, obs_len-1, 1], trajs[0, obs_len-1, 0])
        trajs = rotate_trajs(trajs, theta, traj_dim)
    else:
        if np.any(np.isnan(trajs[0, obs_len-2:obs_len])):
            theta = 0
        else:
            ## Rotate
            last1_obs = trajs[0, obs_len-1]
            last2_obs = trajs[0, obs_len-2]
            diff = np.array([last1_obs[0] - last2_obs[0], last1_obs[1] - last2_obs[1]])
            theta = np.arctan2(diff[1], diff[0])
            # rotation = -theta#-thet + np.pi/2
            trajs = rotate_trajs(trajs, theta)
    return trajs, center, theta

def is_near(trajs, radius=6.0):
    """
    derive label for near neighbors closer than r meters from target even at least once
    """
    if len(trajs) == 1:
        return [True]
    else:
        dist = np.sum(np.square(trajs - trajs[0, :]), axis=2)
        return np.nanmin(dist, axis=1) < radius ** 2

def get_interactability(tgt_traj, nbr_trajs, obs_len=8):
    """
    :param trajs:
    :return:
    1) calculate positions and velocities:
      - Neighbors should have the last two positions (To, To-1)

    2) calculate interact
    3) calculate
    """
    # dt = 0.4
    tgt_v = (tgt_traj[obs_len-1, :] - tgt_traj[obs_len-2, :]) # / dt # (N, 2)
    nbr_v = (nbr_trajs[:, obs_len-1, :] - nbr_trajs[:, obs_len-2, :]) # / dt # (N, 2)
    rel_v = tgt_v - nbr_v
    rel_p = nbr_trajs[:, obs_len-1, :] - tgt_traj[obs_len-1, :]# (N, 2)
    # attractive_force = np.sqrt(np.sum(rel_v ** 2, axis=-1))/np.sqrt(np.sum(rel_p ** 2, axis=-1))
    prod = np.einsum('ij,ij->i', rel_v, rel_p)
    score = prod/(np.sqrt(np.sum(rel_v ** 2, axis=-1)+1e-16)*np.sqrt(np.sum(rel_p ** 2, axis=-1)+1e-16))
    return score









