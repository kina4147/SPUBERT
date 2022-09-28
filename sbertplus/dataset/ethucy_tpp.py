'''
NOTE: May 6
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
from .preprocessing import get_node_timestep_data
import torch
import tqdm
sys.path.append(os.path.realpath('./sbertplus/dataset'))
from torch.utils.data import Dataset
import dill
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
from SocialBERT.sbertplus.dataset.indicator import Spatial6DIndicator, IntegerIndicator, SpatialIndicator
from .util import *
import json

class ETHUCYTPPDataset(SocialBERTDataset):
    def __init__(self, split, args):
        super().__init__(split, args)


        # May 20, updated based on discussiing with Trajectron++
        # June 4th, change hist len to 7 so that the total len is 8
        # hyperparams['minimum_history_length'] = self.args.obs_len - 1 if self.split == 'test' else 1
        conf_json = open('./configs/ethucy_tpp.json', 'r')
        hyperparams = json.load(conf_json)
        # self.min_obs_len = self.args.min_obs_len - 1 if self.split == 'train' else self.args.obs_len - 1
        if self.args.input_dim == 6:
            self.s_ind = Spatial6DIndicator(bound_range=args.view_range)
        elif self.args.input_dim == 2:
            self.s_ind = SpatialIndicator(bound_range=args.view_range)
        else:
            assert False
        # hyperparams = {}
        hyperparams['minimum_history_length'] = self.args.obs_len - 1 if self.split == 'test' else self.args.min_obs_len - 1
        hyperparams['maximum_history_length'] = self.args.obs_len - 1
        # hyperparams['maximum_history_length'] = self.args.obs_len - 1
        # hyperparams['minimum_history_length'] = self.min_obs_len # self.args.obs_len - 1 if split == 'test' else 7 #1 # different from trajectron++, we don't use short histories.
        if self.args.input_dim == 6:
            hyperparams['state'] = {'PEDESTRIAN': {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y']}}
        elif self.args.input_dim == 2:
            hyperparams['state'] = {'PEDESTRIAN': {'position': ['x', 'y']}}#, 'velocity': ['x', 'y'], 'acceleration': ['x', 'y']}}
        else:
            assert False
        hyperparams['pred_state'] = {'PEDESTRIAN': {'position': ['x', 'y']}} #, 'velocity': ['x', 'y'], 'acceleration': ['x', 'y']}}

        if split == 'train':
            f = open(os.path.join(self.path, self.args.dataset_split + '_train.pkl'), 'rb')
            min_history_timesteps = self.args.min_obs_len - 1
            augment = self.args.aug
        elif split == 'val':
            f = open(os.path.join(self.path, self.args.dataset_split + '_val.pkl'), 'rb')
            min_history_timesteps = self.args.obs_len - 1
            augment = False
        elif split == 'test':
            min_history_timesteps = self.args.obs_len - 1
            f = open(os.path.join(self.path, self.args.dataset_split + '_test.pkl'), 'rb')
            augment = False
        else:
            raise ValueError()

        # print("Open dataset file.")
        train_env = dill.load(f, encoding='latin1')
        node_type = train_env.NodeType[0]
        train_env.attention_radius[(node_type, node_type)] = 3.0  # 10.0
        self.dataset = NodeTypeDataset(train_env,
                                       node_type,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       augment=augment,
                                       min_history_timesteps=min_history_timesteps,
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=False)

        # self.dt = 0.4
        # self.outputs = []
        for data in tqdm.tqdm(self.dataset, total=len(self.dataset)):
            first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data, neighbors_data_st, neighbors_lower_upper, _, _, _, _, _, _, mu, std = data # self.dataset.__getitem__(item)
            first_idx = copy.deepcopy(first_history_index)
            tgt_obs_traj = copy.deepcopy(x_t).numpy()
            tgt_pred_traj = copy.deepcopy(y_t).numpy()
            nbr_data = copy.deepcopy(neighbors_data)
            nbr_lower_upper = copy.deepcopy(neighbors_lower_upper)
            tgt_obs_traj = tgt_obs_traj[:, :self.args.input_dim]
            tgt_obs_traj[:first_idx, :] = np.nan
            tgt_traj = np.vstack((tgt_obs_traj, tgt_pred_traj))

            obs_trajs = []
            obs_trajs.append(tgt_traj)
            for nbr_obs, nbr_obs_st, nbr_lower in zip(nbr_data[('PEDESTRIAN', 'PEDESTRIAN')], neighbors_data_st[('PEDESTRIAN', 'PEDESTRIAN')],
                                                    nbr_lower_upper[('PEDESTRIAN', 'PEDESTRIAN')]):

                nbr_obs = nbr_obs[:, :self.args.input_dim]
                nbr_obs[:nbr_lower[0], :] = np.nan
                if torch.all(torch.isnan(nbr_obs)):
                    continue
                nbr_traj = np.full((self.args.obs_len + self.args.pred_len, 2), np.nan)
                nbr_traj[:self.args.obs_len] = nbr_obs
                obs_trajs.append(nbr_traj)
                # obs_st_trajs.append(nbr_obs_st)
            obs_trajs = np.stack(obs_trajs)

            self.all_trajs.append(obs_trajs)
            # if len(self.all_trajs) > 100:
            #     break

        self.all_scenes = [0]*len(self.all_trajs)
        self.scales = [1.0]


    #
    #
    #
    #
    #
    #
    #
    #
    #
    #         # obs_st_trajs = np.stack(obs_st_trajs)
    #         # obs_st_trajs = obs_st_trajs[:, :, :self.args.input_dim]
    #
    #
    #         #################################################
    #         # fig1, ax1 = plt.subplots()
    #         # for idx, traj in enumerate(obs_trajs):
    #         #     ax1.scatter(traj[:, 0], traj[:, 1])
    #         #################################################
    #
    #         #################################################
    #         # fig2, ax2 = plt.subplots()
    #         # for idx, traj in enumerate(obs_trajs):
    #         #     ax2.scatter(traj[:, 0], traj[:, 1])
    #         # ax2.scatter(tgt_pred_traj[:, 0], tgt_pred_traj[:, 1], c='red', edgecolor='k')
    #         # ax2.set_xlim([-self.args.view_range/2, self.args.view_range/2])
    #         # ax2.set_ylim([-self.args.view_range/2, self.args.view_range/2])
    #         #################################################
    #
    #         obs_trajs, tgt_pred_traj = neighbor_trajs_filtering(obs_trajs=obs_trajs, tgt_pred_traj=tgt_pred_traj, # obs_st_trajs=obs_st_trajs,
    #                                                             num_nbr=self.args.num_nbr, obs_len=self.args.obs_len,
    #                                                             pred_len=self.args.pred_len,
    #                                                             view_range=self.args.view_range,
    #                                                             view_angle=self.args.view_angle,
    #                                                             social_range=self.args.social_range,
    #                                                             input_dim=self.args.input_dim)
    #
    #         #################################################
    #         # fig3, ax3 = plt.subplots()
    #         # for idx, traj in enumerate(obs_trajs):
    #         #     ax3.scatter(traj[:, 0], traj[:, 1])
    #         # ax3.scatter(tgt_pred_traj[:, 0], tgt_pred_traj[:, 1], c='red', edgecolor='k')
    #         # ax3.set_xlim([-self.args.view_range/3, self.args.view_range/3])
    #         # ax3.set_ylim([-self.args.view_range/3, self.args.view_range/3])
    #         # plt.show()
    #         #################################################
    #         # for target pedestrian
    #         tgt_obs_traj = copy.deepcopy(obs_trajs[0])
    #         nbr_obs_trajs = copy.deepcopy(obs_trajs[1:])
    #
    #         # vel_t1 = (tgt_pred_st_traj[-1, :] - tgt_pred_st_traj[-2, :]) / self.dt
    #         # vel_t2 = (tgt_pred_st_traj[-2, :] - tgt_pred_st_traj[-3, :]) / self.dt
    #         vel_t1 = (tgt_pred_traj[-1, :] - tgt_pred_traj[-2, :]) / self.dt
    #         vel_t2 = (tgt_pred_traj[-2, :] - tgt_pred_traj[-3, :]) / self.dt
    #         acc_t1 = (vel_t1 - vel_t2) / self.dt
    #         # goal_lbs = np.hstack((tgt_pred_st_traj[-1, :], vel_t1, acc_t1))
    #         goal_lbs = np.hstack((tgt_pred_traj[-1, :], vel_t1, acc_t1))
    #         # goal_lbs = tgt_pred_traj[-1, :self.args.goal_dim]
    #         # goal_ids = tgt_pred_st_traj[:self.args.goal_dim]/std[:self.args.goal_dim]
    #
    #         goal_lbs = goal_lbs[:self.args.input_dim]
    #
    #         traj_lbs = tgt_pred_traj
    #         nan_idx = np.isnan(tgt_obs_traj[:, 0])
    #         tgt_obs_traj[nan_idx] = self.s_ind.pad_id
    #
    #         ## All Prediction Padding
    #         mgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id] * (
    #                     self.args.pred_len)
    #         mgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [
    #             self.i_ind.pad_id] * (self.args.pred_len)
    #         mgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
    #                                                   zip(range(1, self.args.obs_len + 1),
    #                                                       nan_idx[:self.args.obs_len])] + [self.i_ind.pad_id] * (
    #                                        self.args.pred_len)
    #         mgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in
    #                                                  nan_idx[:self.args.obs_len]] + [self.i_ind.false_val] * (
    #                                     self.args.pred_len)
    #
    #         # Last Prediction Masking
    #         # mgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id] * (
    #         #             self.args.pred_len - 1) + [self.s_ind.msk_id]
    #         # mgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [
    #         #     self.i_ind.pad_id] * (self.args.pred_len - 1) + [1]
    #         # mgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
    #         #                                           zip(range(1, self.args.obs_len + 1),
    #         #                                               nan_idx[:self.args.obs_len])] + [self.i_ind.pad_id] * (
    #         #                                self.args.pred_len - 1) + [self.seq_len]
    #         # mgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in
    #         #                                          nan_idx[:self.args.obs_len]] + [self.i_ind.false_val] * (
    #         #                             self.args.pred_len - 1) + [self.i_ind.true_val]
    #
    #         ## All Prediction Masking
    #         # mgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.msk_id] * self.args.pred_len
    #         # mgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * self.args.pred_len
    #         # mgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
    #         #                                           zip(range(1, self.args.obs_len + 1),
    #         #                                               nan_idx[:self.args.obs_len])] + [x for x in np.arange(
    #         #     self.args.obs_len + 1, self.seq_len + 1)]
    #         # mgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in
    #         #                                          nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * (
    #         #                     self.args.pred_len)
    #
    #
    #
    #
    #
    #         tgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.msk_id] * (
    #                     self.args.pred_len - 1) + [goal_lbs.tolist()]
    #         tgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * (
    #             self.args.pred_len)
    #         tgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
    #                                                   zip(range(1, self.args.obs_len + 1),
    #                                                       nan_idx[:self.args.obs_len])] + [x for x in np.arange(
    #             self.args.obs_len + 1, self.seq_len + 1)]
    #         tgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in
    #                                                  nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * (
    #                             self.args.pred_len)
    #
    #         # for neighbor pedestrian
    #         spatial_ids = []
    #         segment_ids = []
    #         temporal_ids = []
    #         attn_mask = []
    #         for nbr_idx, nbr_traj in enumerate(nbr_obs_trajs):
    #             # NAN handling
    #             nbr_nan_idx = np.isnan(nbr_traj[:, 0])
    #             nbr_traj[nbr_nan_idx] = self.s_ind.pad_id
    #             nbr_attn_mask = np.ones(len(nbr_traj))
    #             nbr_attn_mask[nbr_nan_idx] = self.i_ind.false_val
    #             spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
    #             segment_ids += [nbr_idx + 2] + [self.i_ind.pad_id if nan else nbr_idx + 2 for nan in nbr_nan_idx]
    #             temporal_ids += [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
    #                                                    zip(range(1, self.args.obs_len + 1), nbr_nan_idx)]
    #             attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()
    #
    #         # collate
    #         ext_len = self.seq_input_len - len(spatial_ids) - len(mgp_spatial_ids)
    #         spatial_ids_padding = [self.s_ind.pad_id] * ext_len  # [self.s_ind.pad_id for _ in range(ext_len)]
    #         spatial_ids.extend(spatial_ids_padding)
    #         integer_ids_padding = [self.i_ind.pad_id] * ext_len  # [self.i_ind.pad_id for _ in range(ext_len)]
    #         segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(
    #             integer_ids_padding)
    #
    #         mgp_spatial_ids += spatial_ids
    #         mgp_segment_ids += segment_ids
    #         mgp_temporal_ids += temporal_ids
    #         mgp_attn_mask += attn_mask
    #         tgp_spatial_ids += spatial_ids
    #         tgp_segment_ids += segment_ids
    #         tgp_temporal_ids += temporal_ids
    #         tgp_attn_mask += attn_mask
    #         # spatial_ids = torch.tensor(spatial_ids, dtype=torch.float)
    #
    #         goal_lbs = goal_lbs[:self.args.goal_dim]
    #         scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # np.array([self.args.view_range, self.args.view_range])
    #         scale = torch.tensor(scale[:self.args.goal_dim], dtype=torch.float)
    #         output = {'mgp_spatial_ids': torch.tensor(mgp_spatial_ids, dtype=torch.float),
    #                   'mgp_segment_ids': torch.tensor(mgp_segment_ids, dtype=torch.long),
    #                   'mgp_temporal_ids': torch.tensor(mgp_temporal_ids, dtype=torch.long),
    #                   'mgp_attn_mask': torch.tensor(mgp_attn_mask, dtype=torch.float),
    #                   'tgp_spatial_ids': torch.tensor(tgp_spatial_ids, dtype=torch.float),
    #                   'tgp_segment_ids': torch.tensor(tgp_segment_ids, dtype=torch.long),
    #                   'tgp_temporal_ids': torch.tensor(tgp_temporal_ids, dtype=torch.long),
    #                   'tgp_attn_mask': torch.tensor(tgp_attn_mask, dtype=torch.float),
    #                   'traj_lbs': torch.tensor(traj_lbs, dtype=torch.float),
    #                   'goal_lbs': torch.tensor(goal_lbs, dtype=torch.float),
    #                   'scale': scale,
    #                   }
    #
    #         self.outputs.append(output)
    #         # if len(self.outputs) > 100:
    #         #     break
    #         if self.split == 'test' and self.args.viz and len(self.outputs) > 1000:
    #             break
    #
    # def __len__(self):
    #     return len(self.outputs)
    #
    #
    # def __getitem__(self, item):
    #     return self.outputs[item]

    # def __getitem__(self, item):
    #     first_history_index, x_t, y_t, _, _, neighbors_data, _, neighbors_lower_upper, neighbors_future, _, _, _, _, _ = self.dataset.__getitem__(item)
    #     first_idx = copy.deepcopy(first_history_index)
    #     tgt_obs_traj = copy.deepcopy(x_t).numpy()
    #     tgt_pred_traj = copy.deepcopy(y_t).numpy()
    #     nbr_data = copy.deepcopy(neighbors_data)
    #     nbr_future = copy.deepcopy(neighbors_future)
    #     nbr_lower_upper = copy.deepcopy(neighbors_lower_upper)
    #     tgt_obs_traj[:first_idx, :] = np.nan
    #     obs_trajs = []
    #     obs_trajs.append(tgt_obs_traj)
    #     for nbr_obs, nbr_pred, nbr_lower in zip(nbr_data[('PEDESTRIAN', 'PEDESTRIAN')], nbr_future[('PEDESTRIAN', 'PEDESTRIAN')], nbr_lower_upper[('PEDESTRIAN', 'PEDESTRIAN')]):
    #         # print(nbr_obs)
    #         nbr_obs[:nbr_lower[0], :] = np.nan
    #         if torch.all(torch.isnan(nbr_obs)):
    #             continue
    #         obs_trajs.append(nbr_obs)
    #     obs_trajs = np.stack(obs_trajs)
    #     obs_trajs = obs_trajs[:, :, :self.args.input_dim]
    #     #################################################
    #     # fig1, ax1 = plt.subplots()
    #     # for idx, traj in enumerate(obs_trajs):
    #     #     ax1.scatter(traj[:, 0], traj[:, 1])
    #     #################################################
    #
    #     #################################################
    #     # fig2, ax2 = plt.subplots()
    #     # for idx, traj in enumerate(obs_trajs):
    #     #     ax2.scatter(traj[:, 0], traj[:, 1])
    #     # ax2.scatter(tgt_pred_traj[:, 0], tgt_pred_traj[:, 1], c='red', edgecolor='k')
    #     # ax2.set_xlim([-self.args.view_range/2, self.args.view_range/2])
    #     # ax2.set_ylim([-self.args.view_range/2, self.args.view_range/2])
    #     #################################################
    #
    #     obs_trajs, tgt_pred_traj = neighbor_trajs_filtering(obs_trajs=obs_trajs, tgt_pred_traj=tgt_pred_traj, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len, pred_len=self.args.pred_len,
    #                                view_range=self.args.view_range, view_angle=self.args.view_angle,
    #                                social_range=self.args.social_range, input_dim=self.args.input_dim)
    #
    #
    #     #################################################
    #     # fig3, ax3 = plt.subplots()
    #     # for idx, traj in enumerate(obs_trajs):
    #     #     ax3.scatter(traj[:, 0], traj[:, 1])
    #     # ax3.scatter(tgt_pred_traj[:, 0], tgt_pred_traj[:, 1], c='red', edgecolor='k')
    #     # ax3.set_xlim([-self.args.view_range/3, self.args.view_range/3])
    #     # ax3.set_ylim([-self.args.view_range/3, self.args.view_range/3])
    #     # plt.show()
    #     #################################################
    #     # for target pedestrian
    #     # tgt_traj = copy.copy(obs_trajs[0])
    #     tgt_obs_traj = copy.deepcopy(obs_trajs[0])
    #     nbr_obs_trajs = copy.deepcopy(obs_trajs[1:])
    #     vel_t1 = (tgt_pred_traj[-1, :] - tgt_pred_traj[-2, :])/self.dt
    #     vel_t2 = (tgt_pred_traj[-2, :] - tgt_pred_traj[-3, :])/self.dt
    #     acc_t1 = (vel_t1 - vel_t2) / self.dt
    #     goal_lbs = np.hstack((tgt_pred_traj[-1, :], vel_t1, acc_t1))
    #     goal_lbs = goal_lbs[:self.args.goal_dim]
    #     # goal_lbs = tgt_pred_traj[-1, :]
    #     traj_lbs = tgt_pred_traj
    #     nan_idx = np.isnan(tgt_obs_traj[:, 0])
    #     tgt_obs_traj[nan_idx] = self.s_ind.pad_id
    #     spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id]*(self.args.pred_len-1) + [self.s_ind.msk_id]
    #
    #     # print("gt_goal in data: ", goal_lbs)
    #
    #     mgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.pad_id] * (self.args.pred_len-1) + [1]
    #     mgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nan_idx[:self.args.obs_len])] + [self.i_ind.pad_id] * (self.args.pred_len-1) + [self.seq_len]
    #     mgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.false_val] * (self.args.pred_len-1) + [self.i_ind.true_val]
    #
    #     tgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * (self.args.pred_len)
    #     tgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nan_idx[:self.args.obs_len])] + [x for x in np.arange(self.args.obs_len+1, self.seq_len+1)]
    #     tgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * (self.args.pred_len)
    #
    #     # nbr_trajs = copy.copy(trajs[1:, :self.args.obs_len, :])
    #     # for neighbor pedestrian
    #     segment_ids = []
    #     temporal_ids = []
    #     attn_mask = []
    #     for nbr_idx, nbr_traj in enumerate(nbr_obs_trajs):
    #         # NAN handling
    #         nbr_nan_idx = np.isnan(nbr_traj[:, 0])
    #         nbr_traj[nbr_nan_idx] = self.s_ind.pad_id
    #         nbr_attn_mask = np.ones(len(nbr_traj))
    #         nbr_attn_mask[nbr_nan_idx] = self.i_ind.false_val
    #         spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
    #         segment_ids += [nbr_idx+2] + [self.i_ind.pad_id if nan else nbr_idx+2 for nan in nbr_nan_idx]
    #         temporal_ids += [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nbr_nan_idx)]
    #         attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()
    #
    #     # collate
    #     ext_len = self.seq_input_len - len(spatial_ids)
    #     spatial_ids_padding = [self.s_ind.pad_id] * ext_len # [self.s_ind.pad_id for _ in range(ext_len)]
    #     spatial_ids.extend(spatial_ids_padding)
    #     integer_ids_padding = [self.i_ind.pad_id] * ext_len # [self.i_ind.pad_id for _ in range(ext_len)]
    #     segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(integer_ids_padding)
    #     # tgp_segment_ids.extend(integer_ids_padding), tgp_temporal_ids.extend(integer_ids_padding), tgp_attn_mask.extend(integer_ids_padding)
    #
    #     mgp_segment_ids += segment_ids
    #     mgp_temporal_ids += temporal_ids
    #     mgp_attn_mask += attn_mask
    #     tgp_segment_ids += segment_ids
    #     tgp_temporal_ids += temporal_ids
    #     tgp_attn_mask += attn_mask
    #     # print(np.array(spatial_ids).shape)
    #     # print(np.array(mgp_segment_ids).shape, np.array(mgp_temporal_ids).shape, np.array(tgp_segment_ids).shape, np.array(tgp_temporal_ids).shape)
    #     # fig4, ax4 = plt.subplots()
    #     # tgt_env.plot_grid_map_in_space(ax=ax4)
    #     # for idx, traj in enumerate(trajs):
    #     #     ax4.scatter(traj[:, 0], traj[:, 1])
    #     # plt.show()
    #
    #
    #     spatial_ids = torch.tensor(spatial_ids, dtype=torch.float)
    #     if self.args.scale:
    #         spatial_ids = spatial_ids/self.args.view_range
    #     output = {'spatial_ids': spatial_ids,
    #               'mgp_segment_ids': torch.tensor(mgp_segment_ids, dtype=torch.long),
    #               'mgp_temporal_ids': torch.tensor(mgp_temporal_ids, dtype=torch.long),
    #               'mgp_attn_mask': torch.tensor(mgp_attn_mask, dtype=torch.float),
    #               'tgp_segment_ids': torch.tensor(tgp_segment_ids, dtype=torch.long),
    #               'tgp_temporal_ids': torch.tensor(tgp_temporal_ids, dtype=torch.long),
    #               'tgp_attn_mask': torch.tensor(tgp_attn_mask, dtype=torch.float),
    #               'traj_lbs': torch.tensor(traj_lbs, dtype=torch.float),
    #               'goal_lbs': torch.tensor(goal_lbs, dtype=torch.float),
    #               }
    #
    #     return output



class NodeTypeDataset(Dataset):
    '''
    from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus
    '''

    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]


    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()

        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] * \
                             (scene.frequency_multiplier if scene_freq_mult else 1) * \
                             (node.frequency_multiplier if node_freq_mult else 1)
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]
        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)
        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)

    # def __getitem__(self, i):
    #     (scene, t, node) = self.index[i]
    #     if self.augment:
    #         scene_aug = scene.augmented[self.aug_idx]
    #         scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    #         node = scene_aug.get_node_by_id(node.id)
    #     return get_node_timestep_data(self.env, scene_aug, t, node, self.state, self.pred_state,
    #                                   self.edge_types, self.max_ht, self.max_ft, self.hyperparams)

    # def __getitem__(self, item):
    #     # deep copy
    #     trajs = self.all_trajs[item]
    #
    #     # fig1, ax1 = plt.subplots()
    #     # if self.args.scene:
    #     #     env.plot_grid_map_in_space(ax=ax1)
    #     # for idx, traj in enumerate(trajs):
    #     #     ax1.scatter(traj[:, 0], traj[:, 1])
    #
    #     trajs, trans, rot = transform_to_target(trajs=trajs, obs_len=self.args.obs_len)
    #     if self.args.scene:
    #         env = self.envs[self.all_scenes[item]]
    #         tgt_env = extract_map(env, range=self.args.env_range, resol=self.args.env_resol, trans=trans, rot=rot)
    #     else:
    #         tgt_env = None
    #
    #     # fig2, ax2 = plt.subplots()
    #     # tgt_env.plot_grid_map_in_space(ax=ax2)
    #     # for idx, traj in enumerate(trajs):
    #     #     ax2.scatter(traj[:, 0], traj[:, 1])
    #
    #     if self.args.aug:
    #         # trajs = aug_random_scale(trajs)
    #         trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len,
    #                                    view_range=self.args.view_range, view_angle=self.args.view_angle,
    #                                    social_range=self.args.social_range)
    #         trajs, tgt_env = aug_random_flip(trajs, tgt_env)
    #         trajs, tgt_env = aug_random_rotation(trajs, tgt_env)
    #     else:
    #         trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len,
    #                                    view_range=self.args.view_range, view_angle=self.args.view_angle,
    #                                    social_range=self.args.social_range)
    #
    #     # fig3, ax3 = plt.subplots()
    #     # if self.args.scene:
    #     #     tgt_env.plot_grid_map_in_space(ax=ax3)
    #     # for idx, traj in enumerate(trajs):
    #     #     ax3.scatter(traj[:, 0], traj[:, 1])
    #     # plt.show()
    #     # for target pedestrian
    #     tgt_traj = copy.copy(trajs[0])
    #     goal_lbs = tgt_traj[-1, :]
    #     traj_lbs = tgt_traj[self.args.obs_len:]
    #     if self.args.mode == "pretrain":
    #         if self.args.train_mode == "tgp":
    #             # NAN => PAD => ALL PAD => ALL ATTN MASK
    #             spatial_ids = [self.s_ind.sot_id] + tgt_traj[:self.args.obs_len].tolist() + [self.s_ind.msk_id]*(self.args.pred_len-1) + [tgt_traj[self.seq_len-1].tolist()]
    #             segment_ids = [1 for _ in range(self.seq_len + 1)]
    #             temporal_ids = [self.i_ind.pad_id] + [x for x in range(1, self.seq_len + 1)]
    #             attn_mask = [self.i_ind.true_val] * (self.seq_len + 1)
    #         elif self.args.train_mode == "mgp":
    #             spatial_ids = [self.s_ind.sot_id] + tgt_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id] * (
    #                         self.args.pred_len - 1) + [self.s_ind.msk_id]
    #             segment_ids = [1 for _ in range(self.seq_len + 1)]
    #             temporal_ids = [self.i_ind.pad_id] + [x for x in range(1, self.seq_len + 1)]
    #             attn_mask = [self.i_ind.true_val] * (self.args.obs_len + 1) + [self.i_ind.false_val] * (
    #                         self.args.pred_len - 1) + [self.i_ind.true_val]
    #         elif self.args.train_mode == "mtp":
    #             # MTP + SIP + EIP
    #             goal_lbs = copy.copy(tgt_traj[-1, :])
    #             traj_lbs = copy.copy(tgt_traj[self.args.obs_len:])
    #             # PAD : MSK : VAL = PAD (NEG): MSK (EVL): VAL (EVL) = 6 : 5 : 1
    #             pred_ids = np.arange(self.args.obs_len, self.args.obs_len+self.args.pred_len)
    #             np.random.shuffle(pred_ids) # Shuffle
    #             # pads = pred_ids[:6]
    #             # msks = pred_ids[6:11]
    #             # vals = pred_ids[11]
    #             tgt_traj[pred_ids[:6]] = self.s_ind.pad_id
    #             tgt_traj[pred_ids[6:11]] = self.s_ind.msk_id
    #             # rand_mask = np.random.choice(self.args.pred_len, 3, replace=False) + self.args.obs_len
    #             # tgt_traj[rand_mask] = self.s_ind.msk_id
    #             spatial_ids = [self.s_ind.sot_id] + tgt_traj.tolist()
    #             segment_ids = [1 for _ in range(self.seq_len + 1)]
    #             temporal_ids = [self.i_ind.pad_id] + [x for x in range(1, self.seq_len + 1)]
    #             attn_mask = np.array([self.i_ind.true_val] * (self.seq_len + 1))
    #             attn_mask[pred_ids[:6]+1] = self.i_ind.false_val
    #             attn_mask = attn_mask.tolist()
    #         else:
    #             raise ValueError("Pretrain's train mode does not exist.")
    #
    #     elif self.args.mode == "finetune":
    #         nan_idx = np.isnan(tgt_traj[:, 0])
    #         # print(nan_idx)
    #         tgt_traj[nan_idx] = self.s_ind.pad_id
    #         # for target pedestrian
    #         spatial_ids = [self.s_ind.sot_id] + tgt_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id]*(self.args.pred_len-1) + [self.s_ind.msk_id]
    #         segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx]
    #         temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.seq_len+1), nan_idx)]
    #         attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx]
    #         # segment_ids = [1 for _ in range(self.seq_len + 1)]
    #         # temporal_ids = [self.i_ind.pad_id] + [x for x in range(1, self.seq_len + 1)]
    #         # attn_mask = [self.i_ind.true_val] * (self.args.obs_len+1) + [self.i_ind.false_val] * (self.args.pred_len-1) + [self.i_ind.true_val ]# [self.i_ind.true_val] * (self.seq_len+1)
    #         # not full target pedestrian
    #     # print(nan_idx)
    #     # print(spatial_ids)
    #     # print(segment_ids)
    #     # print(temporal_ids)
    #     # print(attn_mask)
    #     # print("============================")
    #     nbr_trajs = copy.copy(trajs[1:, :self.args.obs_len, :])
    #     # for neighbor pedestrian
    #     for nbr_idx, nbr_traj in enumerate(nbr_trajs):
    #         # NAN handling
    #         nan_elem = np.isnan(nbr_traj)
    #         nbr_traj[nan_elem] = self.s_ind.pad_val  # ignore or no input
    #         nbr_attn_mask = np.ones(len(nbr_traj))
    #         nbr_attn_mask[nan_elem[:, 0]] = self.i_ind.false_val
    #
    #         spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
    #         segment_ids += [nbr_idx+2 for _ in range(self.args.obs_len+1)]
    #         temporal_ids += [self.i_ind.pad_id] + [x for x in range(1, self.args.obs_len+1)]
    #         attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()
    #
    #     # collate
    #     spatial_ids_padding = [self.s_ind.pad_id for _ in range(self.seq_input_len - len(spatial_ids))]
    #     spatial_ids.extend(spatial_ids_padding)
    #     integer_ids_padding = [self.i_ind.pad_id for _ in range(self.seq_input_len - len(segment_ids))]
    #     segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(integer_ids_padding)
    #
    #     # fig4, ax4 = plt.subplots()
    #     # tgt_env.plot_grid_map_in_space(ax=ax4)
    #     # for idx, traj in enumerate(trajs):
    #     #     ax4.scatter(traj[:, 0], traj[:, 1])
    #     # env patch (patch cell size) even number
    #     # plt.show()
    #     if self.args.scene:
    #         if tgt_env.width % self.args.patch_size != 0: # add padding
    #             pad_size = self.args.patch_size - (tgt_env.width % self.args.patch_size) // 2
    #             tgt_env = expand_map_with_pad(tgt_env, 2, pad_size)
    #         #
    #         # fig5, ax5 = plt.subplots()
    #         # tgt_env.plot_grid_map_in_space(ax=ax5)
    #         # for idx, traj in enumerate(trajs):
    #         #     ax5.scatter(traj[:, 0], traj[:, 1])
    #         # plt.show()
    #
    #         # to patch
    #         env_spatial_ids = extract_patch_from_map(tgt_env, self.args.patch_size)
    #         env_segment_ids = np.arange(start=self.args.num_nbr+1, stop=self.args.num_nbr+len(env_spatial_ids)+1)
    #         env_temporal_ids = np.ones(len(env_spatial_ids)) * self.args.obs_len
    #         env_attn_mask = np.ones(len(env_spatial_ids))
    #         envs_params = [tgt_env.min_x, tgt_env.min_y, tgt_env.width, tgt_env.height, tgt_env.resolution, 0.0]
    #         output = {'spatial_ids': torch.tensor(spatial_ids, dtype=torch.float),
    #                   'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
    #                   'temporal_ids': torch.tensor(temporal_ids, dtype=torch.long),
    #                   'attn_mask': torch.tensor(attn_mask, dtype=torch.float),
    #                   'env_spatial_ids': torch.tensor(env_spatial_ids, dtype=torch.float),
    #                   'env_segment_ids': torch.tensor(env_segment_ids, dtype=torch.long),
    #                   'env_temporal_ids': torch.tensor(env_temporal_ids, dtype=torch.long),
    #                   'env_attn_mask': torch.tensor(env_attn_mask, dtype=torch.float),
    #                   'traj_lbs': torch.tensor(traj_lbs, dtype=torch.float),
    #                   'goal_lbs': torch.tensor(goal_lbs, dtype=torch.float),
    #                   'envs': torch.tensor(tgt_env.grid_map, dtype=torch.float),
    #                   'envs_params': torch.tensor(envs_params, dtype=torch.float),
    #                   }
    #     else:
    #         output = {'spatial_ids': torch.tensor(spatial_ids, dtype=torch.float),
    #                   'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
    #                   'temporal_ids': torch.tensor(temporal_ids, dtype=torch.long),
    #                   'attn_mask': torch.tensor(attn_mask, dtype=torch.float),
    #                   'traj_lbs': torch.tensor(traj_lbs, dtype=torch.float),
    #                   'goal_lbs': torch.tensor(goal_lbs, dtype=torch.float),
    #                   }
    #
    #     return output

    # def __len__(self):
    #     return len(self.dataset)
    #
    # def __getitem__(self, index):
    #     first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data, neighbors_data_st, neighbors_lower_upper, neighbors_future, \
    #     neighbors_edge_value, robot_traj_st_t, map_tuple, scene_name, timestep = self.dataset.__getitem__(index)  #
    #     ret = {}
    #     ret['first_history_index'] = first_history_index
    #     ret['input_x'] = x_t
    #     ret['input_x_st'] = x_st_t
    #     ret['target_y'] = y_t
    #     ret['target_y_st'] = y_st_t
    #     ret['cur_image_file'] = ''
    #     ret['pred_resolution'] = torch.ones_like(y_t)
    #     ret['neighbors_x'] = neighbors_data
    #     ret['neighbors_x_st'] = neighbors_data_st
    #     ret['neighbors_lower_upper'] = neighbors_lower_upper
    #     ret['neighbors_target_y'] = neighbors_future
    #     ret['neighbors_adjacency'] = neighbors_edge_value
    #     ret['scene_name'] = scene_name
    #     ret['timestep'] = timestep
    #     return ret


# if __name__ == '__main__':
#     dataset = ETHUCYDataset(hyperparams)
