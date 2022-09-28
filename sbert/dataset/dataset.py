
import torch
import os
from torch.utils.data import Dataset
import tqdm
from .viz import *
from SocialBERT.sbert.dataset.indicator import SpatialIndicator, IntegerIndicator
from SocialBERT.sbert.dataset.util import *


class SocialBERTDataset(Dataset):
    def __init__(self, split, args):
        self.trajs = []
        self.path = os.path.join(args.dataset_path, args.dataset_name)
        self.seq_len = args.obs_len + args.pred_len
        self.seq_input_len = (args.obs_len+1) * (args.num_nbr + 1) + args.pred_len
        self.all_trajs = []
        self.all_scenes = []
        self.scene_images = {}
        self.envs = {}
        self.s_ind = SpatialIndicator(bound_range=args.view_range)
        self.i_ind = IntegerIndicator()
        self.args = args
        self.split = split

    def __len__(self):
        return len(self.all_trajs)

    def get_traj(self, item):
        return self.all_trajs[item]

    def __getitem__(self, item):
        trajs = copy.deepcopy(self.all_trajs[item])
        scale = 1.0/self.scales[self.all_scenes[item]]
        #################################################
        # fig1, ax1 = plt.subplots()
        # for idx, traj in enumerate(trajs):
        #     ax1.scatter(traj[:, 0], traj[:, 1])
        #################################################
        trajs, trans, rot = transform_to_target(trajs=trajs, obs_len=self.args.obs_len, traj_dim=trajs.shape[2])
        #################################################
        # fig2, ax2 = plt.subplots()
        # for idx, traj in enumerate(trajs):
        #     if idx == 0:
        #         ax2.scatter(traj[:, 0], traj[:, 1], c="k")
        #     else:
        #         ax2.scatter(traj[:, 0], traj[:, 1])
        # ax2.set_xlim([-self.args.view_range/2, self.args.view_range/2])
        # ax2.set_ylim([-self.args.view_range/2, self.args.view_range/2])
        # plt.show()
        #################################################
        if self.args.aug and self.split == 'train':
            # if self.args.scene is None:
            trajs = aug_random_scale(trajs)
            trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len, pred_len=self.args.pred_len,
                                       view_range=self.args.view_range, view_angle=self.args.view_angle,
                                       social_range=self.args.social_range)
            # print(trajs.shape)
            trajs = aug_random_flip(trajs)
            trajs = aug_random_rotation(trajs)
        else:
            trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len, pred_len=self.args.pred_len,
                                       view_range=self.args.view_range, view_angle=self.args.view_angle,
                                       social_range=self.args.social_range)

        #################################################
        # fig3, ax3 = plt.subplots()
        # for idx, traj in enumerate(trajs):
        #     if idx == 0:
        #         ax3.scatter(traj[:, 0], traj[:, 1], c="k")
        #     else:
        #         ax3.scatter(traj[:, 0], traj[:, 1])
        # ax3.set_xlim([-self.args.view_range/2, self.args.view_range/2])
        # ax3.set_ylim([-self.args.view_range/2, self.args.view_range/2])
        # plt.show()
        #################################################

        if self.args.mode == "pretrain":
            is_near_lbl = is_near(trajs=trajs, radius=self.args.social_range)

            trajs[1:, self.args.obs_len:, :] = np.nan
            tgt_traj = copy.deepcopy(trajs[0, :self.seq_len])
            # nan_idx = np.isnan(tgt_traj[:, 0])
            #
            # tgt_traj[nan_idx] = self.s_ind.pad_id
            # attn_mask = np.ones(len(nan_idx))
            # attn_mask[nan_idx] = self.i_ind.false_val
            #
            traj_mask = np.full(tgt_traj.shape, self.s_ind.false_val)
            rand_mask = np.random.choice(self.seq_len, 3, replace=False)
            for i, idx in enumerate(rand_mask):
                traj_mask[idx] = self.s_ind.true_id
                tgt_traj[idx] = self.s_ind.msk_id
            spatial_ids = [self.s_ind.sot_id] + tgt_traj.tolist()
            segment_ids = [1] + [1] * self.seq_len
            temporal_ids = [self.i_ind.pad_id] + [x for x in np.arange(1, self.seq_len+1)]
            attn_mask = [self.i_ind.true_val] + [self.i_ind.true_val] * self.seq_len
            traj_mask = traj_mask.tolist()  # [self.s_ind.false_val] + traj_mask.tolist()
            traj_lbl = copy.deepcopy(trajs[0, :self.seq_len]).tolist()
            # traj_lbl = tgt_traj
            # segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * self.args.pred_len
            # temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nan_idx[:self.args.obs_len])] + [x for x in np.arange(self.args.obs_len+1, self.seq_len+1)]
            # attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * self.args.pred_len

            nbr_obs_trajs = copy.deepcopy(trajs[1:, :self.args.obs_len, :])
            # nbr_traj_lbl = copy.deepcopy(trajs[1:, :self.args.obs_len, :])
            is_near_lbl = is_near_lbl[1:]
            near_lbl = []
            for nbr_idx, nbr_traj in enumerate(nbr_obs_trajs):
                nbr_traj_lbl = copy.deepcopy(nbr_traj)
                nbr_nan_idx = np.isnan(nbr_traj[:, 0])
                nbr_traj[nbr_nan_idx] = self.s_ind.pad_id
                nbr_traj_lbl[nbr_nan_idx] = self.s_ind.pad_id
                nbr_attn_mask = np.ones(len(nbr_traj))
                nbr_attn_mask[nbr_nan_idx] = self.i_ind.false_val
                nbr_traj_mask = np.full(nbr_traj.shape, self.s_ind.false_val)  # true value

                pos_idx = np.nonzero(~nbr_nan_idx)[0]
                if len(pos_idx) >= 4:
                    num_mask = 1  # int(len(inbound_pos) * 0.15)
                    rand_mask = np.random.choice(pos_idx, num_mask, replace=False)
                    nbr_traj[rand_mask] = self.s_ind.msk_id
                    nbr_traj_mask[rand_mask] = self.s_ind.true_id

                spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
                segment_ids += [nbr_idx+2] + [self.i_ind.pad_id if nan else nbr_idx+2 for nan in nbr_nan_idx]
                temporal_ids += [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nbr_nan_idx)]
                attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()
                traj_lbl += nbr_traj_lbl.tolist()
                traj_mask += nbr_traj_mask.tolist()
                near_lbl += [self.i_ind.near_lbl] if is_near_lbl[nbr_idx] else [self.i_ind.far_lbl]
            # collate
            ext_len = self.seq_input_len - len(spatial_ids)
            spatial_ids_padding = [self.s_ind.pad_id] * ext_len # [self.s_ind.pad_id for _ in range(ext_len)]
            spatial_ids.extend(spatial_ids_padding)
            integer_ids_padding = [self.i_ind.pad_id] * ext_len # [self.i_ind.pad_id for _ in range(ext_len)]
            segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(integer_ids_padding)
            # traj_lbl_ext_len = self.seq_len - (self.args.num_nbr+1) - len(traj_lbl)
            near_lbl_ext_len = self.args.num_nbr - len(near_lbl)
            near_padding = [self.i_ind.none_lbl] * near_lbl_ext_len
            near_lbl.extend(near_padding)
            traj_lbl += [self.s_ind.pad_id] * near_lbl_ext_len * self.args.obs_len
            traj_mask += [self.s_ind.false_id] * near_lbl_ext_len * self.args.obs_len
            # if np.any(np.isnan(np.array(traj_lbl))) or np.any(np.isnan(np.array(traj_mask))) or np.any(np.isnan(np.array(spatial_ids))):
            #     print("traj_lbl:", np.array(traj_lbl))
            #     # print("traj_mask:", traj_mask)
            #     print("spatial_ids:", np.array(spatial_ids))
            #     print("attn_mask:", np.array(attn_mask))
            output = {'spatial_ids': torch.tensor(spatial_ids, dtype=torch.float),
                      'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
                      'temporal_ids': torch.tensor(temporal_ids, dtype=torch.long),
                      'attn_mask': torch.tensor(attn_mask, dtype=torch.float),
                      'traj_mask': torch.tensor(traj_mask, dtype=torch.float),
                      'traj_lbl': torch.tensor(traj_lbl, dtype=torch.float),
                      'near_lbl': torch.tensor(near_lbl, dtype=torch.long),
                      'scales': torch.tensor(scale, dtype=torch.float)
                      }
            return output

        else: # Fine-tuning
            # Target pedestrian trajectory
            tgt_obs_traj = copy.deepcopy(trajs[0, :self.args.obs_len])
            nan_idx = np.isnan(tgt_obs_traj[:, 0])
            tgt_obs_traj[nan_idx] = self.s_ind.pad_id
            spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.msk_id] * self.args.pred_len
            segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * self.args.pred_len
            temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nan_idx[:self.args.obs_len])] + [x for x in np.arange(self.args.obs_len+1, self.seq_len+1)]
            attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * self.args.pred_len
            # traj_mask = np.full((self.seq_len, self.args.input_dim), self.s_ind.false_val)
            # traj_mask = traj_mask.tolist()
            traj_lbl = copy.deepcopy(trajs[0, self.args.obs_len:])

            nbr_obs_trajs = copy.deepcopy(trajs[1:, :self.args.obs_len, :])
            for nbr_idx, nbr_traj in enumerate(nbr_obs_trajs):
                nbr_nan_idx = np.isnan(nbr_traj[:, 0])
                nbr_traj[nbr_nan_idx] = self.s_ind.pad_id
                # nbr_attn_mask = np.zeros(len(nbr_traj))
                nbr_attn_mask = np.ones(len(nbr_traj))
                nbr_attn_mask[nbr_nan_idx] = self.i_ind.false_val
                # nbr_traj_mask = np.full(nbr_traj.shape, self.s_ind.false_val)  # true value
                spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
                segment_ids += [nbr_idx+2] + [self.i_ind.pad_id if nan else nbr_idx+2 for nan in nbr_nan_idx]
                temporal_ids += [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nbr_nan_idx)]
                attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()

            ext_len = self.seq_input_len - len(spatial_ids)
            spatial_ids_padding = [self.s_ind.pad_id] * ext_len # [self.s_ind.pad_id for _ in range(ext_len)]
            spatial_ids.extend(spatial_ids_padding)
            integer_ids_padding = [self.i_ind.pad_id] * ext_len # [self.i_ind.pad_id for _ in range(ext_len)]
            segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(integer_ids_padding)

            output = {'spatial_ids': torch.tensor(spatial_ids, dtype=torch.float),
                      'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
                      'temporal_ids': torch.tensor(temporal_ids, dtype=torch.long),
                      'attn_mask': torch.tensor(attn_mask, dtype=torch.float),
                      'traj_lbl': torch.tensor(traj_lbl, dtype=torch.float),
                      'scales': torch.tensor(scale, dtype=torch.float)
                      }
            return output


