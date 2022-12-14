import os
import torch
from torch.utils.data import Dataset
import tqdm
from SPUBERT.dataset.util import *
from SPUBERT.dataset.indicator import SpatialIndicator, IntegerIndicator


class SPUBERTDataset(Dataset):
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

    def generate_scenes(self, raw_data, frame_gap=12):
        raw_data.sort_values(by=['frame', 'id'], inplace=True)
        frame_ids = raw_data.frame.unique().tolist() # sorted
        seqs = []
        start_frames = []
        start_frame_ids = []
        all_trajs = []
        for idx in range(len(frame_ids) - self.seq_len + 1):
            # incomplete sequences
            if (frame_ids[idx+self.seq_len-1] - frame_ids[idx]) / frame_gap == self.seq_len-1:
                start_frames.append(frame_ids[idx])
                start_frame_ids.append(idx)

            seqs.append(np.array(raw_data[raw_data.frame == frame_ids[idx]]))

        for start_frame, start_frame_id in tqdm.tqdm(zip(start_frames, start_frame_ids), total=len(start_frames)):
            seq = np.concatenate(seqs[start_frame_id:(start_frame_id + self.seq_len)], axis=0)
            pids = np.unique(seq[:, 1])
            # Nped X Nseq X Ndim (x, y)
            for tgt_pidx, tgt_pid in enumerate(pids):
                trajs = np.full((len(pids), self.seq_len, 2), np.nan)
                tgt_seq = seq[seq[:, 1] == tgt_pid, :]
                tgt_frames = (tgt_seq[:, 0] - start_frame) // frame_gap
                tgt_frames = tgt_frames.astype(int)
                trajs[0, tgt_frames, :] = tgt_seq[:, 2:4]

                if np.any(np.isnan(trajs[0, (self.args.obs_len - self.min_obs_len):, 0])):
                   continue
                if is_target_outbound(trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                    print("target is outbound")
                    continue

                nbr_pids = np.delete(pids, np.where(pids == tgt_pid))
                for nbr_pidx, nbr_pid in enumerate(nbr_pids):
                    nbr_seq = seq[seq[:, 1] == nbr_pid, :]
                    nbr_frames = (nbr_seq[:, 0] - start_frame) // frame_gap
                    nbr_frames = nbr_frames.astype(int)
                    trajs[nbr_pidx+1, nbr_frames, :] = nbr_seq[:, 2:4]
                traj_valid = [False if np.all(np.isnan(traj[:self.args.obs_len, 0])) else True for traj in trajs]
                trajs = trajs[traj_valid]
                all_trajs.append(trajs)
        return all_trajs

    def __getitem__(self, item):
        trajs = copy.deepcopy(self.all_trajs[item])
        scale = 1.0/self.scales[self.all_scenes[item]]
        if self.args.scene:
            env = copy.deepcopy(self.envs[self.all_scenes[item]])
        trajs, trans, rot = transform_to_target(trajs=trajs, obs_len=self.args.obs_len, traj_dim=trajs.shape[2])
        if self.args.scene:
            tgt_env = extract_map(env, range=1.5*self.args.env_range, resol=self.args.env_resol, trans=trans, rot=rot)
        else:
            tgt_env = None
        if self.split == 'train':
            trajs = aug_random_scale(trajs)
            trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len, pred_len=self.args.pred_len,
                                       view_range=self.args.view_range, view_angle=self.args.view_angle,
                                       social_range=self.args.social_range)
            trajs, tgt_env = aug_random_flip(trajs, tgt_env)
            trajs, tgt_env = aug_random_rotation(trajs, tgt_env)
        else:
            trajs = neighbor_filtering(trajs, num_nbr=self.args.num_nbr, obs_len=self.args.obs_len, pred_len=self.args.pred_len,
                                       view_range=self.args.view_range, view_angle=self.args.view_angle,
                                       social_range=self.args.social_range)

        if self.args.scene:
            tgt_env = extract_map(tgt_env, range=self.args.env_range, resol=self.args.env_resol)

        trajs[1:, self.args.obs_len:, :] = np.nan
        tgt_pred_traj = copy.deepcopy(trajs[0, self.args.obs_len:])
        tgt_obs_traj = copy.deepcopy(trajs[0, :self.args.obs_len])
        goal_lbl = tgt_pred_traj[-1, :]
        traj_lbl = tgt_pred_traj
        nan_idx = np.isnan(tgt_obs_traj[:, 0])
        tgt_obs_traj[nan_idx] = self.s_ind.pad_id
        mgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.pad_id] * (
                    self.args.pred_len - 1) + [self.s_ind.msk_id]
        mgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [
            self.i_ind.pad_id] * (self.args.pred_len - 1) + [1]
        mgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in
                                                  zip(range(1, self.args.obs_len + 1),
                                                      nan_idx[:self.args.obs_len])] + [self.i_ind.pad_id] * (
                                       self.args.pred_len - 1) + [self.seq_len]
        mgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in
                                                 nan_idx[:self.args.obs_len]] + [self.i_ind.false_val] * (
                                    self.args.pred_len - 1) + [self.i_ind.true_val]

        tgp_spatial_ids = [self.s_ind.sot_id] + tgt_obs_traj[:self.args.obs_len].tolist() + [self.s_ind.msk_id] * (self.args.pred_len - 1) + [goal_lbl.tolist()]
        tgp_segment_ids = [1] + [self.i_ind.pad_id if nan else 1 for nan in nan_idx[:self.args.obs_len]] + [1] * (self.args.pred_len)
        tgp_temporal_ids = [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nan_idx[:self.args.obs_len])] + [x for x in np.arange(self.args.obs_len+1, self.seq_len+1)]
        tgp_attn_mask = [self.i_ind.true_val] + [self.i_ind.false_val if nan else self.i_ind.true_val for nan in nan_idx[:self.args.obs_len]] + [self.i_ind.true_val] * (self.args.pred_len)

        nbr_obs_trajs = copy.deepcopy(trajs[1:, :self.args.obs_len, :])
        spatial_ids = []
        segment_ids = []
        temporal_ids = []
        attn_mask = []
        for nbr_idx, nbr_traj in enumerate(nbr_obs_trajs):
            nbr_nan_idx = np.isnan(nbr_traj[:, 0])
            nbr_traj[nbr_nan_idx] = self.s_ind.pad_id
            nbr_attn_mask = np.ones(len(nbr_traj))
            nbr_attn_mask[nbr_nan_idx] = self.i_ind.false_val
            spatial_ids += [self.s_ind.sep_id] + nbr_traj.tolist()
            segment_ids += [nbr_idx+2] + [self.i_ind.pad_id if nan else nbr_idx+2 for nan in nbr_nan_idx]
            temporal_ids += [self.i_ind.pad_id] + [self.i_ind.pad_id if nan else x for x, nan in zip(range(1, self.args.obs_len+1), nbr_nan_idx)]
            attn_mask += [self.i_ind.true_val] + nbr_attn_mask.tolist()
        # collate
        ext_len = self.seq_input_len - len(spatial_ids) - len(mgp_spatial_ids)
        spatial_ids_padding = [self.s_ind.pad_id] * ext_len
        spatial_ids.extend(spatial_ids_padding)
        integer_ids_padding = [self.i_ind.pad_id] * ext_len
        segment_ids.extend(integer_ids_padding), temporal_ids.extend(integer_ids_padding), attn_mask.extend(integer_ids_padding)
        mgp_spatial_ids += spatial_ids
        mgp_segment_ids += segment_ids
        mgp_temporal_ids += temporal_ids
        mgp_attn_mask += attn_mask
        tgp_spatial_ids += spatial_ids
        tgp_segment_ids += segment_ids
        tgp_temporal_ids += temporal_ids
        tgp_attn_mask += attn_mask
        if self.args.scene:
            if tgt_env.width % self.args.patch_size != 0:
                pad_size = self.args.patch_size - (tgt_env.width % self.args.patch_size) // 2
                tgt_env = expand_map_with_pad(tgt_env, 0, pad_size)

            env_spatial_ids, env_attn_mask  = extract_patch_from_map(tgt_env, self.args.patch_size)
            env_segment_ids = np.arange(start=self.args.num_nbr+1, stop=self.args.num_nbr+len(env_spatial_ids)+1)
            env_temporal_ids = np.ones(len(env_spatial_ids)) * self.args.obs_len
            envs_params = [tgt_env.min_x, tgt_env.min_y, tgt_env.width, tgt_env.height, tgt_env.resolution, 2]

            output = {'mgp_spatial_ids': torch.tensor(mgp_spatial_ids, dtype=torch.float),
                      'mgp_segment_ids': torch.tensor(mgp_segment_ids, dtype=torch.long),
                      'mgp_temporal_ids': torch.tensor(mgp_temporal_ids, dtype=torch.long),
                      'mgp_attn_mask': torch.tensor(mgp_attn_mask, dtype=torch.float),
                      'tgp_spatial_ids': torch.tensor(tgp_spatial_ids, dtype=torch.float),
                      'tgp_segment_ids': torch.tensor(tgp_segment_ids, dtype=torch.long),
                      'tgp_temporal_ids': torch.tensor(tgp_temporal_ids, dtype=torch.long),
                      'tgp_attn_mask': torch.tensor(tgp_attn_mask, dtype=torch.float),
                      'env_spatial_ids': torch.tensor(np.array(env_spatial_ids), dtype=torch.float),
                      'env_segment_ids': torch.tensor(env_segment_ids, dtype=torch.long),
                      'env_temporal_ids': torch.tensor(env_temporal_ids, dtype=torch.long),
                      'env_attn_mask': torch.tensor(env_attn_mask, dtype=torch.float),
                      'traj_lbl': torch.tensor(traj_lbl, dtype=torch.float),
                      'goal_lbl': torch.tensor(goal_lbl, dtype=torch.float),
                      'envs': torch.tensor(tgt_env.grid_map, dtype=torch.float),
                      'envs_params': torch.tensor(envs_params, dtype=torch.float),
                      'scales': torch.tensor(scale, dtype=torch.float),
                      }
        else:
            output = {'mgp_spatial_ids': torch.tensor(mgp_spatial_ids, dtype=torch.float),
                      'mgp_segment_ids': torch.tensor(mgp_segment_ids, dtype=torch.long),
                      'mgp_temporal_ids': torch.tensor(mgp_temporal_ids, dtype=torch.long),
                      'mgp_attn_mask': torch.tensor(mgp_attn_mask, dtype=torch.float),
                      'tgp_spatial_ids': torch.tensor(tgp_spatial_ids, dtype=torch.float),
                      'tgp_segment_ids': torch.tensor(tgp_segment_ids, dtype=torch.long),
                      'tgp_temporal_ids': torch.tensor(tgp_temporal_ids, dtype=torch.long),
                      'tgp_attn_mask': torch.tensor(tgp_attn_mask, dtype=torch.float),
                      'traj_lbl': torch.tensor(traj_lbl, dtype=torch.float),
                      'goal_lbl': torch.tensor(goal_lbl, dtype=torch.float),
                      'scales': torch.tensor(scale, dtype=torch.float)
                      }

        return output


