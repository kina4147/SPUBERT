'''
NOTE: May 6
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
# sys.path.append(os.path.realpath('./sbertplus/dataset'))
import tqdm
import cv2
import yaml
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
from SocialBERT.sbertplus.dataset.grid_map_numpy import RectangularGridMap
from SocialBERT.sbertplus.dataset.util import is_target_outbound


class SDDYNetDataset(SocialBERTDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        filepath = os.path.join(self.path, split + '_trajnet.pkl')
        df_data = pd.read_pickle(filepath)
        df_data.head()
        self.env = {}
        self.min_obs_len = self.args.min_obs_len if self.split == 'train' else self.args.obs_len


        with open(os.path.join(self.path, 'scales.yml'), 'r') as f:
            self.scales = yaml.load(f, Loader=yaml.FullLoader)

        # for sid in self.scales:
        #     self.scales[sid] = 0.25

        scene_trajs, meta, scene_ids, scene_frames, scene_start_frames = self.split_trajectories_by_scene(df_data, self.args.obs_len+self.args.pred_len)

        if args.scene:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                img_path = os.path.join(self.path, split + '_masks', scene_id + '_mask.png')
                # homo_path = os.path.join(self.path, scene_id + '_H.txt')
                # homo_mat = np.loadtxt(homo_path)
                # num_ped, seq_len, sdim = trajs.shape
                # if scene_id in ['eth', 'hotel']:
                # trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
                # trajs = trajs.reshape(-1, 2)
                # trajs = self.image2world(trajs, homo_mat)
                # trajs = trajs.reshape(num_ped, seq_len, sdim)
                # num_scene = 0
                # if self.args.scene:
                # scenes = [scene_id] * num_scene
                # self.all_scenes += scenes

                map = cv2.imread(img_path, 0)
                width = float(map.shape[1])
                height = float(map.shape[0])
                x_ids, y_ids = np.mgrid[slice(0, width, 1), slice(0, height, 1)]
                x_ids = x_ids.flatten().astype(np.int)
                y_ids = y_ids.flatten().astype(np.int)
                x_pos = self.scales[scene_id] * x_ids.astype(np.float)
                y_pos = self.scales[scene_id] * y_ids.astype(np.float)
                num_rows = int((self.scales[scene_id] * height) / args.env_resol)
                num_cols = int((self.scales[scene_id] * width) / args.env_resol)
                num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
                num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
                grid_map = RectangularGridMap(width=num_cols, height=num_rows, resolution=args.env_resol, center_x=self.scales[scene_id] * width/2, center_y=self.scales[scene_id] * height/2)
                grid_map.set_values_from_xy_pos(x_pos=x_pos, y_pos=y_pos, vals=map[y_ids, x_ids])
                self.envs[scene_id] = grid_map

                for start_frame in start_frames:
                    curr_trajs = trajs[frames == start_frame]
                    curr_trajs = self.scales[scene_id] * curr_trajs
                    for idx in range(len(curr_trajs)):
                        tmp_curr_trajs = copy.deepcopy(curr_trajs)
                        tmp_curr_tgt_traj = copy.deepcopy(curr_trajs[0])
                        tmp_curr_trajs[0] = tmp_curr_trajs[idx]
                        tmp_curr_trajs[idx] = tmp_curr_tgt_traj
                        # collision Filtering
                        # vals, valid = grid_map.get_value_from_xy_pos(tmp_curr_trajs[0][:, 0], tmp_curr_trajs[0][:, 1])
                        # if np.any(vals > 2):
                        #     print("Trajectory is on structure.")
                        #     continue
                        if is_target_outbound(tmp_curr_trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                            print("Target is outbound.")
                            continue
                        self.all_trajs.append(tmp_curr_trajs)
                        self.all_scenes.append(scene_id)

                # self.all_trajs = self.all_trajs[:100]
                if self.split == 'test' and self.args.viz and len(self.all_trajs) > 100:
                    break
        else:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                for start_frame in start_frames:
                    curr_trajs = trajs[frames == start_frame]
                    curr_trajs = self.scales[scene_id] * curr_trajs
                    for idx in range(len(curr_trajs)):
                        tmp_curr_trajs = copy.deepcopy(curr_trajs)
                        tmp_curr_tgt_traj = copy.deepcopy(curr_trajs[0])
                        tmp_curr_trajs[0] = tmp_curr_trajs[idx]
                        tmp_curr_trajs[idx] = tmp_curr_tgt_traj
                        if is_target_outbound(tmp_curr_trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                            print("target is outbound")
                            continue
                        self.all_trajs.append(tmp_curr_trajs)
                        self.all_scenes.append(scene_id)
                if self.split == 'test' and self.args.viz and len(self.all_trajs) > 100:
                    break


    def split_trajectories_by_scene(self, data, total_len):
        trajectories = []
        meta = []
        scene_list = []
        scene_frames = []
        first_frames = []
        for meta_id, meta_df in data.groupby('sceneId', as_index=False):
            frames = meta_df[['frame']].to_numpy().astype('float32').reshape(-1, total_len)
            frames = np.min(frames, axis=-1)
            first_frames.append(frames)
            scene_frames.append(np.unique(frames))
            trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
            meta.append(meta_df)
            scene_list.append(meta_id)

        return np.array(trajectories), meta, scene_list, first_frames, scene_frames
    #
    # def generate_scenes(self, raw_data, frame_gap=12, scale=1.0):
    #     raw_data = raw_data.sort_values(by=['frame', 'trackId'])
    #     frame_ids = raw_data.frame.unique().tolist() # sorted
    #     seqs = []
    #     start_frames = []
    #     start_frame_ids = []
    #     all_trajs = []
    #     for idx in range(len(frame_ids) - self.seq_len + 1):
    #         if (frame_ids[idx+self.seq_len-1] - frame_ids[idx]) / frame_gap == self.seq_len-1:
    #             start_frames.append(frame_ids[idx])
    #             start_frame_ids.append(idx)
    #         seqs.append(np.array(raw_data[raw_data.frame == frame_ids[idx]]))
    #
    #     # frame_iter = tqdm.tqdm(zip(start_frames, start_frame_ids, frames), total=len(start_frames))
    #     for start_frame, start_frame_id in tqdm.tqdm(zip(start_frames, start_frame_ids), total=len(start_frames)):
    #         seq = np.concatenate(seqs[start_frame_id:(start_frame_id + self.seq_len)], axis=0)
    #         pids = np.unique(seq[:, 1]) # pedestrian ids
    #         # Nped X Nseq X Ndim (x, y)
    #         for tgt_pidx, tgt_pid in enumerate(pids):
    #             trajs = np.full((len(pids), self.seq_len, 2), np.nan)
    #             tgt_seq = seq[seq[:, 1] == tgt_pid, :]
    #             # target outbound
    #             tgt_frames = (tgt_seq[:, 0] - start_frame) // frame_gap
    #             tgt_frames = tgt_frames.astype(int)
    #             trajs[0, tgt_frames, :] = scale*tgt_seq[:, 2:4]
    #
    #             if np.any(np.isnan(trajs[0, (self.args.obs_len - self.min_obs_len):, 0])):
    #                continue
    #             if is_target_outbound(trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
    #                 print("target is outbound")
    #                 continue
    #             nbr_pids = np.delete(pids, np.where(pids == tgt_pid))
    #             for nbr_pidx, nbr_pid in enumerate(nbr_pids):
    #                 nbr_seq = seq[seq[:, 1] == nbr_pid, :]
    #                 nbr_frames = (nbr_seq[:, 0] - start_frame) // frame_gap
    #                 nbr_frames = nbr_frames.astype(int)
    #                 # if len(nbr_frames[nbr_frames < self.args.obs_len]) < 4:
    #                 #     continue
    #                 # if max(nbr_frames) > self.seq_len:
    #                 #     print(seq, start_frame, nbr_frames, nbr_seq[:, 0], seq[0, 0], seq[-1, 0])
    #                 trajs[nbr_pidx+1, nbr_frames, :] = scale*nbr_seq[:, 2:4]
    #
    #             # NaN nbr removal
    #             traj_valid = [False if np.all(np.isnan(traj[:self.args.obs_len, 0])) else True for traj in trajs]
    #             trajs = trajs[traj_valid]
    #             all_trajs.append(trajs)
    #     return all_trajs

