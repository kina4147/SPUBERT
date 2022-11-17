import os
import copy
import numpy as np
import pandas as pd
import yaml
import cv2
from SPUBERT.dataset.dataset import SPUBERTDataset
from SPUBERT.dataset.grid_map_numpy import RectangularGridMap
from SPUBERT.dataset.util import is_target_outbound

class ETHUCYDataset(SPUBERTDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        filepath = os.path.join(self.path, self.args.dataset_split + '_' + self.split + '.pkl')
        df_data = pd.read_pickle(filepath)
        df_data.head()
        self.env = {}
        self.min_obs_len = 2 if self.split == 'train' else self.args.obs_len
        with open(os.path.join(self.path, 'scales.yml'), 'r') as f:
            self.scales = yaml.load(f, Loader=yaml.FullLoader)
        scene_trajs, meta, scene_ids, scene_frames, scene_start_frames = self.split_trajectories_by_scene(df_data, self.args.obs_len+self.args.pred_len)

        if args.scene:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                img_path = os.path.join(self.path, scene_id, 'oracle.png')
                homo_path = os.path.join(self.path, scene_id + '_H.txt')
                homo_mat = np.loadtxt(homo_path)
                num_ped, seq_len, sdim = trajs.shape
                trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
                trajs = trajs.reshape(-1, 2)
                trajs = self.image2world(trajs, homo_mat)
                trajs = trajs.reshape(num_ped, seq_len, sdim)
                map = cv2.imread(img_path, 0)
                corner_uv = np.array([[0, 0], [map.shape[1], 0],[0, map.shape[0]], [map.shape[1], map.shape[0]]])
                corner_xy = self.image2world(corner_uv, homo_mat)
                corner_xy = self.scales[scene_id] * corner_xy
                min_x, max_x = min(corner_xy[:, 0]), max(corner_xy[:, 0])
                min_y, max_y = min(corner_xy[:, 1]), max(corner_xy[:, 1])
                width = max_x - min_x
                height = max_y - min_y
                cntr_x = (max_x + min_x) / 2.0
                cntr_y = (max_y + min_y) / 2.0
                occupied_idxs = np.where(map == 0)
                occupied_idxs = np.stack(occupied_idxs)
                occupied_xys = self.image2world(occupied_idxs.T, homo_mat)
                occupied_xys = self.scales[scene_id] * occupied_xys
                unoccupied_idxs = np.where(map == 1)
                unoccupied_idxs = np.stack(unoccupied_idxs)
                unoccupied_xys = self.image2world(unoccupied_idxs.T, homo_mat)
                unoccupied_xys = self.scales[scene_id] * unoccupied_xys
                num_rows = int(height / args.env_resol)
                num_cols = int(width / args.env_resol)
                num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
                num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
                grid_map = RectangularGridMap(width=num_cols, height=num_rows, resolution=args.env_resol, center_x=cntr_x, center_y=cntr_y)
                grid_map.set_value_from_xy_pos(occupied_xys[:, 0], occupied_xys[:, 1], 2.0)
                grid_map.set_value_from_xy_pos(unoccupied_xys[:, 0], unoccupied_xys[:, 1], 1.0)
                self.envs[scene_id] = grid_map

                for start_frame in start_frames:
                    curr_trajs = trajs[frames == start_frame]
                    curr_trajs = self.scales[scene_id] * curr_trajs
                    for idx in range(len(curr_trajs)):
                        tmp_curr_trajs = copy.deepcopy(curr_trajs)
                        tmp_curr_tgt_traj = copy.deepcopy(curr_trajs[0])
                        tmp_curr_trajs[0] = tmp_curr_trajs[idx]
                        tmp_curr_trajs[idx] = tmp_curr_tgt_traj
                        if is_target_outbound(tmp_curr_trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                            continue
                        self.all_trajs.append(tmp_curr_trajs)
                        self.all_scenes.append(scene_id)
        else:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                homo_path = os.path.join(self.path, scene_id + '_H.txt')
                homo_mat = np.loadtxt(homo_path)
                num_ped, seq_len, sdim = trajs.shape
                trajs = trajs.reshape(-1, 2)
                trajs = self.image2world(trajs, homo_mat)
                trajs = trajs.reshape(num_ped, seq_len, sdim)

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

    def world2image(self, traj_w, H_inv):
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
        traj_cam = np.matmul(H_inv, traj_homog)
        traj_uvz = np.transpose(traj_cam / traj_cam[2])
        return traj_uvz[:, :2].astype(int)

    def image2world(self, traj_px, H):
        pp = np.stack((traj_px[:, 0], traj_px[:, 1], np.ones(len(traj_px))), axis=1)
        PP = np.matmul(H, pp.T).T
        P_normal = PP[:, :2] / np.repeat(PP[:, 2].reshape((-1, 1)), 2, axis=1)
        return P_normal
