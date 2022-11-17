'''
NOTE: May 6
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import cv2
import yaml
import copy
import numpy as np
import pandas as pd
from SPUBERT.dataset.dataset import SPUBERTDataset
from SPUBERT.dataset.grid_map_numpy import RectangularGridMap
from SPUBERT.dataset.util import is_target_outbound


class SDDDataset(SPUBERTDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        filepath = os.path.join(self.path, split + '_trajnet.pkl')
        df_data = pd.read_pickle(filepath)
        df_data.head()
        self.env = {}
        self.min_obs_len = 2 if self.split == 'train' else self.args.obs_len


        with open(os.path.join(self.path, 'scales.yml'), 'r') as f:
            self.scales = yaml.load(f, Loader=yaml.FullLoader)

        scene_trajs, meta, scene_ids, scene_frames, scene_start_frames = self.split_trajectories_by_scene(df_data, self.args.obs_len+self.args.pred_len)

        if args.scene:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                img_path = os.path.join(self.path, split + '_masks', scene_id + '_mask.png')
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
                        if is_target_outbound(tmp_curr_trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                            print("Target is outbound.")
                            continue
                        self.all_trajs.append(tmp_curr_trajs)
                        self.all_scenes.append(scene_id)

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