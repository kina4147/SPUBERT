'''
NOTE: May 6
Adopt the Trajectron dataset to make experiment easier

TODO: convert to our own dataset format later
'''
import os
import sys
import copy
# sys.path.append(os.path.realpath('./dataset'))
import numpy as np
import pandas as pd
import tqdm
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
        self.min_obs_len = self.args.min_obs_len if self.split == 'train' else self.args.obs_len
        # num_scene = 0
        # self.scales = {}
        with open(os.path.join(self.path, 'scales.yml'), 'r') as f:
            self.scales = yaml.load(f, Loader=yaml.FullLoader)
        scene_trajs, meta, scene_ids, scene_frames, scene_start_frames = self.split_trajectories_by_scene(df_data, self.args.obs_len+self.args.pred_len)

        if args.scene:
            for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
                img_path = os.path.join(self.path, scene_id, 'oracle.png')
                homo_path = os.path.join(self.path, scene_id + '_H.txt')
                homo_mat = np.loadtxt(homo_path)
                num_ped, seq_len, sdim = trajs.shape
                # if scene_id in ['eth', 'hotel']:
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
                        # collision Filtering
                        # vals, valid = grid_map.get_value_from_xy_pos(tmp_curr_trajs[0][:, 0], tmp_curr_trajs[0][:, 1])
                        # if np.any(vals > 1):
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
                homo_path = os.path.join(self.path, scene_id + '_H.txt')
                homo_mat = np.loadtxt(homo_path)
                num_ped, seq_len, sdim = trajs.shape
                # if scene_id in ['eth', 'hotel']:
                #     trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
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
                if self.split == 'test' and self.args.viz and len(self.all_trajs) > 100:
                    break

                if len(self.all_trajs) > 100:
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

    def world2image(self, traj_w, H_inv):
        # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
        traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
        # to camera frame
        traj_cam = np.matmul(H_inv, traj_homog)
        # to pixel coords
        traj_uvz = np.transpose(traj_cam / traj_cam[2])
        return traj_uvz[:, :2].astype(int)

    def image2world(self, traj_px, H):
        pp = np.stack((traj_px[:, 0], traj_px[:, 1], np.ones(len(traj_px))), axis=1)
        PP = np.matmul(H, pp.T).T
        P_normal = PP[:, :2] / np.repeat(PP[:, 2].reshape((-1, 1)), 2, axis=1)
        return P_normal


    # def image2world(self, image_coords, scene, homo_mat, resize):
    #     traj_image2world = copy.deepcopy(image_coords) #.clone()
    #     # if traj_image2world.dim() == 4:
    #     #     traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
    #     if scene in ['eth', 'hotel']:
    #         traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
    #     traj_image2world = traj_image2world / resize
    #     traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
    #     traj_image2world = traj_image2world.reshape(-1, 3)
    #     traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
    #     traj_image2world = traj_image2world / traj_image2world[:, 2:]
    #     traj_image2world = traj_image2world[:, :2]
    #     traj_image2world = traj_image2world.view_as(image_coords)
    #     return traj_image2world
    #
    #
    # def generate_scenes(self, raw_data, scene, homo_mat, frame_gap=10):
    #     raw_data = raw_data.sort_values(by=['frame'])#, 'trackId'])
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
    #         pids = np.unique(seq[:, 0]) # pedestrian ids
    #         # Nped X Nseq X Ndim (x, y)
    #         for tgt_pidx, tgt_pid in enumerate(pids):
    #             trajs = np.full((len(pids), self.seq_len, 2), np.nan)
    #             tgt_seq = seq[seq[:, 0] == tgt_pid, :]
    #             # target outbound
    #             tgt_frames = (tgt_seq[:, 1] - start_frame) // frame_gap
    #             tgt_frames = tgt_frames.astype(int)
    #             trajs[0, tgt_frames, :] = tgt_seq[:, 2:4]
    #             if np.any(np.isnan(trajs[0, (self.args.obs_len - self.min_obs_len):, 0])):
    #                continue
    #             # tgt_traj = copy.deepcopy(trajs[0])
    #             # tgt_traj = self.image2world(tgt_traj, homo_mat)
    #
    #             nbr_pids = np.delete(pids, np.where(pids == tgt_pid))
    #             for nbr_pidx, nbr_pid in enumerate(nbr_pids):
    #                 nbr_seq = seq[seq[:, 0] == nbr_pid, :]
    #                 nbr_frames = (nbr_seq[:, 1] - start_frame) // frame_gap
    #                 nbr_frames = nbr_frames.astype(int)
    #                 # if len(nbr_frames[nbr_frames < self.args.obs_len]) < 4:
    #                 #     continue
    #                 # if max(nbr_frames) > self.seq_len:
    #                 #     print(seq, start_frame, nbr_frames, nbr_seq[:, 0], seq[0, 0], seq[-1, 0])
    #                 trajs[nbr_pidx+1, nbr_frames, :] = nbr_seq[:, 2:4]
    #
    #             # NaN nbr removal
    #             traj_valid = [False if np.all(np.isnan(traj[:self.args.obs_len, 0])) else True for traj in trajs]
    #             trajs = trajs[traj_valid]
    #             # if scene in ['eth', 'hotel']:
    #             trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
    #             num_ped, seq_len, sdim = trajs.shape
    #             trajs = trajs.reshape(-1, 2)
    #             trajs = self.image2world(trajs, homo_mat)
    #             trajs = trajs.reshape(num_ped, seq_len, sdim)
    #             if is_target_outbound(trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
    #                 print("target is outbound")
    #                 continue
    #
    #             all_trajs.append(trajs)
    #     return all_trajs

    # def generate_scenes(self, raw_data, frame_gap=10):
    #     # raw_data = raw_data.sort_values(by=['sceneId', 'frame', 'trackId'])
    #     scene_ids = raw_data.sceneId.unique().tolist()
    #     seqs = []
    #     start_frames = []
    #     start_frame_ids = []
    #     all_trajs = []
    #     all_scenes = []
    #
    #     for sidx, sid in enumerate(scene_ids):
    #         scene_data = raw_data[raw_data.sceneId == sid]
    #         scene_data = scene_data.sort_values(by=['frame', 'trackId'])
    #         frame_ids = scene_data.frame.unique().tolist() # sorted
    #         # # none empty frame?
    #         # for fidx, fids in enumerate(frame_ids):
    #         for idx in range(len(frame_ids) - self.seq_len + 1):
    #             if (frame_ids[idx+self.seq_len-1] - frame_ids[idx]) / frame_gap == self.seq_len-1:
    #                 start_frames.append(frame_ids[idx])
    #                 start_frame_ids.append(idx)
    #             seqs.append(np.array(scene_data[scene_data.frame == frame_ids[idx]]))
    #         for start_frame, start_frame_id in tqdm.tqdm(zip(start_frames, start_frame_ids), total=len(start_frames)):
    #             seq = np.concatenate(seqs[start_frame_id:(start_frame_id + self.seq_len)], axis=0)
    #             pids = np.unique(seq[:, 0]) # pedestrian ids
    #             # Nped X Nseq X Ndim (x, y)
    #             for tgt_pidx, tgt_pid in enumerate(pids):
    #                 trajs = np.full((len(pids), self.seq_len, 2), np.nan)
    #                 tgt_seq = seq[seq[:, 0] == tgt_pid, :]
    #                 # target outbound
    #                 tgt_frames = (tgt_seq[:, 1] - start_frame) // frame_gap
    #                 tgt_frames = tgt_frames.astype(int)
    #                 trajs[0, tgt_frames, :] = tgt_seq[:, 2:4]
    #                 if np.any(np.isnan(trajs[0, (self.args.obs_len - self.min_obs_len):, 0])):
    #                    continue
    #                 if is_target_outbound(trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
    #                     print("target is outbound")
    #                     continue
    #                 nbr_pids = np.delete(pids, np.where(pids == tgt_pid))
    #                 for nbr_pidx, nbr_pid in enumerate(nbr_pids):
    #                     nbr_seq = seq[seq[:, 0] == nbr_pid, :]
    #                     nbr_frames = (nbr_seq[:, 1] - start_frame) // frame_gap
    #                     nbr_frames = nbr_frames.astype(int)
    #                     # if len(nbr_frames[nbr_frames < self.args.obs_len]) < 4:
    #                     #     continue
    #                     # if max(nbr_frames) > self.seq_len:
    #                     #     print(seq, start_frame, nbr_frames, nbr_seq[:, 0], seq[0, 0], seq[-1, 0])
    #                     trajs[nbr_pidx+1, nbr_frames, :] = nbr_seq[:, 2:4]
    #                 # NaN nbr removal
    #                 traj_valid = [False if np.all(np.isnan(traj[:self.args.obs_len, 0])) else True for traj in trajs]
    #                 trajs = trajs[traj_valid]
    #                 all_trajs.append(trajs)
    #                 all_scenes.append(sid)
    #         return all_trajs, all_scenes

    # def image2world(self, image_coords, scene, homo_mat):
    #     """
    #     Transform trajectories of one scene from image_coordinates to world_coordinates
    #     :param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
    #     :param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
    #     :param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
    #     :param resize: float, resize factor
    #     :return: trajectories in world_coordinates
    #     """
    #     traj_image2world = copy.deepcopy(image_coords) # .clone()
    #     if scene in ['eth', 'hotel']:
    #         # eth and hotel have different coordinate system than ucy data
    #         traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
    #     traj_image2world = traj_image2world.reshape(-1, 3)
    #     traj_image2world = np.matmul(homo_mat[scene], traj_image2world.T).T
    #     traj_image2world = traj_image2world / traj_image2world[:, 2:]
    #     traj_image2world = traj_image2world[:, :2]
    #     traj_image2world = traj_image2world.view_as(image_coords)
    #     return traj_image2world

    # num_scene += 1

    # self.scales[scene_id] = 1.0
    # scenes = [scene_id] * num_scene
    # self.all_scenes += scenes

    # for scene in df_data.sceneId.unique():
    #     img_path = os.path.join(self.path, scene, 'oracle.png')
    #     homo_path = os.path.join(self.path, scene + '_H.txt')
    #     homo_mat = np.loadtxt(homo_path)
    #     df_scene = df_data[df_data.sceneId == scene] # scene data
    #     scene_trajs = self.generate_scenes(df_scene, scene, homo_mat, frame_gap=10)
    #     self.all_trajs += scene_trajs
    #     if self.args.scene:
    #         scenes = [scene] * len(scene_trajs)
    #         self.all_scenes += scenes
    #         map = cv2.imread(img_path, 0)
    #         corner_uv = np.array([[0, 0], [map.shape[1], 0],[0, map.shape[0]], [map.shape[1], map.shape[0]]])
    #         corner_xy = self.image2world(corner_uv, homo_mat)
    #         min_x, max_x = min(corner_xy[:, 0]), max(corner_xy[:, 0])
    #         min_y, max_y = min(corner_xy[:, 1]), max(corner_xy[:, 1])
    #         width = max_x - min_x
    #         height = max_y - min_y
    #         cntr_x = (max_x + min_x) / 2.0
    #         cntr_y = (max_y + min_y) / 2.0
    #         occupied_idxs = np.where(map != 1)
    #         occupied_idxs = np.stack(occupied_idxs)
    #         occupied_xys = self.image2world(occupied_idxs.T, homo_mat)
    #         num_rows = int(height / args.env_resol)
    #         num_cols = int(width / args.env_resol)
    #         num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    #         num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    #         grid_map = RectangularGridMap(width=num_cols, height=num_rows, resolution=args.env_resol, center_x=cntr_x, center_y=cntr_y)
    #         grid_map.set_value_from_xy_pos(occupied_xys[:, 0], occupied_xys[:, 1], 1.0)
    #         self.envs[scene] = grid_map

    # if self.split == 'test' and self.args.viz and len(self.all_trajs) > 100:
    #     break