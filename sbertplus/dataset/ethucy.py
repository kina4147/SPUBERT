

import os
import pandas as pd
import argparse
import yaml
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
from SocialBERT.sbertplus.dataset.viz import *
from SocialBERT.sbertplus.dataset.grid_map_numpy import RectangularGridMap

cross_validation = {
    'eth': ['hotel', 'univ', 'zara01', 'zara02'],
    'hotel': ['eth', 'univ', 'zara01', 'zara02'],
    'univ': ['eth', 'hotel', 'zara01', 'zara02'],
    'zara1': ['eth', 'hotel', 'univ', 'zara02'],
    'zara2': ['eth', 'hotel', 'univ', 'zara01'],
}

# STAR
class ETHUCYDataset(SocialBERTDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        csv_columns = ["frame", "id", "px_x", "px_y"]

        # if args.mode == "pretrain" or args.mode =="finetune":
        #     self.train = True
        # else:
        #     self.train = False
        # self.train = True if args.mode == "pretrain" or args.mode =="finetune" else False
        self.min_obs_len = self.args.min_obs_len if self.split == 'train' else self.args.obs_len


        with open(os.path.join(self.path, 'splits.yml'), 'r') as f:
            cross_validation = yaml.load(f, Loader=yaml.FullLoader)

        for split in cross_validation[args.dataset_split][self.split]:
            # split_path = os.path.join(self.path, split)
            # files = os.listdir(split_path)
            # traj_files = [file for file in files if file.endswith(".txt")]
            traj_filepath = os.path.join(self.path, split, split + '.txt')
            # for traj_file in traj_files:
            #     traj_filepath = os.path.join(split_path, traj_file)
            raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
            # img_filepath = traj_filepath.replace('.txt', '.png')
            # img = pilimg.open(img_filepath)
            trajs = self.generate_scenes(raw_data=raw_data, frame_gap=10)
            env_filepath = traj_filepath.replace('.txt', '_map_xy.txt')
            env = np.loadtxt(env_filepath)
            env_corners_filepath = traj_filepath.replace('.txt', '_map_corner_xy.txt')
            env_corner = np.loadtxt(env_corners_filepath)
            # env_resol = 0.1 # 10cm
            num_rows = env_corner[1] / self.args.env_resol
            num_cols = env_corner[0] / self.args.env_resol
            num_rows = int(num_rows) if (num_rows % 2) == 0 else int(num_rows+1)
            num_cols = int(num_cols) if (num_cols % 2) == 0 else int(num_cols+1)
            grid_map = RectangularGridMap(width=num_cols, height=num_rows, resolution=self.args.env_resol, center_x=env_corner[2], center_y=env_corner[3])
            if len(env) > 1:
                grid_map.set_value_from_xy_pos(env[:, 0], env[:, 1], 1.0)
            scenes = [split] * len(trajs)
            self.all_trajs += trajs
            self.all_scenes += scenes
            self.envs[split] = grid_map
            # if len(self.all_trajs) > 1000:
            #     break

    # for split in cross_validation[args.dataset_split][args.mode]:
    #     split_path = os.path.join(self.path, split)
    #     files = os.listdir(split_path)
    #     traj_files = [file for file in files if file.endswith("_px.txt")]
    #     for traj_file in traj_files:
    #         traj_filepath = os.path.join(split_path, traj_file)
    #         raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
    #         img_filepath = traj_filepath.replace('_px.txt', '.png')
    #         img = pilimg.open(img_filepath)
    #         trajs = self.generate_scenes(raw_data=raw_data, frame_gap=10)
    #         viz_scene(trajs=trajs, bg_img=img)
    #         scene_name = traj_file.replace("_px.txt", "")
    #         scenes = [scene_name]*len(trajs)
    #         self.all_trajs += trajs
    #         self.all_scenes += scenes
    #         self.scene_images[scene_name] = np.array(img)
    # on image (pixel)
    # for split in cross_validation[args.dataset_split][args.mode]:
    #     split_path = os.path.join(self.path, split)
    #     files = os.listdir(split_path)
    #     traj_files = [file for file in files if file.endswith("_px.txt")]
    #     for traj_file in traj_files:
    #         traj_filepath = os.path.join(split_path, traj_file)
    #         raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
    #         img_filepath = traj_filepath.replace('_px.txt', '.png')
    #         img = pilimg.open(img_filepath)
    #         trajs = self.generate_scenes(raw_data=raw_data, frame_gap=10)
    #
    #         map_filepath = traj_filepath.replace('_px.txt', '_map.png')
    #         map = pilimg.open(map_filepath)
    #         # viz_scene(trajs=trajs, bg_img=map)
    #         scene_name = traj_file.replace("_px.txt", "")
    #         scenes = [scene_name] * len(trajs)
    #         self.all_trajs += trajs
    #         self.all_scenes += scenes
    #         self.scene_images[scene_name] = np.array(img)

    # on map (position)


if __name__ == '__main__':
    print("ETH/UCY Datasets.")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.obs_len = 8
    args.pred_len = 12
    args.dataset_path = './data'
    args.dataset_name = 'ethucy'
    args.dataset_split = 'eth'
    args.mode = 'test'
    args.num_nbr = 2
    args.view_range = 20.0
    args.view_angle = np.pi/3
    args.social_range = 1
    args.aug = True
    args.patch_size = 32
    args.env_range = 10
    args.env_resol = 0.1
    args.scene = "rgrid_map"
    cfgs = 0
    dataset = ETHUCYDataset(False, args, cfgs)
    for data in dataset:
        print("a")





#         data_dirs = ['eth', 'hotel', 'zara1', 'zara2', 'univ/students001', 'univ/students003', 'uni_examples', 'zara3']
#         for d in data_dirs:
#             self.data_dirs.append(os.path.join(self.path, d))
#         skip = [6, 10, 10, 10, 10, 10, 10, 10]
#         train_set = [i for i in range(len(self.data_dirs))]
#         name = DATASET_NAME_TO_NUM[name]
#
#         if name == 4 or name == 5:
#             self.test_set = [4, 5]
#         else:
#             self.test_set = [name]
#
#         for x in self.test_set:
#             train_set.remove(x)
#
#         self.train_dir = [self.data_dirs[x] for x in train_set]
#         self.trainskip = [skip[x] for x in train_set]
#
#         self.test_dir = [self.data_dirs[x] for x in self.test_set]
#         self.testskip = [skip[x] for x in self.test_set]
#         print("Creating pre-processed data from raw data.")
#         self.traject_preprocess(subset)
#         print("Done.")
#         print("Preparing data batches.")
#         self.trajs = self.dataPreprocess(subset)
#
#     def preprocess(self):
#
#         all_frame_data = []
#         valid_frame_data = []
#         numFrame_data = []
#         Pedlist_data = []
#         self.frameped_dict = []  # peds id contained in a certain frame
#         self.pedtraject_dict = []  # trajectories of a certain ped
#         # For each dataset
#         for seti, directory in enumerate(data_dirs):
#             file_path = os.path.join(directory, 'true_pos_.csv')
#             # Load the data from the csv file
#             data = np.genfromtxt(file_path, delimiter=',')
#             # Frame IDs of the frames in the current dataset
#             Pedlist = np.unique(data[1, :]).tolist()
#             # numPeds = len(Pedlist)
#             # Add the list of frameIDs to the frameList_data
#             Pedlist_data.append(Pedlist)
#             # Initialize the list of numpy arrays for the current dataset
#             all_frame_data.append([])
#             # Initialize the list of numpy arrays for the current dataset
#             valid_frame_data.append([])
#             numFrame_data.append([])
#             self.frameped_dict.append({})
#             self.pedtraject_dict.append({})
#
#             for ind, pedi in enumerate(Pedlist):
#                 # if ind % 100 == 0:
#                 #     print(ind, len(Pedlist))
#                 # Extract trajectories of one person
#                 FrameContainPed = data[:, data[1, :] == pedi]
#                 # Extract peds list
#                 FrameList = FrameContainPed[0, :].tolist()
#                 if len(FrameList) < 2:  #
#                     continue
#                 # Add number of frames of this trajectory
#                 numFrame_data[seti].append(len(FrameList))
#                 # Initialize the row of the numpy array
#                 Trajectories = []
#                 # For each ped in the current frame
#                 for fi, frame in enumerate(FrameList):
#                     # Extract their x and y positions
#                     current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
#                     current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
#                     # Add their pedID, x, y to the row of the numpy array
#                     Trajectories.append([int(frame), current_x, current_y])
#                     if int(frame) not in self.frameped_dict[seti]:
#                         self.frameped_dict[seti][int(frame)] = []
#                     self.frameped_dict[seti][int(frame)].append(pedi)
#                 self.pedtraject_dict[seti][pedi] = np.array(Trajectories)
#
#     def convert(self):
#         if self.split != 'train':
#             shuffle = False
#         else:
#             shuffle = True
#         data_index = self.get_data_index(self.frameped_dict, self.split, ifshuffle=shuffle)
#         # val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
#         # train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]
#         self.trajs = self.get_seq_from_index_balance(self.frameped_dict, self.pedtraject_dict, data_index, setname)
#
#     def get_data_index(self, data_dict, setname, ifshuffle=True):
#         '''
#         Get the dataset sampling index.
#         '''
#         set_id = []
#         frame_id_in_set = []
#         total_frame = 0
#         for seti, dict in enumerate(data_dict):
#             frames = sorted(dict)
#             maxframe = max(frames) - self.sum_length
#             frames = [x for x in frames if not x > maxframe]
#             total_frame += len(frames)
#             set_id.extend(list(seti for i in range(len(frames))))
#             frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))
#
#         all_frame_id_list = list(i for i in range(total_frame))
#
#         data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
#                                      np.array([all_frame_id_list], dtype=int)), 0)
#         if ifshuffle:
#             random.Random().shuffle(all_frame_id_list)
#         data_index = data_index[:, all_frame_id_list]
#
#         # to make full use of the data # 뒷부분이 모자라서 짤리지 않도록 덧붙여주는 작업
#         if setname == 'train':
#             data_index = np.append(data_index, data_index[:, :self.config.batch_size], 1)
#         return data_index
#
#     def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
#         '''
#         Query the trajectories fragments from data sampling index.
#         Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
#                This function takes less gpu memory.
#         '''
#         batch_data_mass = []
#         batch_data = []
#         Batch_id = []
#
#         # temp = self.batch_around_ped
#         if setname == 'train':
#             skip = self.trainskip
#         else:
#             skip = self.testskip
#
#         ped_cnt = 0
#         last_frame = 0
#         all_trajectories = []
#         for i in range(data_index.shape[1]):
#             # if i % 100 == 0:
#             #     print(i, '/', data_index.shape[1])
#             cur_frame, cur_set, _ = data_index[:, i]
#             framestart_pedi = set(frameped_dict[cur_set][cur_frame])
#             try:
#                 frameend_pedi = set(frameped_dict[cur_set][cur_frame + self.sum_length * skip[cur_set]])
#             except:
#                 continue
#
#             # print(framestart_pedi, frameend_pedi)
#             # assert False
#             present_pedi = framestart_pedi | frameend_pedi # 교집합
#             if (framestart_pedi & frameend_pedi).__len__() == 0:
#                 continue
#             traject = ()
#             IFfull = []
#             for ped in present_pedi:
#                 cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
#                                                                                cur_frame,
#                                                                                self.sum_length, skip[cur_set])
#                 if len(cur_trajec) == 0:
#                     continue
#                 if ifexistobs == False:
#                     # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
#                     continue
#                 if sum(cur_trajec[:, 0] > 0) < 5:
#                     # filter trajectories have too few frame data
#                     continue
#                 cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
#                 traject = traject.__add__(cur_trajec)
#                 IFfull.append(iffull)
#             if traject.__len__() < 1:
#                 continue
#             if sum(IFfull) < 1:
#                 continue
#
#             traject_batch = np.concatenate(traject, 1)
#             traject_batch[traject_batch == 0] = np.nan
#             traject_batch = to_trajectory(traject_batch)
#             num_peds = len(traject_batch)
#             for t_pidx in range(num_peds):
#                 trajectories = np.copy(traject_batch)
#                 if np.any(np.isnan(traject_batch[t_pidx])):
#                     continue
#                 trajectories[0], trajectories[t_pidx] = traject_batch[t_pidx], traject_batch[0]
#                 all_trajectories.append(trajectories)
#         return all_trajectories
#
#     def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
#         '''
#         Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
#         '''
#         return_trajec = np.zeros((seq_length, 3))
#         endframe = startframe + (seq_length) * skip
#         start_n = np.where(trajectory[:, 0] == startframe)
#         end_n = np.where(trajectory[:, 0] == endframe)
#         iffull = False
#         ifexsitobs = False
#         if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
#             start_n = 0
#             end_n = end_n[0][0]
#             if end_n == 0:
#                 return return_trajec, iffull, ifexsitobs
#
#         elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
#             start_n = start_n[0][0]
#             end_n = trajectory.shape[0]
#
#         elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
#             start_n = 0
#             end_n = trajectory.shape[0]
#         else:
#             end_n = end_n[0][0]
#             start_n = start_n[0][0]
#
#         candidate_seq = trajectory[start_n:end_n]
#         offset_start = int((candidate_seq[0, 0] - startframe) // skip)
#         offset_end = self.sum_length + int((candidate_seq[-1, 0] - endframe) // skip)
#
#         return_trajec[offset_start:offset_end + 1, :3] = candidate_seq
#
#         if return_trajec[self.config.obs_length - 1, 1] != 0:
#             ifexsitobs = True
#
#         if offset_end - offset_start >= seq_length - 1:
#             iffull = True
#
#         return return_trajec, iffull, ifexsitobs
#
# # TF
# class ETHUCYDataset(SocialBERTDataset):
#     def __init__(self, args, cfgs):
#         super().__init__(args, cfgs)
#         trajs = self.load()
#         for trajs in trajs:
#             if len(trajs[0]) < self.seq_len:
#                 print("The number of positions in target trajectory is not enough.")
#                 continue
#             if target_in_bound(trajs, self.args.obs_length, self.args.spatial_range):
#                 self.trajs.append(trajs)
#
#
#     def load(self):
#         dataset_path = os.path.join(self.path, subset)
#         dataset = os.listdir(dataset_path)
#         sum_length = obs_length + pred_length
#
#         step = 1
#         all_data = tqdm.tqdm(enumerate(dataset), total=len(dataset))
#         all_trajectories = []
#         for idx, data in all_data:
#             raw_data = pd.read_csv(os.path.join(dataset_path, data), sep='\t', index_col=False, header=None)
#             raw_data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
#             raw_data.sort_values(by=['frame_id', 'track_id'], inplace=True)
#             # frame * ped * spatial
#             frame_ids = raw_data.frame_id.unique().tolist()
#             frames = []
#             start_frames = []
#             start_frame_ids = []
#             for i in range(1 + (len(frame_ids) - seq_len) // step):
#                 frames.append(np.array(raw_data[raw_data.frame_id == frame_ids[i]]))
#                 start_frames.append(frame_ids[i])
#                 start_frame_ids.append(i)
#
#             # num_sequences = len(frame_data)
#             # frame, ped, x, y
#             frame_iter = tqdm.tqdm(zip(start_frames, start_frame_ids, frames), total=len(start_frames))
#             # frames => id
#             for start_frame, start_frame_id, frame in frame_iter:
#                 seq_data = np.concatenate(frames[start_frame_id:start_frame_id + seq_len], axis=0)
#                 ped_ids = np.unique(seq_data[:, 1])
#                 for t_pidx, t_ped_id in enumerate(ped_ids):
#                     trajectories = np.full((len(ped_ids), seq_len, 2), np.nan)  # nan?
#                     t_seq_data = seq_data[seq_data[:, 1] == t_ped_id, :]
#                     if len(t_seq_data) != seq_len:  # full trajectory for target
#                         # print("Target pedestrian trajectory are not enough")
#                         continue
#                     trajectories[0, :, :] = t_seq_data[:, 2:4]
#                     frame_gap = t_seq_data[1, 0] - t_seq_data[0, 0]
#                     n_ped_ids = np.delete(ped_ids, np.where(ped_ids == t_ped_id))
#                     for n_pidx, n_ped_id in enumerate(n_ped_ids):
#                         n_seq_data = seq_data[seq_data[:, 1] == n_ped_id, :]
#                         n_frame_ids = (n_seq_data[:, 0] - start_frame) // frame_gap
#                         n_frame_ids = n_frame_ids.astype(int)
#                         trajectories[n_pidx + 1, n_frame_ids, :] = n_seq_data[:, 2:4]
#                     all_trajectories.append(trajectories)
#         return all_trajectories
#
#     def convert(self):
#         self.path