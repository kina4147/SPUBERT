import random
import os
import numpy as np
import pickle
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset

DATASET_NAME_TO_NUM = {
    'eth': 0,
    'hotel': 1,
    'zara1': 2,
    'zara2': 3,
    'univ': 4
}
class ETHUCYSTARDataset(SocialBERTDataset):
    # def __init__(self, path=None, name=None, pretrain=False, subset='train', config=None):
    def __init__(self, split, args):
        super().__init__(split, args)
        data_dirs = ['eth', 'hotel', 'zara1', 'zara2', 'univ/students001', 'univ/students003', 'uni_examples', 'zara3']
        self.data_dirs = []
        for d in data_dirs:
            self.data_dirs.append(os.path.join(self.path, d))
        # Data directory where the pre-processed pickle file resides
        skip = [6, 10, 10, 10, 10, 10, 10, 10]

        train_set = [i for i in range(len(self.data_dirs))]
        # assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)
        name = DATASET_NAME_TO_NUM[args.dataset_split]
        if name == 4 or name == 5:
            self.test_set = [4, 5]
        else:
            self.test_set = [name]
        for x in self.test_set:
            train_set.remove(x)

        self.train_dir = [self.data_dirs[x] for x in train_set]
        self.test_dir = [self.data_dirs[x] for x in self.test_set]
        self.trainskip = [skip[x] for x in train_set]
        self.testskip = [skip[x] for x in self.test_set]

        self.traject_preprocess(self.split)
        self.all_trajs = self.dataPreprocess(self.split)
        # self.all_trajs = self.all_trajs[:100]
        self.all_scenes = [0]*len(self.all_trajs)
        self.scales = [1.0]




    def traject_preprocess(self, setname):
        '''
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        if setname == 'train':
            data_dirs = self.train_dir
        else:
            data_dirs = self.test_dir

        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []

        Pedlist_data = []
        self.frameped_dict = []  # peds id contained in a certain frame
        self.pedtraject_dict = []  # trajectories of a certain ped
        # For each dataset
        for seti, directory in enumerate(data_dirs):
            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            # Frame IDs of the frames in the current dataset
            Pedlist = np.unique(data[1, :]).tolist()
            # numPeds = len(Pedlist)
            # Add the list of frameIDs to the frameList_data
            Pedlist_data.append(Pedlist)
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])
            numFrame_data.append([])
            self.frameped_dict.append({})
            self.pedtraject_dict.append({})

            for ind, pedi in enumerate(Pedlist):
                # if ind % 100 == 0:
                #     print(ind, len(Pedlist))
                # Extract trajectories of one person
                FrameContainPed = data[:, data[1, :] == pedi]
                # Extract peds list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2: #
                    continue
                # Add number of frames of this trajectory
                numFrame_data[seti].append(len(FrameList))
                # Initialize the row of the numpy array
                Trajectories = []
                # For each ped in the current frame
                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]
                    # Add their pedID, x, y to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y])
                    if int(frame) not in self.frameped_dict[seti]:
                        self.frameped_dict[seti][int(frame)] = []
                    self.frameped_dict[seti][int(frame)].append(pedi)
                self.pedtraject_dict[seti][pedi] = np.array(Trajectories)


    def get_data_index(self, data_dict, setname, ifshuffle=True):
        '''
        Get the dataset sampling index.
        '''
        set_id = []
        frame_id_in_set = []
        total_frame = 0
        for seti, dict in enumerate(data_dict):
            frames = sorted(dict)
            maxframe = max(frames) - self.seq_len
            frames = [x for x in frames if not x > maxframe]
            total_frame += len(frames)
            set_id.extend(list(seti for i in range(len(frames))))
            frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))

        all_frame_id_list = list(i for i in range(total_frame))

        data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int),
                                     np.array([all_frame_id_list], dtype=int)), 0)
        if ifshuffle:
            random.Random().shuffle(all_frame_id_list)
        data_index = data_index[:, all_frame_id_list]

        # to make full use of the data # 뒷부분이 모자라서 짤리지 않도록 덧붙여주는 작업
        if setname == 'train':
            data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
        return data_index

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]
        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        data_index = self.get_data_index(self.frameped_dict, setname, ifshuffle=shuffle)
        # val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        # train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]
        trajectories = self.get_seq_from_index_balance(self.frameped_dict, self.pedtraject_dict, data_index, setname)
        # val_all_trajectories = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, val_index, setname)

        return trajectories

    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        # batch_data_mass = []
        # batch_data = []
        # Batch_id = []

        if setname == 'train':
            skip = self.trainskip
        else:
            skip = self.testskip
        #
        # ped_cnt = 0
        # last_frame = 0
        all_trajectories = []
        for i in range(data_index.shape[1]):
            # if i % 100 == 0:
            #     print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])
            try:
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + self.seq_len * skip[cur_set]])
            except:
                continue

            # print(framestart_pedi, frameend_pedi)
            # assert False
            present_pedi = framestart_pedi | frameend_pedi # 교집합
            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue
            traject = ()
            IFfull = []
            for ped in present_pedi:
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped],
                                                                               cur_frame,
                                                                               self.seq_len, skip[cur_set])
                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data
                    continue
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 2),)
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)
            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue

            traject_batch = np.concatenate(traject, 1)
            traject_batch[traject_batch == 0] = np.nan
            traject_batch = traject_batch.transpose(1, 0, 2)
            num_peds = len(traject_batch)
            for t_pidx in range(num_peds):
                trajectories = np.copy(traject_batch)
                if np.any(np.isnan(traject_batch[t_pidx])):
                    continue
                trajectories[0], trajectories[t_pidx] = traject_batch[t_pidx], traject_batch[0]
                all_trajectories.append(trajectories)
        return all_trajectories

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
        '''
        Query the trajectory fragment based on the index. Replace where data isn't exsist with 0.
        '''
        return_trajec = np.zeros((seq_length, 3))
        endframe = startframe + (seq_length) * skip
        start_n = np.where(trajectory[:, 0] == startframe)
        end_n = np.where(trajectory[:, 0] == endframe)
        iffull = False
        ifexsitobs = False
        if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:
            start_n = 0
            end_n = end_n[0][0]
            if end_n == 0:
                return return_trajec, iffull, ifexsitobs

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
            start_n = start_n[0][0]
            end_n = trajectory.shape[0]

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
            start_n = 0
            end_n = trajectory.shape[0]
        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]

        candidate_seq = trajectory[start_n:end_n]
        offset_start = int((candidate_seq[0, 0] - startframe) // skip)
        offset_end = self.seq_len + int((candidate_seq[-1, 0] - endframe) // skip)

        return_trajec[offset_start:offset_end + 1, :3] = candidate_seq

        if return_trajec[self.args.obs_len - 1, 1] != 0:
            ifexsitobs = True

        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs
