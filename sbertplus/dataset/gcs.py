import os
import pandas as pd
import argparse
import PIL.Image as pilimg
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
from SocialBERT.sbertplus.dataset.viz import *

# dataset = ['bookstore_0', 'bookstore_1', 'bookstore_2', 'bookstore_3',
#            'coupa_3',
#            'deathCircle_0', 'deathCircle_1', 'deathCircle_2', 'deathCircle_3', 'deathCircle_4',
#            'gates_0', 'gates_1', 'gates_3', 'gates_4', 'gates_5', 'gates_6', 'gates_7', 'gates_8',
#            'hyang_4', 'hyang_5', 'hyang_6', 'hyang_7', 'hyang_9',
#            'nexus_0', 'nexus_1', 'nexus_2', 'nexus_3', 'nexus_4', 'nexus_7', 'nexus_8', 'nexus_9']



# k-fold cross-validation
# leave-one-out cross-validation
# cross_validation = {
#
#
# }

'''
-------------- px_x
|
|
| px_y
'''
# cross_validation = {
#     'bookstore': ['coupa', 'deathCircle', 'gates', 'hyang', 'nexus'],
#     'deathCircle': [],
#     'gates': [],
#     'hyang': [],
#     'nexus': []
# }
class GCSDataset(SocialBERTDataset):
    def __init__(self, args, cfgs):
        super().__init__(args, cfgs)
        csv_columns = ["frame", "id", "px_x", "px_y"]
        self.all_trajs = []
        self.all_scenes = []
        self.scene_images = {}

        traj_filepath = os.path.join(self.path, 'gcs_px.txt')
        raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
        img_filepath = traj_filepath.replace('_px.txt', '.jpg')
        img = pilimg.open(img_filepath)
        trajs = self.generate_scenes(raw_data=raw_data, frame_gap=20)
        viz_scene_seq(trajs, img)

        # scene_name = traj_file.replace(".txt", "")
        # scenes = [scene_name]*len(trajs)
        # self.all_trajs += trajs
        # self.all_scenes += scenes
        # self.scene_images[scene_name] = np.array(img)
        #
        # if args.mode == 'train':
        #     for split in cross_validation[args.dataset_split]:
        #         split_path = os.path.join(self.path, split)
        #         files = os.listdir(split_path)
        #         traj_files = [file for file in files if file.endswith(".txt")]
        #         for traj_file in traj_files:
        #             traj_filepath = os.path.join(split_path, traj_file)
        #             raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
        #             img_filepath = traj_filepath.replace('.txt', '.jpg')
        #             img = pilimg.open(img_filepath)
        #             trajs = self.generate_scenes(raw_data=raw_data, frame_gap=12)
        #             # viz_scene(trajs=trajs, bg_img=img)
        #             scene_name = traj_file.replace(".txt", "")
        #             scenes = [scene_name]*len(trajs)
        #             self.all_trajs += trajs
        #             self.all_scenes += scenes
        #             self.scene_images[scene_name] = np.array(img)
        # elif args.mode == 'test':
        #     split_path = os.path.join(self.path, args.dataset_split)
        #     files = os.listdir(split_path)
        #     traj_files = [file for file in files if file.endswith(".txt")]
        #     for traj_file in traj_files:
        #         traj_filepath = os.path.join(split_path, traj_file)
        #         raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
        #         img_filepath = traj_filepath.replace('.txt', '.jpg')
        #         img = pilimg.open(img_filepath)
        #         trajs = self.generate_scenes(raw_data=raw_data, frame_gap=12)
        #         # viz_scene(trajs=trajs, bg_img=img)
        #         scene_name = traj_file.replace(".txt", "")
        #         scenes = [scene_name]*len(trajs)
        #         self.all_trajs += trajs
        #         self.all_scenes += scenes
        #         self.scene_images[scene_name] = np.array(img)
        #
        # else:
        #     pass
        # print(len(self.all_trajs), self.all_scenes, self.scene_images.keys())


    # def convert(self):
    #     pass




if __name__ == '__main__':
    print("New York Grand Central Station Datasets.")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.obs_len = 8
    args.pred_len = 12
    args.dataset_path = './data'
    args.dataset_name = 'gcs'
    args.dataset_split = 'bookstore'
    args.mode = 'test'
    cfgs = 0
    dataset = GCSDataset(args, cfgs)