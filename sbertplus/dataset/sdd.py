import os
import pandas as pd
import argparse
import PIL.Image as pilimg
import yaml
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
from SocialBERT.sbertplus.dataset.viz import *

dataset = ['bookstore_0', 'bookstore_1', 'bookstore_2', 'bookstore_3',
           'coupa_3',
           'deathCircle_0', 'deathCircle_1', 'deathCircle_2', 'deathCircle_3', 'deathCircle_4',
           'gates_0', 'gates_1', 'gates_3', 'gates_4', 'gates_5', 'gates_6', 'gates_7', 'gates_8',
           'hyang_4', 'hyang_5', 'hyang_6', 'hyang_7', 'hyang_9',
           'nexus_0', 'nexus_1', 'nexus_2', 'nexus_3', 'nexus_4', 'nexus_7', 'nexus_8', 'nexus_9']



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
#     'train': [
#         'bookstore_0', 'bookstore_1', 'bookstore_2','bookstore_4', 'bookstore_5',
#         'nexus_0', 'nexus_1', 'nexus_2', 'nexus_5', 'nexus_6', 'nexus_8', 'nexus_9', 'nexus_10', 'nexus_11',
#         'hyang_0','hyang_1', 'hyang_5', 'hyang_8', 'hyang_10', 'hyang_12', 'hyang_14',
#         'coupa_0', 'coupa_2','coupa_3',
#         'deathCircle_0', 'deathCircle_1', 'deathCircle_2', 'deathCircle_3',
#         'gates_0','gates_2','gates_5','gates_6',
#         'quad_0','quad_1','quad_3',
#         'little_0'],
#     'test': []
# }
# cross_validation = {
#     'bookstore': ['coupa', 'deathCircle', 'gates', 'hyang', 'nexus'],
#     'deathCircle': [],
#     'gates': [],
#     'hyang': [],
#     'nexus': []
# }
class SDDDataset(SocialBERTDataset):
    def __init__(self, args, cfgs):
        super().__init__(args, cfgs)
        csv_columns = ["frame", "id", "px_x", "px_y"]
        with open(os.path.join(self.path, 'splits.yml'), 'r') as f:
            cross_validation = yaml.load(f, Loader=yaml.FullLoader)
        print(cross_validation[args.dataset_split][args.mode])
        for split in cross_validation[args.dataset_split][args.mode]:
            # split_path = os.path.join(self.path, split + '.txt')
            # files = os.listdir(split_path)
            # traj_files = [file for file in files if file.endswith(".txt")]
            traj_filepath = os.path.join(self.path, split + '.txt')
            # for traj_file in traj_files:
            #     traj_filepath = os.path.join(split_path, traj_file)
            raw_data = pd.read_csv(traj_filepath, sep=" ", header=None, names=csv_columns)
            img_filepath = traj_filepath.replace('.txt', '.jpg')
            img = pilimg.open(img_filepath)
            trajs = self.generate_scenes(raw_data=raw_data, frame_gap=12)
            # viz_scene(trajs=trajs, bg_img=img)
            scene_name = split # traj_file.replace(".txt", "")
            scenes = [scene_name]*len(trajs)
            self.all_trajs += trajs
            self.all_scenes += scenes
            self.scene_images[scene_name] = np.array(img)


        else:
            pass
        print(len(self.all_trajs), self.all_scenes, self.scene_images.keys())


    # def convert(self):
    #     pass




if __name__ == '__main__':
    print("Stanford Drone Datasets.")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.obs_len = 8
    args.pred_len = 12
    args.dataset_path = './data'
    args.dataset_name = 'sdd'
    args.dataset_split = 'fold2'
    args.mode = 'test'
    cfgs = 0
    dataset = SDDDataset(args, cfgs)