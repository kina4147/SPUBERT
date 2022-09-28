

from SocialBERT.sbertplus.dataset.jaad_data import JAAD
from SocialBERT.sbertplus.dataset.pie_data import PIE
from SocialBERT.sbertplus.dataset.dataset import SocialBERTDataset
import argparse



class JAADPIEDataset(SocialBERTDataset):
    def __init__(self, args, cfgs):
        super().__init__(args, cfgs)
        if args.dataset_split == 'jaad':
            imdb = JAAD(data_path=self.path)
            opts = {'fstride': 1,
                    # 'sample_type': 'all',
                    # 'height_rng': [0, float('inf')],
                    # 'squarify_ratio': 0,
                    # 'data_split_type': 'default',  # kfold, random, default
                    'seq_type': 'trajectory',
                    'min_track_size': 61,
                    'random_params': {'ratios': None,
                                      'val_data': True,
                                      'regen_data': True},
                    'kfold_params': {'num_folds': 5, 'fold': 1}}

        elif args.dataset_split == 'pie':
            imdb = PIE(data_path=self.path)
            opts = {'fstride': 1,
                    # 'sample_type': 'all',
                    # 'height_rng': [0, float('inf')],
                    # 'squarify_ratio': 0,
                    # 'data_split_type': 'default',  # kfold, random, default
                    'seq_type': 'trajectory',
                    'min_track_size': 61,
                    'random_params': {'ratios': None,
                                      'val_data': True,
                                      'regen_data': True},
                    'kfold_params': {'num_folds': 5, 'fold': 1}}
        else:
            pass

        data = imdb.generate_data_trajectory_sequence(image_set=args.mode)
        self.trajs = self.convert(data)

    def convert(self, data): # Formatting
        # self.args.obs_len
        # self.args.pred_len
        # self.seq_len


        for imgs, pids, pos in zip(data['image'], data['pid'], data['center']):
            print(len(imgs), len(pids), len(pos))
            for img, pid, pos in zip(imgs, pids, pos):
                print(img)
                print(pid)
                print(pos) # pixel-wise position


    def __getitem__(self, item):

        return output


if __name__ == '__main__':
    print("JAAD & PIE Dataset.")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = './data'
    args.dataset_name = 'jaad'
    args.dataset_split = 'jaad'
    args.mode = 'test'
    cfgs = 0
    dataset = JAADPIEDataset(args, cfgs)