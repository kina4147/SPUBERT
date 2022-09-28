
'''
common
. obs_len
. pred_len
. num_nbr
. view_range
. view_angle
. social_range
. env_range
. env_resol
. patch_size
. scene: rgrid_map, cgrid_map, ray_cast
. embedding type: concat, sum
. social interaction prediction (SIP)
. environment interaction prediction (EIP)

. traj_enc_hidden
. traj_enc_layer
. traj_enc_head


pretrainer
. pretrain method: SIP, EIP, Random Mask Prediction (RMP), Trajectory To Goal Prediction (TGP)
. RMP (MTP): add num_mask or prob_mask
. TGP:


trainer
. goal
. goal_enc_hidden
. goal_enc_layer
. goal_enc_head
. goal_hidden
. goal_latent
. k_sample

'''
# from typing import Optional
# import easydict import EasyDict
from collections import OrderedDict
import os
import glob
import argparse
import yaml
import numpy as np
# arrange files
# num_nbr_4
# obs_len_8
# OrderDict
# level1:
# level2:


# CODIFY
# DICT / LIST / STRING / INT / FLOAT


################# Level Definition #################
DATASET = ["dataset_name"] # DATASET
SPLIT = ["dataset_split"] # SPLIT
MODE = ["mode"] # train_mode # pretrain # finetune
DATA_REP = ["num_nbr", "view_range", "view_angle", "social_range"] # COMMON
SBERT_ENC = ["hidden", "layer", "head"] # TRAJ
PRETRAIN_PARAM = ['aug', 'seed', 'sip']
FINETUNE_PARAM = ['aug', 'seed', 'train_mode']

PRETRAIN_CFG = [DATASET, SPLIT, MODE, DATA_REP, SBERT_ENC, PRETRAIN_PARAM]
FINETUNE_CFG = [DATASET, SPLIT, MODE, DATA_REP, SBERT_ENC, FINETUNE_PARAM]

################# Option Selection #################
opt_conds = {
    'num_nbr': [2, 4, 6, 8, 0],
    'view_range': [10, 15, 20],
    'view_angle': [1.57, 2.09, 3.14],
    'social_range': [1.0, 1.5, 2.0, 0.0],
    'act_fn': ['gelu', 'relu'],

    'hidden': [64, 128, 256, 512],
    'layer': [4, 6],
    'head': [4, 6, 8],
    'min_obs_len': [1, 2, 3, 4, 5, 6, 7, 8],
    'input_dim': [2, 4, 6],

    ## LV8: TRAIN
    'lr': [1e-3, 1e-4, 5e-4, 1e-5, 5e-5, 5e-6, 1e-6],
    'lr_scheduler': ['linear', 'cosine', 'cyclic', 'exp', 'lambda'],
    'batch_size': [1, 4, 8, 16, 32, 64, 128, 256],
    'epoch': [100, 150, 200, 300, 400],
}


class Config:
    def __init__(self, args):
        self.args = vars(args)

    def save_yml_config(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, 'config.yml')
        with open(filepath, 'w') as outfile:
            yaml.dump(self.args, outfile, default_flow_style=False)

    def get_path(self, mode):
        if mode == "pretrain":
            return self.encoding(PRETRAIN_CFG)
        else:
            return self.encoding(FINETUNE_CFG)

    def encoding(self, levels):
        path = "" # self.root_path
        for lv in levels:
            lv_dir = str()
            if len(lv) == 1:
                lv_dir = str(self.args[lv[0]])
            else:
                for l in lv:
                    if l in opt_conds.keys():
                        opt_idx = opt_conds[l].index(self.args[l])
                        al = self.get_acronym(l)
                        lv_dir += al + str(opt_idx)
                    else:
                        al = self.get_acronym(l)
                        if type(self.args[l]) == bool:
                            if self.args[l]:
                                lv_dir += al + '_'
                            else:
                                continue
                        else:
                            lv_dir += al + str(self.args[l])
            path = os.path.join(path, lv_dir)
        return path

    def get_acronym(self, full_str, delim='_'):
        # add first letter
        short_str = full_str[0]
        # iterate over string
        for i in range(1, len(full_str)):
            if full_str[i - 1] == delim:
                # add letter next to space
                short_str += full_str[i]
        # uppercase oupt
        short_str = short_str.upper()
        return short_str

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_path = './output'
    args.dataset_name = 'ethucy'
    args.dataset_split = 'eth'

    args.obs_len = 8
    args.pred_len = 12
    args.num_nbr = 6

    args.view_range = 20.0
    args.view_angle = 1.57
    args.social_range = 1

    args.patch_size = 32
    args.env_range = 10
    args.env_resol = 0.1
    args.scene = "rgrid_map"

    args.traj_hidden = 256
    args.traj_layer = 4
    args.traj_head = 4


    args.goal_hidden = 256
    args.goal_layer = 4
    args.goal_head = 4
    args.goal_enc = [128, 256]

    args.goal_latent = 64
    args.k_sample = 20

    args.lr = 1e-5
    args.aug = True
    args.batch_size = 32
    args.epoch = 100
    args.mode = 'test'

    cfgs = Config(args.dataset_path, args)
    cfgs.save_yml_config([DATASET, SPLIT, TRAJ_REP])


    # self.DATASETS = OrderedDict({
    #     'dataset_name': args.dataset_name,
    #     'dataset_split': args.dataset_split,
    # })
    # self.TRAJECTORY = OrderedDict({
    #     'num_nbr': args.num_nbr,
    #     'obs_len': args.obs_len,
    #     'pred_len': args.pred_len,
    #     'view_range': args.view_range,
    #     'view_angle': args.view_angle,
    #     'social_range': args.social_range
    # })
    # self.ENVIRONMENT = OrderedDict({
    #     'RNG': args.env_range,
    #     'RES': args.env_resol,
    #     'PATCH': args.patch_size,
    #     'SCENE': args.scene
    # })
    # self.TRAINING = OrderedDict({
    #     'NBR': args.num_nbr,
    #     'VIEW_RNG': args.view_range,
    #     'VIEW_ANG': args.view_angle,
    #     'SOCIAL_RNG': args.social_range
    # })
    # self.PRETRAINING = OrderedDict({
    #     'NBR': args.num_nbr,
    # })
    # self.FINETUNING = OrderedDict({
    #     'NBR': args.num_nbr,
    # })
    # self.TRIAL =