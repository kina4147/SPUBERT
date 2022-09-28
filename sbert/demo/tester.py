import argparse
import os
import copy
import time
import tqdm
import random
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from SocialBERT.sbert.dataset.ethucy_star import ETHUCYSTARDataset
from SocialBERT.sbert.dataset.ethucy_ynet import ETHUCYYNetDataset
from SocialBERT.sbert.dataset.sdd_ynet import SDDYNetDataset
from SocialBERT.sbert.model.sbert import (
    SBertConfig, SBertModel, SBertFTModel
)
from SocialBERT.sbert.model.loss import ADError, FDError
from SocialBERT.sbert.util.viz import *
from SocialBERT.sbert.util.config import Config

import torch
import random

def trainer():
    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy_ynet', help='dataset name (eth, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split (eth-eth, hotel, univ, zara1, zara2')
    parser.add_argument("--mode", default='finetune', help='dataset split (eth, hotel, univ, zara1, zara2')
    parser.add_argument("--train_mode", default='fs', help='dataset split (eth, hotel, univ, zara1, zara2')
    parser.add_argument('--output_path', default='./output/sbert', help='glob expression for data files')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA: true, or false")

    # Training Parameters
    parser.add_argument('--aug', action='store_true', help='augment scenes')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="learning rate of adam")
    parser.add_argument('--warm_up', default=0, type=float, help='sample ratio of train/val scenes')
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs")
    parser.add_argument("--num_worker", type=int, default=4, help="dataloader worker size")
    parser.add_argument("--patience", type=int, default=-1, help="patience for early stopping")
    parser.add_argument("--lr_scheduler", default='linear', help="patience for early stopping")
    parser.add_argument('--clip_grads', action='store_true', help='augment scenes')

    # Model Paramters
    parser.add_argument('--input_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument('--output_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--min_obs_len", type=int, default=2, help="number of observation frames")
    parser.add_argument("--num_nbr", type=int, default=4, help="number of neighbors")
    parser.add_argument("--view_range", type=float, default=20.0, help="accessible range of target pedestrian")
    parser.add_argument("--view_angle", type=float, default=2.09, help="accessible range of target pedestrian")
    parser.add_argument("--social_range", type=float, default=2.0, help="socially-aware range")

    # Ablation Setting Parameters
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layer", type=int, default=4, help="number of layers")
    parser.add_argument("--head", type=int, default=4, help="number of attention heads")
    parser.add_argument('--act_fn', default='gelu', help='glob expression for data files')
    parser.add_argument('--sip', action='store_true', help='augment scenes')

    ## Embedding & Loss Parameters
    parser.add_argument('--viz', action='store_true', help='augment scenes')
    parser.add_argument("--viz_layer", type=int, default=3, help="embedding size")
    parser.add_argument("--viz_head", type=int, default=0, help="embedding size")
    parser.add_argument('--viz_self_attn', action='store_true', help='augment scenes')
    parser.add_argument("--test_batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument('--viz_save', action='store_true', help='augment scenes')
    parser.add_argument("--num_save", type=int, default=5, help="number of batch_size")
    parser.add_argument('--shuffle', action='store_true', help='augment scenes')

    parser.add_argument('--test', action='store_true', help='augment scenes')
    parser.add_argument("--seed", type=int, default=20, help="embedding size")


    args = parser.parse_args()
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    test_args = copy.copy(args)
    test_args.aug = False
    test_dataloader = None
    if args.dataset_name == 'ethucy_star':
        print("STAR ETH/UCY Dataset Loading...")
        test_dataset = ETHUCYSTARDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, shuffle=args.shuffle)
    elif args.dataset_name == 'ethucy_ynet':
        print("YNet ETH/UCY Dataset Loading...")
        dataset = ETHUCYYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, shuffle=args.shuffle)
    elif args.dataset_name == 'sdd_ynet':
        print("YNet SDD Dataset Loading...")
        args.dataset_split = 'default'
        dataset = SDDYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, shuffle=args.shuffle)
    else:
        print("Dataset is not loaded.")

    if args.viz:
        args.cuda = False

    cfgs = Config(args)
    cfgs_path = cfgs.get_path(mode=args.mode)
    sbert_modelpath = os.path.join(args.output_path, cfgs_path, "full_model.pth")
    print(sbert_modelpath)
    sbert_cfgs = SBertConfig(
        input_dim=args.input_dim, output_dim=args.output_dim,
        hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
        pred_len=args.pred_len, num_nbr=args.num_nbr, dropout_prob=args.dropout_prob, act_fn=args.act_fn,
        view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range, sip=args.sip)

    model = SBertFTModel(sbert_cfgs)
    model.load_state_dict(torch.load(sbert_modelpath))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        parallel = True
    else:
        parallel = False
    model.to(device)


    with torch.no_grad():
        model.eval()
        data_iter = tqdm.tqdm(enumerate(test_dataloader),
                              desc="%s" % ("test"),
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")
        total_aderror = 0
        total_fderror = 0
        total_data = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            data = {key: value.to(device) for key, value in data.items()}
            outputs = model(train=False, spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                            segment_ids=data["segment_ids"], attn_mask=data["attn_mask"], output_attentions=True)
            if args.viz:
                viz_input_trajs = data["spatial_ids"][0, :, :2]
                axs = []
                input_ax = init_viz(title="INPUT")
                result_ax = init_viz(title="RESULT")

                gt_color = "red"
                pred_color = "black"
                obs_color = "red"
                nbr_colors = ['blue', 'green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown', 'gray', 'navy']

                # INPUT
                viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                          view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                          pad_val=-args.view_range, ax=input_ax, zorder=10, viz_interest_area=True, arrow=True, nbr_colors=nbr_colors)
                viz_gt_trajs(data["traj_lbl"][0], dot=True, color=gt_color, alpha=1.0, ax=input_ax, zorder=0)
                viz_pred_trajs(outputs["pred_traj"][0], alpha=1.0, dot=True, color=pred_color, ax=result_ax, zorder=28)

                # RESULT
                viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                          view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                          pad_val=-args.view_range, ax=result_ax, zorder=10, nbr_colors=nbr_colors, viz_interest_area=True, arrow=True)
                viz_gt_trajs(data["traj_lbl"][0], dot=True, color=gt_color, alpha=1.0, ax=result_ax, zorder=0)
                viz_pred_trajs(outputs["pred_traj"][0], alpha=1.0, dot=True, color=pred_color, ax=result_ax, zorder=28)
                axs.append(input_ax)
                axs.append(result_ax)

                for head in range(args.head):
                    # ATTN FOR only one output
                    attn_ax = init_viz(title="ATTN_H" + str(head))
                    viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                              view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                              pad_val=-args.view_range, ax=attn_ax, zorder=10, nbr_colors=nbr_colors, viz_interest_area=True, arrow=True)
                    viz_gt_trajs(data["traj_lbl"][0], dot=True, color=gt_color, alpha=1.0, ax=attn_ax, zorder=0)
                    # Batch 0 ~ k_sample-1
                    viz_trajectory_attention(input_trajs=viz_input_trajs, input_gt_trajs=data["traj_lbl"][0], input_pred_trajs=outputs["pred_traj"][0], input_traj_attentions=outputs["attentions"][args.viz_layer][0],
                                             self_attn=args.viz_self_attn, layer_id=args.viz_layer, head_id=head,
                                             obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                             ofs_x=-2, ofs_y=-2, ax=attn_ax, nbr_colors=nbr_colors)
                    viz_pred_trajs(outputs["pred_traj"][0], alpha=1.0, color=pred_color, dot=True, ax=attn_ax, zorder=30)
                    axs.append(attn_ax)

                if args.viz_save:
                    viz_save(x_min=-args.view_range/2, x_max=args.view_range/2, y_min=-args.view_range/2,
                             y_max=args.view_range/2, axs=axs, desc=str(i), path=os.path.join(args.output_path, args.dataset_name))
                    if i > args.num_save:
                        break
                else:
                    viz_show(x_min=-args.view_range/2, x_max=args.view_range/2, y_min=-args.view_range/2,
                             y_max=args.view_range/2, axs=axs)
            else:
                data["traj_lbl"] = torch.einsum('ijk,i->ijk', data["traj_lbl"], data["scales"])
                outputs["pred_traj"] = torch.einsum('ijk,i->ijk', outputs["pred_traj"], data["scales"])

                aderror = ADError(pred_traj=outputs["pred_traj"], gt_traj=data["traj_lbl"])
                fderror = FDError(pred_final_pos=outputs["pred_traj"][:, -1, :].squeeze(dim=1),
                                  gt_final_pos=data["traj_lbl"][:, -1, :].squeeze(dim=1))
                total_data += len(data["spatial_ids"])
                total_aderror += aderror
                total_fderror += fderror
                print("Epoch Evaluation Result >>>>> ADE: %f, FDE: %f" % (
                aderror / len(data["spatial_ids"]), fderror / len(data["spatial_ids"])))

        print("Total Evaluation Result >>>>> ADE: %f, FDE: %f" % (total_aderror/total_data, total_fderror/total_data))
        end_time = time.time()

if __name__=='__main__':
    trainer()