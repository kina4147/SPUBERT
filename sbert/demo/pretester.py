import argparse
import os
import time
import tqdm
import copy
from SocialBERT.datasets.ethucy import ETHUCYDataset
from SocialBERT.datasets.ethucy_tpp import ETHUCYTPPDataset
from SocialBERT.datasets.ethucy_ynet import ETHUCYYNetDataset
from SocialBERT.datasets.sdd_ynet import SDDYNetDataset
from SocialBERT.models.sbertplus.trainer import SBertPlusTrainer
from SocialBERT.models.utils.viz import *
from SocialBERT.models.utils.stopper import EarlyStopping
from SocialBERT.models.sbertplus.sbertplus import (
    SBertPlusTGPConfig, SBertPlusMGPConfig,
    SBertPlusMTPModel, SBertPlusTGPModel, SBertPlusMGPModel
)
from SocialBERT.models.utils.loss import traj_loss, attn_traj_loss, mgp_cvae_loss #, traj_collision_loss, goal_collision_loss, pos_collision_loss
from SocialBERT.datasets.util import convert_patches_to_map
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from SocialBERT.models.utils.grid_map_numpy import estimate_map_length, estimate_num_patch
from SocialBERT.models.utils.viz import *
from SocialBERT.models.utils.config import Config
import matplotlib.pyplot as plt
def trainer():

    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy_tpp', help='dataset name (eth, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split (eth-eth, hotel, univ, zara1, zara2')

    parser.add_argument("--mode", default='pretrain', help='dataset split (eth, hotel, univ, zara1, zara2')
    parser.add_argument("--train_mode", default='tgp', help='mtp, tgp, mgp')

    parser.add_argument('--output_path', default='./output', help='glob expression for data files')
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
    parser.add_argument('--goal_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument('--output_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--min_obs_len", type=int, default=2, help="number of observation frames")
    parser.add_argument("--num_nbr", type=int, default=4, help="number of neighbors")
    parser.add_argument("--view_range", type=float, default=20.0, help="accessible range of target pedestrian")
    parser.add_argument("--view_angle", type=float, default=2.09, help="accessible range of target pedestrian")
    parser.add_argument("--social_range", type=float, default=2.0, help="socially-aware range")
    parser.add_argument('--scale', action='store_true', help='augment scenes')

    parser.add_argument("--env_range", type=float, default=10.0, help="socially-aware range")
    parser.add_argument("--env_resol", type=float, default=5.0, help="socially-aware range")
    parser.add_argument("--patch_size", type=int, default=32, help="socially-aware range")
    parser.add_argument('--scene', default=None, help='glob expression for data files')


    ## Ablation Setting Parameters
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--traj_layer", type=int, default=4, help="number of layers")
    parser.add_argument("--traj_head", type=int, default=4, help="number of attention heads")
    parser.add_argument("--goal_layer", type=int, default=4, help="number of layers")
    parser.add_argument("--goal_head", type=int, default=4, help="number of attention heads")

    parser.add_argument("--goal_hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--goal_latent", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--act_fn', default='gelu', help='glob expression for data files')

    # MGP params
    parser.add_argument("--cvae_sigma", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--kld_clamp", type=float, default=None, help="learning rate of adam")
    parser.add_argument("--col_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--traj_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--goal_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--kld_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--num_cycle", type=int, default=0, help="learning rate of adam")
    parser.add_argument('--normal', action='store_true', help='augment scenes')


    ## Embedding & Loss Parameters
    parser.add_argument("--sampling", type=float, default=1, help="sampling dataset")
    parser.add_argument("--k_sample", type=int, default=20, help="embedding size")
    parser.add_argument('--viz', action='store_true', help='augment scenes')
    parser.add_argument("--viz_layer", type=int, default=0, help="embedding size")
    parser.add_argument("--viz_head", type=int, default=0, help="embedding size")
    parser.add_argument('--viz_attn', action='store_true', help='augment scenes')
    parser.add_argument("--test_batch_size", type=int, default=64, help="number of batch_size")

    args = parser.parse_args()
    test_args = copy.copy(args)
    test_args.aug = False
    test_dataloader = None

    if args.dataset_name == 'ethucy':
        print("Original ETH/UCY Dataset Loading...")
        dataset = ETHUCYDataset(split="test", args=test_args)
        indices = list(range(len(dataset)))
        indices = indices[:int(len(indices) * args.sampling)]
        test_sampler = SubsetRandomSampler(indices)
        test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, sampler=test_sampler)
    elif args.dataset_name == 'ethucy_tpp':
        print("Trajectron++ ETH/UCY Dataset Loading...")

        test_dataset = ETHUCYTPPDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_worker,
                                     shuffle=False)
    elif args.dataset_name == 'ethucy_ynet':
        print("YNet ETH/UCY Dataset Loading...")
        dataset = ETHUCYYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_worker,
                                     shuffle=False)
    elif args.dataset_name == 'sdd_ynet':
        print("YNet SDD Dataset Loading...")
        args.dataset_split = 'default'
        dataset = SDDYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.num_worker,
                                     shuffle=False)
    else:
        print("Dataset is not loaded.")

    # args.traj2goal_modelpath = os.path.join(args.traj2goal_modelpath, 'ckpt.pth')
    num_patch = estimate_num_patch(estimate_map_length(args.env_range * 2, args.env_resol), args.patch_size)
    #
    # sbert_goal_cfgs = SBertPlusMGPConfig(hidden_size=args.hidden, num_layer=args.goal_layer,
    #                                           num_head=args.goal_head, k_sample=args.k_sample,
    #                                           goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent,
    #                                           obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
    #                                           scene=args.scene, num_patch=num_patch, patch_size=args.patch_size)

    cfgs = Config(args)
    cfgs_path = cfgs.get_path(mode=args.mode, train_mode=args.train_mode)
    ckpt_path = os.path.join(args.output_path, cfgs_path)
    if args.train_mode == "mtp":
        sbert_cfgs = SBertPlusTGPConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.traj_layer, num_head=args.traj_head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
            dropout_prob=args.dropout_prob,
            patch_size=args.patch_size, col_weight=args.col_weight, traj_weight=args.traj_weight, act_fn=args.act_fn,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range, scale=args.scale)
        pretrain_modelpath = os.path.join(ckpt_path, "full_model.pth")
        model = SBertPlusMTPModel(sbert_cfgs)
        model.load_state_dict(torch.load(pretrain_modelpath))
    elif args.train_mode == "tgp":
        sbert_cfgs = SBertPlusTGPConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.traj_layer, num_head=args.traj_head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
            dropout_prob=args.dropout_prob,
            patch_size=args.patch_size, col_weight=args.col_weight, traj_weight=args.traj_weight, act_fn=args.act_fn,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range, scale=args.scale)
        pretrain_modelpath = os.path.join(ckpt_path, "full_model.pth")
        model = SBertPlusTGPModel(sbert_cfgs)
        model.load_state_dict(torch.load(pretrain_modelpath))
    elif args.train_mode == "mgp":
        sbert_cfgs = SBertPlusMGPConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.goal_layer, num_head=args.goal_head, k_sample=args.k_sample,
            goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
            dropout_prob=args.dropout_prob, patch_size=args.patch_size, kld_weight=args.kld_weight,
            col_weight=args.col_weight,
            kld_clamp=args.kld_clamp, cvae_sigma=args.cvae_sigma, goal_weight=args.goal_weight, act_fn=args.act_fn,
            normal=args.normal, view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
            scale=args.scale)
        pretrain_modelpath = os.path.join(ckpt_path, "full_model.pth")
        model = SBertPlusMGPModel(sbert_cfgs)
        model.load_state_dict(torch.load(pretrain_modelpath))
        # pretrain_path = cfgs.get_path(mode="pretrain", train_mode="mgp")
        # pretrain_modelpath = os.path.join(args.output_path, pretrain_path, "full_model.pth")
        # model.mgp_model.load_state_dict(torch.load(pretrain_modelpath))
    elif args.train_mode == "tgp_mgp":
        tgp_pretrain_path = cfgs.get_path(mode="pretrain", train_mode="tgp")
        tgp_pretrain_modelpath = os.path.join(args.output_path, tgp_pretrain_path, "full_model.pth")
        model.tgp_model.load_state_dict(torch.load(tgp_pretrain_modelpath))
        mgp_pretrain_path = cfgs.get_path(mode="pretrain", train_mode="mgp")
        mgp_pretrain_modelpath = os.path.join(args.output_path, mgp_pretrain_path, "full_model.pth")
        model.mgp_model.load_state_dict(torch.load(mgp_pretrain_modelpath))

    else:
        print("from_scatch")



    # sbert_traj_cfgs = SBertPlusTGPConfig(hidden_size=args.hidden, num_layer=args.traj_layer,
    #                                           num_head=args.traj_head,
    #                                           obs_len=args.obs_len, pred_len=args.pred_len,
    #                                           num_nbr=args.num_nbr, scene=args.scene,
    #                                           num_patch=num_patch, patch_size=args.patch_size)
    # model = SBertPlusTrajectory2GoalModel(sbert_traj_cfgs)
    # model.load_state_dict(torch.load(ckpt_path))
    # model = SBertPlusTrajectory2GoalModel.from_pretrained(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # if args.cuda and torch.cuda.device_count() > 1:
    #     parallel = True
    #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
    #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # else:
    #     parallel
    model.to(device)


    with torch.no_grad():
        model.eval()
        data_iter = tqdm.tqdm(enumerate(test_dataloader),
                              desc="%s" % ("test"),
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")
        start_time = time.time()
        total_aderror = 0
        total_fderror = 0
        total_gderror = 0
        for i, data in data_iter:
            # total_data_size += len(data["encoder_spatial_ids"])
            data = {key: value.to(device) for key, value in data.items()}
            if args.train_mode == "tgp":
                if sbert_cfgs.scene:
                    outputs = model(train=False, spatial_ids=data["tgp_spatial_ids"],
                                         temporal_ids=data["tgp_temporal_ids"],
                                         segment_ids=data["tgp_segment_ids"], attn_mask=data["tgp_attn_mask"],
                                         env_spatial_ids=data["env_spatial_ids"],
                                         env_temporal_ids=data["env_temporal_ids"],
                                         env_segment_ids=data["env_segment_ids"],
                                         env_attn_mask=data["env_attn_mask"],
                                         traj_lbs=data["traj_lbs"], goal_lbs=data["goal_lbs"], envs=data["envs"],
                                         envs_params=data["envs_params"])
                else:
                    outputs = model(train=False, spatial_ids=data["tgp_spatial_ids"],
                                         temporal_ids=data["tgp_temporal_ids"],
                                         segment_ids=data["tgp_segment_ids"], attn_mask=data["tgp_attn_mask"],
                                         traj_lbs=data["traj_lbs"], goal_lbs=data["goal_lbs"])

            elif args.train_mode == "mgp":
                if sbert_cfgs.scene:
                    outputs = model(train=False, spatial_ids=data["mgp_spatial_ids"],
                                         temporal_ids=data["mgp_temporal_ids"],
                                         segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"],
                                         env_spatial_ids=data["env_spatial_ids"],
                                         env_temporal_ids=data["env_temporal_ids"],
                                         env_segment_ids=data["env_segment_ids"],
                                         env_attn_mask=data["env_attn_mask"],
                                         traj_lbs=data["traj_lbs"], goal_lbs=data["goal_lbs"], envs=data["envs"],
                                         envs_params=data["envs_params"])
                else:
                    outputs = model(train=False, spatial_ids=data["mgp_spatial_ids"],
                                         temporal_ids=data["mgp_temporal_ids"],
                                         segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"],
                                         traj_lbs=data["traj_lbs"], goal_lbs=data["goal_lbs"])

            if args.train_mode == "tgp":
                aderror, fderror = traj_loss(outputs["pred_trajs"], data["traj_lbs"])
                gderror = 0
            elif args.train_mode == "mtp":
                aderror = attn_traj_loss(outputs["pred_trajs"], data["traj_lbs"], data["attn_mask"], args.obs_len, args.pred_len)
                fderror = 0
                gderror = 0
            elif args.train_mode == "mgp":
                aderror = 0
                fderror = 0
                gderror, best_gde_idx = mgp_cvae_loss(outputs["pred_goals"], data["goal_lbs"], k_sample=sbert_cfgs.k_sample,
                                             goal_dim = sbert_cfgs.goal_dim, output_dim = sbert_cfgs.output_dim, best_of_many = True)
            else:
                raise ValueError("train mode is wrong.")

            if args.viz:
                if args.train_mode == "tgp":
                    if args.scale:
                        viz_input_trajs = data["tgp_spatial_ids"][0] * args.view_range
                    else:
                        viz_input_trajs = data["tgp_spatial_ids"][0]
                elif args.train_mode == "mgp":
                    if args.scale:
                        viz_input_trajs = data["mgp_spatial_ids"][0] * args.view_range
                    else:
                        viz_input_trajs = data["mgp_spatial_ids"][0]

                viz_scale = 0.3
                input_ax = init_viz(title="INPUT")
                mgp_result_ax = init_viz(title="MGP_RESULT")
                tgp_result_ax = init_viz(title="TGP_RESULT")
                # mgp_attn_ax = init_viz(title="MGP_ATTN")

                gt_color = "yellow"
                pred_color = "blue"
                obs_color = "red"
                nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown', 'gray', 'navy', 'green', 'orange', 'magenta', 'cyan']

                viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                          view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                          pad_val=-args.view_range, ax=input_ax, zorder=10, viz_interest_area=True, arrow=True, nbr_colors=nbr_colors)
                viz_gt_trajs(data["traj_lbs"][0], dot=True, color=gt_color, alpha=1.0, ax=input_ax, zorder=0)

                if args.train_mode == "tgp":
                    viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                         num_nbr=args.num_nbr,
                                         view_range=args.view_range, view_angle=args.view_angle,
                                         social_range=args.social_range,
                                         pad_val=-args.view_range, ax=tgp_result_ax, zorder=10, viz_interest_area=True,
                                         arrow=True, nbr_colors=nbr_colors)
                    viz_mask_trajs(viz_input_trajs, outputs["pred_trajs"][0], data["traj_lbs"][0], obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                      view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                      pad_val=-args.view_range, msk_val=args.view_range, ax=tgp_result_ax, zorder=20, scale=2)
                elif args.train_mode == "mtp":
                    viz_mask_trajs(data["spatial_ids"][0], outputs["pred_trajs"][0], data["traj_lbs"][0], obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                      view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                      pad_val=-args.view_range, msk_val=args.view_range, ax=ax)
                elif args.train_mode == "mgp":
                    # viz_gt_goal(data["goal_lbs"][0], ax=ax)
                    # viz_k_pred_goals(outputs["pred_goals"][0], ax=ax)

                    viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                         num_nbr=args.num_nbr,
                                         view_range=args.view_range, view_angle=args.view_angle,
                                         social_range=args.social_range,
                                         pad_val=-args.view_range, ax=mgp_result_ax, zorder=10, nbr_colors=nbr_colors)
                    viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0,
                                  ax=mgp_result_ax, zorder=25)
                    viz_k_pred_goals(outputs["pred_goals"][0], ax=mgp_result_ax, color=pred_color, zorder=20)
                    viz_gt_goal(data["goal_lbs"][0], color=gt_color, scale=2, ax=mgp_result_ax, zorder=50)
                    viz_goal_samples(outputs["pred_goals"][0].numpy(), 0, 0, range=args.view_range, resol=0.1, ax=mgp_result_ax,
                                     zorder=0)

                if args.viz_attn:
                    tgp_attn_ax = init_viz(title="TGP_ATTN")
                    if args.scene:
                        viz_scene_attention(env_patches=data["env_spatial_ids"][0], env_attentions=outputs["attentions"][args.viz_layer][0],
                                            obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, resol=args.env_resol,
                                            layer_id=args.viz_layer, attn_head_id=args.viz_head, alpha=0.5, ax=tgp_attn_ax)

                    viz_show(x_min=-args.view_range * viz_scale, x_max=args.view_range * viz_scale,
                             y_min=-args.view_range * viz_scale,
                             y_max=args.view_range * viz_scale,
                             axs=[input_ax, tgp_result_ax, tgp_attn_ax])
                else:
                    viz_show(x_min=-args.view_range * viz_scale, x_max=args.view_range * viz_scale,
                             y_min=-args.view_range * viz_scale,
                             y_max=args.view_range * viz_scale,
                             axs=[input_ax, tgp_result_ax, mgp_result_ax])
            total_aderror += aderror
            total_fderror += fderror
            total_gderror += gderror
        total_aderror /= len(data_iter)
        total_fderror /= len(data_iter)
        total_gderror /= len(data_iter)

        print("Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (total_aderror, total_fderror, total_gderror))
        end_time = time.time()

if __name__=='__main__':
    trainer()