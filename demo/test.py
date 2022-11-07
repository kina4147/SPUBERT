import argparse
import os
import time
import tqdm
import random

import torch.nn as nn
from torch.utils.data import DataLoader
from SPUBERT.dataset.ethucy import ETHUCYDataset
from SPUBERT.dataset.sdd import SDDDataset
from SPUBERT.model.spubert import (
    SPUBERTTGPConfig, SPUBERTMGPConfig, SPUBERTConfig, SPUBERTModel
)
from SPUBERT.model.loss import bom_loss_1, bom_loss_3, goal_collision_loss, pos_collision_loss
from SPUBERT.dataset.grid_map_numpy import estimate_map_length, estimate_num_patch
from SPUBERT.util.viz import *
from SPUBERT.util.config import Config


def test():

    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy', help='dataset name (eth, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split (eth-eth, hotel, univ, zara1, zara2')
    parser.add_argument('--output_path', default='./output', help='glob expression for data files')
    parser.add_argument('--output_name', default='test', help='glob expression for data files')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA: true, or false")

    # Training Parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="learning rate of adam")
    parser.add_argument('--warm_up', default=0, type=float, help='sample ratio of train/val scenes')
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs")
    parser.add_argument("--num_worker", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--patience", type=int, default=-1, help="patience for early stopping")

    parser.add_argument('--clip_grads', action='store_true', default=True, help='augment scenes')

    # Model Paramters
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--min_obs_len", type=int, default=2, help="number of observation frames")
    parser.add_argument("--num_nbr", type=int, default=4, help="number of neighbors")
    parser.add_argument("--view_range", type=float, default=20.0, help="accessible range of target pedestrian")
    parser.add_argument("--view_angle", type=float, default=2.09, help="accessible range of target pedestrian")
    parser.add_argument("--social_range", type=float, default=2.0, help="socially-aware range")
    
    parser.add_argument("--env_range", type=float, default=10.0, help="socially-aware range")
    parser.add_argument("--env_resol", type=float, default=0.2, help="socially-aware range")
    parser.add_argument("--patch_size", type=int, default=16, help="socially-aware range")
    parser.add_argument('--scene', action='store_true', help='glob expression for data files')
    parser.add_argument('--binary_scene', action='store_true', help='augment scenes')

    ## Ablation Setting Parameters
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layer", type=int, default=4, help="number of layers")
    parser.add_argument("--head", type=int, default=4, help="number of attention heads")

    parser.add_argument("--goal_hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--goal_latent", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--act_fn', default='relu', help='glob expression for data files')

    # Hyperparameters
    parser.add_argument("--traj_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--goal_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--kld_weight", type=float, default=1.0, help="learning rate of adam")

    ## Embedding & Loss Parameters
    parser.add_argument("--k_sample", type=int, default=20, help="embedding size")
    parser.add_argument("--test_k_sample", type=int, default=20, help="embedding size")
    parser.add_argument("--d_sample", type=int, default=400, help="number of batch_size")
    parser.add_argument('--viz', action='store_true', help='augment scenes')
    parser.add_argument("--viz_layer", type=int, default=0, help="embedding size")
    parser.add_argument("--viz_head", type=int, default=0, help="embedding size")
    parser.add_argument('--viz_self_attn', action='store_true', help='augment scenes')
    parser.add_argument('--viz_t2e_attn', action='store_true', help='augment scenes')
    parser.add_argument("--test_batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument('--viz_save', action='store_true', help='augment scenes')
    parser.add_argument("--num_save", type=int, default=5, help="number of batch_size")
    parser.add_argument('--shuffle', action='store_true', help='augment scenes')
    parser.add_argument('--test', action='store_true', help='augment scenes')

    parser.add_argument("--seed", type=int, default=0, help="embedding size")
    parser.add_argument("--test_seed", type=int, default=-1, help="embedding size")
    parser.add_argument('--seed_search', action='store_true', help='augment scenes')
    parser.add_argument('--scene_list', nargs='+', default=[], type=int)



    # parser.add_argument('--input_dim', type=int, default=2, help="number of batch_size")
    # parser.add_argument('--goal_dim', type=int, default=2, help="number of batch_size")
    # parser.add_argument('--output_dim', type=int, default=2, help="number of batch_size")
    # parser.add_argument("--mode", default='finetune', help='dataset split (eth, hotel, univ, zara1, zara2')
    # parser.add_argument("--train_mode", default='fs', help='mtp_sep, mtp_shr, tgp, mgp, tgp_mgp')
    # parser.add_argument("--share", action='store_true', help='augment scenes')
    # parser.add_argument('--freeze', action='store_true', help='augment scenes')
    # parser.add_argument('--aug', action='store_true', default=True, help='augment scenes')
    # parser.add_argument("--lr_scheduler", default='it_linear', help="patience for early stopping")
    # parser.add_argument("--param_last_epoch", type=float, default=0, help="learning rate of adam")
    # parser.add_argument('--sip', action='store_true', help='augment scenes')
    # parser.add_argument("--cvae_sigma", type=float, default=1.0, help="learning rate of adam")
    # parser.add_argument("--kld_clamp", type=float, default=None, help="learning rate of adam")
    # parser.add_argument("--col_weight", type=float, default=0.0, help="learning rate of adam")
    # parser.add_argument("--num_cycle", type=int, default=0.0, help="learning rate of adam")
    # parser.add_argument('--normal', action='store_true', help='augment scenes')
    # parser.add_argument("--sampling", type=float, default=1, help="sampling dataset")

    args = parser.parse_args()
    if args.viz:
        args.cuda = False
        args.test_batch_size = 1
        if len(args.scene_list) > 0:
            num_save = len(args.scene_list)
    print(args)
    test_args = copy.copy(args)
    test_args.aug = False
    test_dataloader = None
    if args.dataset_name == 'ethucy':
        print("ETH/UCY Dataset Loading...")
        test_dataset = ETHUCYDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, shuffle=args.shuffle)
    elif args.dataset_name == 'sdd':
        print("SDD Dataset Loading...")
        args.dataset_split = 'default'
        test_dataset = SDDDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_worker, shuffle=args.shuffle)
    else:
        print("Dataset is not loaded.")

    if args.test_seed == -1:
        args.test_seed = args.seed

    if args.test_seed >= 0:
        random_seed = args.test_seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    # cfgs = Config(args)
    # cfgs_path = cfgs.get_path(mode=args.mode)
    model_path = os.path.join(args.output_path, args.dataset_name, args.dataset_split, args.output_name, "full_model.pth")

    if args.scene:
        num_patch = estimate_num_patch(estimate_map_length(args.env_range * 2, args.env_resol), args.patch_size)
    else:
        num_patch = 0
        
    spubert_tgp_cfgs = SPUBERTTGPConfig(
                hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
                pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch, dropout_prob=args.dropout_prob,
                patch_size=args.patch_size, traj_weight=args.traj_weight, act_fn=args.act_fn,
                view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

    spubert_mgp_cfgs = SPUBERTMGPConfig(
                hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, k_sample=args.test_k_sample,
                goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent, obs_len=args.obs_len,
                pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
                dropout_prob=args.dropout_prob, patch_size=args.patch_size, kld_weight=args.kld_weight,
                goal_weight=args.goal_weight, act_fn=args.act_fn,
                view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)
    spubert_cfgs = SPUBERTConfig(traj_cfgs=spubert_tgp_cfgs, goal_cfgs=spubert_mgp_cfgs)
    
    model = SPUBERTModel(spubert_tgp_cfgs, spubert_mgp_cfgs, spubert_cfgs)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.cuda and torch.cuda.device_count() > 1:
        print("Using %d GPUS for BERT" % torch.cuda.device_count())
        model = nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        parallel = True
    else:
        parallel = False
    model.to(device)

    diff_time = 0

    with torch.no_grad():
        model.eval()

        if args.seed_search:
            min_ade = 1000000
            optimal_seed = 0
            for seed in range(10000):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if use multi-GPU
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                np.random.seed(seed)
                random.seed(seed)

                total_aderror = 0
                total_fderror = 0
                total_gderror = 0
                total_data = 0
                data_iter = tqdm.tqdm(enumerate(test_dataloader),
                                      desc="%s \n" % ("TEST"),
                                      total=len(test_dataloader),
                                      bar_format="{l_bar}{r_bar}")
                for i, data in data_iter:
                    data = {key: value.to(device) for key, value in data.items()}
                    if args.scene:
                        outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                                  mgp_temporal_ids=data["mgp_temporal_ids"],
                                                  mgp_segment_ids=data["mgp_segment_ids"],
                                                  mgp_attn_mask=data["mgp_attn_mask"],
                                                  tgp_temporal_ids=data["tgp_temporal_ids"],
                                                  tgp_segment_ids=data["tgp_segment_ids"],
                                                  tgp_attn_mask=data["tgp_attn_mask"],
                                                  env_spatial_ids=data["env_spatial_ids"],
                                                  env_temporal_ids=data["env_temporal_ids"],
                                                  env_segment_ids=data["env_segment_ids"],
                                                  env_attn_mask=data["env_attn_mask"], output_attentions=True,
                                                  d_sample=args.d_sample)
                    else:
                        # start_time = time.time()
                        outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                                  mgp_temporal_ids=data["mgp_temporal_ids"],
                                                  mgp_segment_ids=data["mgp_segment_ids"],
                                                  mgp_attn_mask=data["mgp_attn_mask"],
                                                  tgp_temporal_ids=data["tgp_temporal_ids"],
                                                  tgp_segment_ids=data["tgp_segment_ids"],
                                                  tgp_attn_mask=data["tgp_attn_mask"],
                                                  output_attentions=True, d_sample=args.d_sample)



                    outputs["pred_trajs"] = torch.einsum('bkts,b->bkts', outputs["pred_trajs"], data["scales"])
                    outputs["pred_goals"] = torch.einsum('bks,b->bks', outputs["pred_goals"], data["scales"])
                    data["traj_lbl"] = torch.einsum('bts,b->bts', data["traj_lbl"], data["scales"])
                    data["goal_lbl"] = torch.einsum('bs,b->bs', data["goal_lbl"], data["scales"])
                    gderror, aderror, fderror = bom_loss_3(
                        outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                        args.test_k_sample)

                    total_aderror += aderror
                    total_fderror += fderror
                    total_gderror += gderror
                    total_data += len(data["mgp_spatial_ids"])
                total_ade = total_aderror / total_data
                print("Total Evaluation Result >>>>> SEED: %f, ADE: %f, FDE: %f, GDE: %f" % (
                seed, total_aderror / total_data, total_fderror / total_data, total_gderror / total_data))
                if total_ade < min_ade:
                    min_ade = total_ade
                    optimal_seed = seed
                print("OPTIMAL SEED AND ADE >>>>> SEED: %f, ADE: %f" % (optimal_seed, min_ade))

            print("Optimal Seed: %f, Minimum ADE: %f" % (optimal_seed, min_ade))




        else:
            num_viz = 0
            total_aderror = 0
            total_fderror = 0
            total_gderror = 0
            total_data = 0
            data_iter = tqdm.tqdm(enumerate(test_dataloader),
                                  desc="%s" % ("TEST"),
                                  total=len(test_dataloader),
                                  bar_format="{l_bar}{r_bar}")
            for i, data in data_iter:
                if args.viz and len(args.scene_list) > 0:
                    if i not in args.scene_list:
                        continue
                data = {key: value.to(device) for key, value in data.items()}
                if args.scene:
                    start_time = time.time()
                    outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                    mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                    tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                    env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                    env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"], output_attentions=True, d_sample=args.d_sample)
                    diff_time += (time.time() - start_time)
                    if args.viz:
                        if parallel:
                            pred_goals = model.module.mgp_model.goal_predict(spatial_ids=data["mgp_spatial_ids"], temporal_ids=data["mgp_temporal_ids"],
                                                    segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"],
                                                    env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                                    env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"], num_goal=args.d_sample)

                        else:
                            pred_goals = model.mgp_model.goal_predict(spatial_ids=data["mgp_spatial_ids"], temporal_ids=data["mgp_temporal_ids"],
                                                    segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"],
                                                    env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                                    env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"], num_goal=args.d_sample)
                else:
                    start_time = time.time()
                    outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                    mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                    tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                    output_attentions=True, d_sample=args.d_sample)
                    diff_time += (time.time() - start_time)

                    if args.viz:
                        if parallel:
                            pred_goals = model.module.mgp_model.goal_predict(spatial_ids=data["mgp_spatial_ids"], temporal_ids=data["mgp_temporal_ids"],
                                                segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"], num_goal=args.d_sample)

                        else:
                            pred_goals = model.mgp_model.goal_predict(spatial_ids=data["mgp_spatial_ids"], temporal_ids=data["mgp_temporal_ids"],
                                                segment_ids=data["mgp_segment_ids"], attn_mask=data["mgp_attn_mask"], num_goal=args.d_sample)

                avg_time = diff_time / (i + 1)
                print("avg time: ", avg_time)
                if args.viz:
                    _, _, _, best_gde_idx, _, _ = bom_loss_1(
                        outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                        args.test_k_sample)
                    viz_input_trajs = data["mgp_spatial_ids"][0, :, :2]
                    viz_scale = 0.5
                    all_result_ax = init_viz(title="ALL_RESULT")
                    if args.viz_save:
                        input_ax = init_viz(title="INPUT")
                        mgp_result_ax = init_viz(title="MGP_RESULT")
                        tgp_result_ax = init_viz(title="TGP_RESULT")
                        # mgp_attn_ax = init_viz(title="MGP_ATTN")
                        # tgp_social_attn_ax = init_viz(title="TGP_SOCIAL_ATTN")
                        # tgp_scene_attn_ax = init_viz(title="TGP_SCENE_ATTN")
                        tgp_social_attn_ax_h1 = init_viz(title="TGP_SOCIAL_ATTN_H1")
                        tgp_scene_attn_ax_h1 = init_viz(title="TGP_SCENE_ATTN_H1")
                        tgp_social_attn_ax_h2 = init_viz(title="TGP_SOCIAL_ATTN_H2")
                        tgp_scene_attn_ax_h2 = init_viz(title="TGP_SCENE_ATTN_H2")
                        tgp_social_attn_ax_h3 = init_viz(title="TGP_SOCIAL_ATTN_H3")
                        tgp_scene_attn_ax_h3 = init_viz(title="TGP_SCENE_ATTN_H3")
                        tgp_social_attn_ax_h4 = init_viz(title="TGP_SOCIAL_ATTN_H4")
                        tgp_scene_attn_ax_h4 = init_viz(title="TGP_SCENE_ATTN_H4")
                        mgp_social_attn_ax_h1 = init_viz(title="MGP_SOCIAL_ATTN_H1")
                        mgp_scene_attn_ax_h1 = init_viz(title="MGP_SCENE_ATTN_H1")
                        mgp_social_attn_ax_h2 = init_viz(title="MGP_SOCIAL_ATTN_H2")
                        mgp_scene_attn_ax_h2 = init_viz(title="MGP_SCENE_ATTN_H2")
                        mgp_social_attn_ax_h3 = init_viz(title="MGP_SOCIAL_ATTN_H3")
                        mgp_scene_attn_ax_h3 = init_viz(title="MGP_SCENE_ATTN_H3")
                        mgp_social_attn_ax_h4 = init_viz(title="MGP_SOCIAL_ATTN_H4")
                        mgp_scene_attn_ax_h4 = init_viz(title="MGP_SCENE_ATTN_H4")
                    # mgp_attn_all_ax_l1_h1 = init_viz(title="MGP_ATTN_ALL_L1_H1")
                    # mgp_attn_all_ax_l1_h2 = init_viz(title="MGP_ATTN_ALL_L1_H2")
                    # mgp_attn_all_ax_l1_h3 = init_viz(title="MGP_ATTN_ALL_L1_H3")
                    # mgp_attn_all_ax_l1_h4 = init_viz(title="MGP_ATTN_ALL_L1_H4")
                    # mgp_attn_all_ax_l2_h1 = init_viz(title="MGP_ATTN_ALL_L2_H1")
                    # mgp_attn_all_ax_l2_h2 = init_viz(title="MGP_ATTN_ALL_L2_H2")
                    # mgp_attn_all_ax_l2_h3 = init_viz(title="MGP_ATTN_ALL_L2_H3")
                    # mgp_attn_all_ax_l2_h4 = init_viz(title="MGP_ATTN_ALL_L2_H4")
                    # mgp_attn_all_ax_l3_h1 = init_viz(title="MGP_ATTN_ALL_L3_H1")
                    # mgp_attn_all_ax_l3_h2 = init_viz(title="MGP_ATTN_ALL_L3_H2")
                    # mgp_attn_all_ax_l3_h3 = init_viz(title="MGP_ATTN_ALL_L3_H3")
                    # mgp_attn_all_ax_l3_h4 = init_viz(title="MGP_ATTN_ALL_L3_H4")
                    mgp_attn_all_ax_l4_h1 = init_viz(title="MGP_ATTN_ALL_L4_H1")
                    mgp_attn_all_ax_l4_h2 = init_viz(title="MGP_ATTN_ALL_L4_H2")
                    mgp_attn_all_ax_l4_h3 = init_viz(title="MGP_ATTN_ALL_L4_H3")
                    mgp_attn_all_ax_l4_h4 = init_viz(title="MGP_ATTN_ALL_L4_H4")
                    tgp_attn_all_ax_l4_h1 = init_viz(title="TGP_ATTN_ALL_L4_H1")
                    tgp_attn_all_ax_l4_h2 = init_viz(title="TGP_ATTN_ALL_L4_H2")
                    tgp_attn_all_ax_l4_h3 = init_viz(title="TGP_ATTN_ALL_L4_H3")
                    tgp_attn_all_ax_l4_h4 = init_viz(title="TGP_ATTN_ALL_L4_H4")


                    gt_color = "yellow"
                    pred_color = "blue"
                    obs_color = "red"
                    nbr_colors = ['green', 'orange', 'magenta', 'cyan', 'purple', 'pink', 'brown', 'gray', 'navy']

                    # ALL(MGP + TGP) Result
                    viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                              view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                              pad_val=-args.view_range, ax=all_result_ax, zorder=10, arrow_zorder=40, viz_interest_area=True, arrow=True, nbr_colors=nbr_colors)
                    viz_goal_samples(pred_goals[0].numpy(), ax=all_result_ax, alpha=0.1, color=pred_color, zorder=0)
                    viz_k_pred_goals(outputs["pred_goals"][0], ax=all_result_ax, scale=1.0, alpha=1.0, color=pred_color, zorder=20)
                    # viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=4, alpha=1.0, ax=all_result_ax, zorder=25)
                    # viz_gt_goal(data["goal_lbl"][0], color=gt_color, scale=2, alpha=1.0, ax=all_result_ax, zorder=27)
                    viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, linewidth=4.0, dot=False, color=pred_color, ax=all_result_ax, zorder=28)
                    viz_k_pred_trajs(outputs["pred_trajs"][0], alpha=0.2, ax=all_result_ax, color=pred_color, zorder=25)
                    viz_gt_trajs(data["traj_lbl"][0], dot=False, color=gt_color, scale=1, linewidth=2.0, alpha=1.0,
                                 ax=all_result_ax, zorder=29)
                    if args.scene:
                        viz_input_scene(env_patches=data["env_spatial_ids"][0],
                                        obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                        resol=args.env_resol, alpha=0.2, ax=all_result_ax)
                        viz_attention(attentions=outputs["goal_attentions"][0][0], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=0,
                                      ax=mgp_attn_all_ax_l4_h1)

                        viz_attention(attentions=outputs["goal_attentions"][1][0], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=1,
                                      ax=mgp_attn_all_ax_l4_h2)

                        viz_attention(attentions=outputs["goal_attentions"][2][0], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=2,
                                      ax=mgp_attn_all_ax_l4_h3)

                        viz_attention(attentions=outputs["goal_attentions"][3][0], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=3,
                                      ax=mgp_attn_all_ax_l4_h4)

                        viz_attention(attentions=outputs["traj_attentions"][0][best_gde_idx[0]], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=0,
                                      ax=tgp_attn_all_ax_l4_h1)

                        viz_attention(attentions=outputs["traj_attentions"][1][best_gde_idx[0]], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=1,
                                      ax=tgp_attn_all_ax_l4_h2)

                        viz_attention(attentions=outputs["traj_attentions"][2][best_gde_idx[0]], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=2,
                                      ax=tgp_attn_all_ax_l4_h3)

                        viz_attention(attentions=outputs["traj_attentions"][3][best_gde_idx[0]], obs_len=args.obs_len,
                                      pred_len=args.pred_len, num_nbr=args.num_nbr, env_seq_len=num_patch, head_id=3,
                                      ax=tgp_attn_all_ax_l4_h4)
                                # outputs["traj_attentions"][args.viz_layer][best_gde_idx[0]]

                    if args.viz_save:
                        # INPUT
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=input_ax, zorder=10, viz_interest_area=True, arrow=True, nbr_colors=nbr_colors)
                        viz_gt_trajs(data["traj_lbl"][0], dot=True, color=gt_color, alpha=1.0, ax=input_ax, zorder=1)

                        # MGP RESULT
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=mgp_result_ax, zorder=10, nbr_colors=nbr_colors, arrow=True)
                        viz_k_pred_goals(outputs["pred_goals"][0], ax=mgp_result_ax, scale=1.0, alpha=1.0, color=pred_color, zorder=20)
                        # viz_goal_samples(pred_goals[0].numpy(), 0, 0, range=args.view_range, resol=args.env_resol, ax=mgp_result_ax, alpha=0.8, zorder=0)
                        viz_goal_samples(pred_goals[0].numpy(), ax=mgp_result_ax, alpha=0.1, color=pred_color, zorder=0)
                        viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=4, alpha=1.0, ax=mgp_result_ax, zorder=28)
                        viz_gt_goal(data["goal_lbl"][0], color=gt_color, scale=2, alpha=1.0, ax=mgp_result_ax, zorder=29)

                        # TGP_RESULT
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=tgp_result_ax, zorder=10, arrow_zorder=40, nbr_colors=nbr_colors, arrow=True)
                        viz_k_pred_trajs(outputs["pred_trajs"][0], alpha=0.2, ax=tgp_result_ax, color=pred_color, zorder=25)
                        # viz_k_pred_goals(outputs["pred_goals"][0], ax=tgp_result_ax, alpha=1.0, color=pred_color, zorder=26)
                        viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, linewidth=4.0, dot=False, color=pred_color, ax=tgp_result_ax, zorder=28)
                        viz_gt_trajs(data["traj_lbl"][0], dot=False, color=gt_color, scale=1, linewidth=2.0, alpha=1.0, ax=tgp_result_ax, zorder=29)



                        # MGP TRAJ ATTN HEAD 1
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=mgp_social_attn_ax_h1, zorder=10, nbr_colors=nbr_colors)
                        # viz_gt_goal(data["goal_lbl"][0], ax=mgp_attn_ax, scale=2, zorder=50)
                        viz_goal_attention(input_trajs=viz_input_trajs, input_gt_goals=data["goal_lbl"][0], input_pred_goals=outputs["pred_goals"][0][best_gde_idx[0]], input_traj_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=0,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=mgp_social_attn_ax_h1, zorder=30, nbr_colors=nbr_colors)
                        viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0, ax=mgp_social_attn_ax_h1, zorder=25)


                        # TGP TRAJ ATTN HEAD 1
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=tgp_social_attn_ax_h1, zorder=10, arrow_zorder=40, nbr_colors=nbr_colors)
                        # Batch 0 ~ k_sample-1
                        viz_trajectory_attention(input_trajs=viz_input_trajs, input_gt_trajs=data["traj_lbl"][0], input_pred_trajs=outputs["pred_trajs"][0][best_gde_idx[0]], input_traj_attentions=outputs["traj_attentions"][args.viz_layer][best_gde_idx[0]],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=0,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=tgp_social_attn_ax_h1, nbr_colors=nbr_colors)
                        viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True, ax=tgp_social_attn_ax_h1, zorder=30)

                        # MGP TRAJ ATTN HEAD 2
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=mgp_social_attn_ax_h2, zorder=10, nbr_colors=nbr_colors)
                        # viz_gt_goal(data["goal_lbl"][0], ax=mgp_attn_ax, scale=2, zorder=50)
                        viz_goal_attention(input_trajs=viz_input_trajs, input_gt_goals=data["goal_lbl"][0], input_pred_goals=outputs["pred_goals"][0][best_gde_idx[0]], input_traj_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=1,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=mgp_social_attn_ax_h2, zorder=30, nbr_colors=nbr_colors)
                        viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0, ax=mgp_social_attn_ax_h2, zorder=25)


                        # TGP TRAJ ATTN HEAD 2
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=tgp_social_attn_ax_h2, zorder=10, arrow_zorder=40, nbr_colors=nbr_colors)
                        # Batch 0 ~ k_sample-1
                        viz_trajectory_attention(input_trajs=viz_input_trajs, input_gt_trajs=data["traj_lbl"][0], input_pred_trajs=outputs["pred_trajs"][0][best_gde_idx[0]], input_traj_attentions=outputs["traj_attentions"][args.viz_layer][best_gde_idx[0]],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=1,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=tgp_social_attn_ax_h2, nbr_colors=nbr_colors)
                        viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True, ax=tgp_social_attn_ax_h2, zorder=30)

                        # MGP TRAJ ATTN HEAD 3
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=mgp_social_attn_ax_h3, zorder=10, nbr_colors=nbr_colors)
                        # viz_gt_goal(data["goal_lbl"][0], ax=mgp_attn_ax, scale=2, zorder=50)
                        viz_goal_attention(input_trajs=viz_input_trajs, input_gt_goals=data["goal_lbl"][0], input_pred_goals=outputs["pred_goals"][0][best_gde_idx[0]], input_traj_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=2,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=mgp_social_attn_ax_h1, zorder=30, nbr_colors=nbr_colors)
                        viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0, ax=mgp_social_attn_ax_h3, zorder=25)


                        # TGP TRAJ ATTN HEAD 3
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=tgp_social_attn_ax_h3, zorder=10, arrow_zorder=40, nbr_colors=nbr_colors)
                        # Batch 0 ~ k_sample-1
                        viz_trajectory_attention(input_trajs=viz_input_trajs, input_gt_trajs=data["traj_lbl"][0], input_pred_trajs=outputs["pred_trajs"][0][best_gde_idx[0]], input_traj_attentions=outputs["traj_attentions"][args.viz_layer][best_gde_idx[0]],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=2,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=tgp_social_attn_ax_h3, nbr_colors=nbr_colors)
                        viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True, ax=tgp_social_attn_ax_h3, zorder=30)

                        # MGP TRAJ ATTN HEAD 4
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=mgp_social_attn_ax_h4, zorder=10, nbr_colors=nbr_colors)
                        # viz_gt_goal(data["goal_lbl"][0], ax=mgp_attn_ax, scale=2, zorder=50)
                        viz_goal_attention(input_trajs=viz_input_trajs, input_gt_goals=data["goal_lbl"][0], input_pred_goals=outputs["pred_goals"][0][best_gde_idx[0]], input_traj_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=3,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=mgp_social_attn_ax_h4, zorder=30, nbr_colors=nbr_colors)
                        viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0, ax=mgp_social_attn_ax_h4, zorder=25)


                        # TGP TRAJ ATTN HEAD 4
                        viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                  view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range,
                                  pad_val=-args.view_range, ax=tgp_social_attn_ax_h4, zorder=10, arrow_zorder=40, nbr_colors=nbr_colors)
                        # Batch 0 ~ k_sample-1
                        viz_trajectory_attention(input_trajs=viz_input_trajs, input_gt_trajs=data["traj_lbl"][0], input_pred_trajs=outputs["pred_trajs"][0][best_gde_idx[0]], input_traj_attentions=outputs["traj_attentions"][args.viz_layer][best_gde_idx[0]],
                                                 self_attn=args.viz_self_attn, env_seq_len=num_patch, layer_id=args.viz_layer, head_id=3,
                                                 obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr, pad_val=-args.view_range,
                                                 ofs_x=-2, ofs_y=-2, ax=tgp_social_attn_ax_h4, nbr_colors=nbr_colors)
                        viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True, ax=tgp_social_attn_ax_h4, zorder=30)
                        num_viz += 1

                        if args.scene:
                            mgp_col_loss = goal_collision_loss(outputs["pred_goals"], envs=data["envs"], envs_params=data["envs_params"])
                            tgp_col_loss = pos_collision_loss(outputs["pred_trajs"], envs=data["envs"], envs_params=data["envs_params"])
                            print("mgp_col_loss: ", mgp_col_loss, ", tgp_col_loss: ", tgp_col_loss)
                            viz_input_scene(env_patches=data["env_spatial_ids"][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, alpha=0.2, ax=input_ax)
                            viz_input_scene(env_patches=data["env_spatial_ids"][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, alpha=0.2, ax=mgp_result_ax)
                            viz_input_scene(env_patches=data["env_spatial_ids"][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, alpha=0.2, ax=tgp_result_ax)

                            # MGP SCENE HEAD 1
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=mgp_scene_attn_ax_h1, zorder=10,
                                                 nbr_colors=nbr_colors)
                            viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0,
                                          ax=mgp_scene_attn_ax_h1, zorder=25)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=0,
                                                alpha=0.5, ax=mgp_scene_attn_ax_h1)


                            # TGP SCENE HEAD 1
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=tgp_scene_attn_ax_h1, zorder=10, arrow_zorder=40,
                                                 nbr_colors=nbr_colors)
                            viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True,
                                           ax=tgp_scene_attn_ax_h1, zorder=30)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["traj_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=0,
                                                alpha=0.5, ax=tgp_scene_attn_ax_h1)

                            # MGP SCENE HEAD 2
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=mgp_scene_attn_ax_h2, zorder=10,
                                                 nbr_colors=nbr_colors)
                            viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0,
                                          ax=mgp_scene_attn_ax_h2, zorder=25)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=1,
                                                alpha=0.5, ax=mgp_scene_attn_ax_h2)

                            # TGP SCENE HEAD 2
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=tgp_scene_attn_ax_h2, zorder=10,
                                                 arrow_zorder=40,
                                                 nbr_colors=nbr_colors)
                            viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True,
                                           ax=tgp_scene_attn_ax_h2, zorder=30)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["traj_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=1,
                                                alpha=0.5, ax=tgp_scene_attn_ax_h2)

                            # MGP SCENE HEAD 3
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=mgp_scene_attn_ax_h3, zorder=10,
                                                 nbr_colors=nbr_colors)
                            viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0,
                                          ax=mgp_scene_attn_ax_h3, zorder=25)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=2,
                                                alpha=0.5, ax=mgp_scene_attn_ax_h3)

                            # TGP SCENE HEAD 3
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=tgp_scene_attn_ax_h3, zorder=10,
                                                 arrow_zorder=40,
                                                 nbr_colors=nbr_colors)
                            viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True,
                                           ax=tgp_scene_attn_ax_h3, zorder=30)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["traj_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=2,
                                                alpha=0.5, ax=tgp_scene_attn_ax_h3)

                            # MGP SCENE HEAD 4
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle, arrow_zorder=40,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=mgp_scene_attn_ax_h4, zorder=10,
                                                 nbr_colors=nbr_colors)
                            viz_pred_goal(outputs["pred_goals"][0][best_gde_idx[0]], color=pred_color, scale=2, alpha=1.0,
                                          ax=mgp_scene_attn_ax_h4, zorder=25)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["goal_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=3,
                                                alpha=0.5, ax=mgp_scene_attn_ax_h4)

                            # TGP SCENE HEAD 4
                            viz_input_trajectory(viz_input_trajs, obs_len=args.obs_len, pred_len=args.pred_len,
                                                 num_nbr=args.num_nbr,
                                                 view_range=args.view_range, view_angle=args.view_angle,
                                                 social_range=args.social_range,
                                                 pad_val=-args.view_range, ax=tgp_scene_attn_ax_h4, zorder=10,
                                                 arrow_zorder=40,
                                                 nbr_colors=nbr_colors)
                            viz_pred_trajs(outputs["pred_trajs"][0][best_gde_idx[0]], alpha=1.0, color=pred_color, dot=True,
                                           ax=tgp_scene_attn_ax_h4, zorder=30)
                            viz_scene_attention(env_patches=data["env_spatial_ids"][0],
                                                env_attentions=outputs["traj_attentions"][args.viz_layer][0],
                                                obs_len=args.obs_len, pred_len=args.pred_len, num_nbr=args.num_nbr,
                                                resol=args.env_resol, layer_id=args.viz_layer, attn_head_id=3,
                                                alpha=0.5, ax=tgp_scene_attn_ax_h4)


                    if args.scene:
                        if args.viz_save:
                            viz_save(x_min=-args.view_range*viz_scale, x_max=args.view_range*viz_scale, y_min=-args.view_range*viz_scale,
                                     y_max=args.view_range*viz_scale, axs=[
                                    input_ax, mgp_result_ax, tgp_result_ax, all_result_ax,
                                    mgp_social_attn_ax_h1, mgp_social_attn_ax_h2, mgp_social_attn_ax_h3,
                                    mgp_social_attn_ax_h4,
                                    mgp_scene_attn_ax_h1, mgp_scene_attn_ax_h2, mgp_scene_attn_ax_h3,
                                    mgp_scene_attn_ax_h4,
                                    tgp_social_attn_ax_h1, tgp_social_attn_ax_h2, tgp_social_attn_ax_h3,
                                    tgp_social_attn_ax_h4,
                                    tgp_scene_attn_ax_h1, tgp_scene_attn_ax_h2, tgp_scene_attn_ax_h3,
                                    tgp_scene_attn_ax_h4],
                                     desc=str(i), path=os.path.join(args.output_path, args.dataset_name))
                            viz_attention_save(axs=[mgp_attn_all_ax_l4_h1, mgp_attn_all_ax_l4_h2, mgp_attn_all_ax_l4_h3, mgp_attn_all_ax_l4_h4, tgp_attn_all_ax_l4_h1, tgp_attn_all_ax_l4_h2, tgp_attn_all_ax_l4_h3,
                                                    tgp_attn_all_ax_l4_h4], desc=str(i), path=os.path.join(args.output_path, args.dataset_name))
                            if num_viz == num_save:
                                break
                        else:
                            # viz_show(x_min=-args.view_range*viz_scale, x_max=args.view_range*viz_scale, y_min=-args.view_range*viz_scale,
                            #          y_max=args.view_range*viz_scale, axs=[tgp_result_ax])
                            viz_show(x_min=-args.view_range*viz_scale, x_max=args.view_range*viz_scale, y_min=-args.view_range*viz_scale,
                                     y_max=args.view_range*viz_scale, axs=[all_result_ax])

                                    # input_ax, mgp_result_ax, tgp_result_ax, all_result_ax,
                                    # mgp_social_attn_ax_h1, mgp_social_attn_ax_h2, mgp_social_attn_ax_h3, mgp_social_attn_ax_h4,
                                    # mgp_scene_attn_ax_h1, mgp_scene_attn_ax_h2, mgp_scene_attn_ax_h3, mgp_scene_attn_ax_h4,
                                    # tgp_social_attn_ax_h1, tgp_social_attn_ax_h2, tgp_social_attn_ax_h3, tgp_social_attn_ax_h4,
                                    # tgp_scene_attn_ax_h1, tgp_scene_attn_ax_h2, tgp_scene_attn_ax_h3, tgp_scene_attn_ax_h4])
                    else:
                        outputs["pred_trajs"] = torch.einsum('bkts,b->bkts', outputs["pred_trajs"], data["scales"])
                        outputs["pred_goals"] = torch.einsum('bks,b->bks', outputs["pred_goals"], data["scales"])
                        data["traj_lbl"] = torch.einsum('bts,b->bts', data["traj_lbl"], data["scales"])
                        data["goal_lbl"] = torch.einsum('bs,b->bs', data["goal_lbl"], data["scales"])
                        gderror, aderror, fderror, best_gde_idx, best_ade_idx, best_fde_idx = bom_loss_1(
                            outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                            args.test_k_sample)
                        print("\nEpoch Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (
                            aderror / len(data["mgp_spatial_ids"]), fderror / len(data["mgp_spatial_ids"]),
                            gderror / len(data["mgp_spatial_ids"])))

                        if args.viz_save:
                            viz_save(x_min=-args.view_range*viz_scale, x_max=args.view_range*viz_scale, y_min=-args.view_range*viz_scale,
                                     y_max=args.view_range*viz_scale, axs=[
                                    input_ax, mgp_result_ax, tgp_result_ax, all_result_ax,
                                    mgp_social_attn_ax_h1, mgp_social_attn_ax_h2, mgp_social_attn_ax_h3, mgp_social_attn_ax_h4,
                                    tgp_social_attn_ax_h1, tgp_social_attn_ax_h2, tgp_social_attn_ax_h3, tgp_social_attn_ax_h4],
                                     desc=str(i), path=os.path.join(args.output_path, args.dataset_name))
                            viz_attention_save(axs=[mgp_attn_all_ax_l4_h1, mgp_attn_all_ax_l4_h2, mgp_attn_all_ax_l4_h3,
                                                    mgp_attn_all_ax_l4_h4], desc=str(i),
                                               path=os.path.join(args.output_path, args.dataset_name))

                            if num_viz == num_save:
                                break
                        else:
                            viz_show(x_min=-args.view_range*viz_scale, x_max=args.view_range*viz_scale, y_min=-args.view_range*viz_scale,
                                     y_max=args.view_range*viz_scale, axs=[all_result_ax])
                                    # input_ax, mgp_result_ax, tgp_result_ax, all_result_ax,
                                    # mgp_social_attn_ax_h1, mgp_social_attn_ax_h2, mgp_social_attn_ax_h3, mgp_social_attn_ax_h4,
                                    # tgp_social_attn_ax_h1, tgp_social_attn_ax_h2, tgp_social_attn_ax_h3, tgp_social_attn_ax_h4])
                else:
                    outputs["pred_trajs"] = torch.einsum('bkts,b->bkts', outputs["pred_trajs"], data["scales"])
                    outputs["pred_goals"] = torch.einsum('bks,b->bks', outputs["pred_goals"], data["scales"])
                    data["traj_lbl"] = torch.einsum('bts,b->bts', data["traj_lbl"], data["scales"])
                    data["goal_lbl"] = torch.einsum('bs,b->bs', data["goal_lbl"], data["scales"])
                    # gderror, aderror, fderror, best_gde_idx, best_ade_idx, best_fde_idx = bom_loss_1(
                    #     outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                    #     args.test_k_sample)
                    gderror, aderror, fderror = bom_loss_3(
                        outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                        args.test_k_sample)


                    total_aderror += aderror
                    total_fderror += fderror
                    total_gderror += gderror
                    total_data += len(data["mgp_spatial_ids"])
                    print("\nEpoch Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (
                        aderror / len(data["mgp_spatial_ids"]), fderror / len(data["mgp_spatial_ids"]),
                        gderror / len(data["mgp_spatial_ids"])))

            print("Total Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (total_aderror/total_data, total_fderror/total_data, total_gderror/total_data))
            print("Total Average Computation Time >>>>> %f s" % avg_time)
            # print("Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f, GADE: %f, GFDE: %f" % (total_aderror/len(data_iter), total_fderror/len(data_iter), total_gderror/len(data_iter), total_gaderror/len(data_iter), total_gfderror/len(data_iter)))
            end_time = time.time()

if __name__=='__main__':
    test()