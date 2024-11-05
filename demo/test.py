import torch
# import numpy as np
import argparse
import os
import time
import tqdm
from torch.utils.data import DataLoader
from SPUBERT.dataset.ethucy import ETHUCYDataset
from SPUBERT.dataset.sdd import SDDDataset
from SPUBERT.model.spubert import (
    SPUBERTTGPConfig, SPUBERTMGPConfig, SPUBERTConfig, SPUBERTModel
)
from SPUBERT.model.loss import bom_loss
from SPUBERT.dataset.grid_map_numpy import estimate_map_length, estimate_num_patch

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy', help='dataset name (ethucy, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split for ethucy dataset(eth, hotel, univ, zara1, zara2')
    parser.add_argument('--output_path', default='./output', help='output path')
    parser.add_argument('--output_name', default='test', help='output model name')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_worker", type=int, default=4, help="dataloader worker size")
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--num_nbr", type=int, default=4, help="number of neighbors")
    parser.add_argument("--view_range", type=float, default=20.0, help="accessible range boundary of target pedestrian")
    parser.add_argument("--view_angle", type=float, default=2.09, help="accessible angle boundary of target pedestrian")
    parser.add_argument("--social_range", type=float, default=2.0, help="socially-aware range")
    parser.add_argument("--env_range", type=float, default=10.0, help="physically-aware range")
    parser.add_argument("--env_resol", type=float, default=0.2, help="physically-aware resolution")
    parser.add_argument("--patch_size", type=int, default=16, help="physically-aware patch size for ViT")
    parser.add_argument('--scene', action='store_true', help='physically-aware, true, or false ')
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layer", type=int, default=4, help="number of layers")
    parser.add_argument("--head", type=int, default=4, help="number of attention heads")
    parser.add_argument("--goal_hidden", type=int, default=64, help="goal hidden size of transformer model")
    parser.add_argument("--goal_latent", type=int, default=32, help="goal latent hidden size of transformer model")
    parser.add_argument("--k_sample", type=int, default=20, help="number of multimodal samples")
    parser.add_argument("--d_sample", type=int, default=1000, help="number of goal intention samples")

    args = parser.parse_args()
    if args.dataset_name == 'ethucy':
        print("ETH/UCY Dataset Loading...")
        test_dataset = ETHUCYDataset(split="test", args=args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    elif args.dataset_name == 'sdd':
        print("SDD Dataset Loading...")
        args.dataset_split = 'default'
        test_dataset = SDDDataset(split="test", args=args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    else:
        print("Dataset is not loaded.")

    model_path = os.path.join(args.output_path, args.dataset_name, args.dataset_split, args.output_name+".pth")

    if args.scene:
        num_patch = estimate_num_patch(estimate_map_length(args.env_range * 2, args.env_resol), args.patch_size)
    else:
        num_patch = 0
        
    spubert_tgp_cfgs = SPUBERTTGPConfig(
                hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
                pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
                patch_size=args.patch_size, view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

    spubert_mgp_cfgs = SPUBERTMGPConfig(
                hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, k_sample=args.k_sample,
                goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent, obs_len=args.obs_len,
                pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch, patch_size=args.patch_size,
                view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)
    spubert_cfgs = SPUBERTConfig(traj_cfgs=spubert_tgp_cfgs, goal_cfgs=spubert_mgp_cfgs)
    
    model = SPUBERTModel(spubert_tgp_cfgs, spubert_mgp_cfgs, spubert_cfgs)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    diff_time = 0
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.no_grad():
        model.eval()
        total_aderror = 0
        total_fderror = 0
        total_gderror = 0
        total_data = 0
        data_iter = tqdm.tqdm(enumerate(test_dataloader),
                              desc="%s" % ("TEST"),
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            data = {key: value.to(device) for key, value in data.items()}
            if args.scene:
                start_time = time.time()
                outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"], output_attentions=True, d_sample=args.d_sample)
                diff_time += (time.time() - start_time)
            else:
                start_time = time.time()
                outputs = model.inference(mgp_spatial_ids=data["mgp_spatial_ids"],
                                mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                output_attentions=True, d_sample=args.d_sample)
                diff_time += (time.time() - start_time)

            avg_time = diff_time / (i + 1)

            outputs["pred_trajs"] = torch.einsum('bkts,b->bkts', outputs["pred_trajs"], data["scales"])
            outputs["pred_goals"] = torch.einsum('bks,b->bks', outputs["pred_goals"], data["scales"])
            data["traj_lbl"] = torch.einsum('bts,b->bts', data["traj_lbl"], data["scales"])
            data["goal_lbl"] = torch.einsum('bs,b->bs', data["goal_lbl"], data["scales"])
            gderror, aderror, fderror = bom_loss(
                outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"],
                args.k_sample)

            total_aderror += aderror
            total_fderror += fderror
            total_gderror += gderror
            total_data += len(data["mgp_spatial_ids"])
            print("\nEpoch Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (
                aderror / len(data["mgp_spatial_ids"]), fderror / len(data["mgp_spatial_ids"]),
                gderror / len(data["mgp_spatial_ids"])))
            print("Epoch Average Time >>>>> ", avg_time)
        print("Total Evaluation Result >>>>> ADE: %f, FDE: %f, GDE: %f" % (total_aderror/total_data, total_fderror/total_data, total_gderror/total_data))
        print("Total Average Computation Time >>>>> %f s" % avg_time)

if __name__=='__main__':
    test()