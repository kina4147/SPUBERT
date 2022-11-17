import argparse
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from SPUBERT.dataset.ethucy import ETHUCYDataset
from SPUBERT.dataset.sdd import SDDDataset
from SPUBERT.model.trainer import SPUBERTTrainer

from SPUBERT.util.stopper import EarlyStopping

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy', help='dataset name (ethucy, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split for ethucy dataset(eth, hotel, univ, zara1, zara2')
    parser.add_argument('--output_path', default='./output', help='output path')
    parser.add_argument('--output_name', default='test', help='output model name')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of AdamW")
    parser.add_argument('--warm_up', default=1000, type=float, help='warmup iteration')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=400, help="number of epochs")
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
    parser.add_argument("--seed", type=int, default=80819, help="random seed")
    parser.add_argument("--patience", type=int, default=-1, help="patience for early stopping")
    args = parser.parse_args()

    if args.dataset_name == 'ethucy':
        print("ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYDataset(split="train", args=args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        test_dataset = ETHUCYDataset(split="test", args=args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    elif args.dataset_name == 'sdd':
        print("SDD Dataset Loading...")
        args.dataset_split = 'default'
        train_dataset = SDDDataset(split="train", args=args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        test_dataset = SDDDataset(split="test", args=args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    else:
        print("Dataset is not loaded.")

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    model_path = os.path.join(args.output_path, args.dataset_name, args.dataset_split)
    os.makedirs(model_path, exist_ok=True)

    trainer = SPUBERTTrainer(train_dataloader=train_dataloader, args=args)
    stopper = EarlyStopping(patience=args.patience, ckpt_path=model_path, verbose=True)
    for epoch in range(args.epoch):
        trainer.train(epoch)

        if epoch > args.epoch/2:
            aderror, fderror, gderror = trainer.test(epoch, test_dataloader, args.d_sample, args.k_sample)
            print("[TEST] ADE({:.4f}), FDE({:.4f}), GDE({:.4f})".format(aderror, fderror, gderror))
            if stopper(aderror, fderror, epoch):
                stopper.save_model(model_path, trainer.model, model_name=args.output_name)

            if stopper.early_stop and args.patience != -1:
                break


if __name__=='__main__':
    train()