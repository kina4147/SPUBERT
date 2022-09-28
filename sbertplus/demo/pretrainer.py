import argparse
import os
import copy
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from SocialBERT.sbertplus.model.pretrainer import SBertPlusPTTrainer
from SocialBERT.sbertplus.dataset.ethucy import ETHUCYDataset
from SocialBERT.sbertplus.dataset.ethucy_tpp import ETHUCYTPPDataset
from SocialBERT.sbertplus.dataset.ethucy_star import ETHUCYSTARDataset
from SocialBERT.sbertplus.dataset.ethucy_ynet import ETHUCYYNetDataset
from SocialBERT.sbertplus.dataset.sdd_ynet import SDDYNetDataset
from SocialBERT.sbertplus.util.stopper import EarlyStopping, save_model
from SocialBERT.sbertplus.util.config import Config

def trainer():

    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy_tpp', help='dataset name (eth, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split (eth-eth, hotel, univ, zara1, zara2')
    parser.add_argument("--mode", default='pretrain', help='dataset split (eth, hotel, univ, zara1, zara2')
    parser.add_argument("--train_mode", default='all', help='all, mtp')
    parser.add_argument('--output_path', default='./output/sbertplus', help='glob expression for data files')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA: true, or false")

    # Training Parameters
    parser.add_argument('--aug', action='store_true', default=True, help='augment scenes')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="learning rate of adam")
    parser.add_argument('--warm_up', default=0, type=float, help='sample ratio of train/val scenes')
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs")
    parser.add_argument("--num_worker", type=int, default=4, help="dataloader worker size")
    parser.add_argument("--patience", type=int, default=-1, help="patience for early stopping")
    parser.add_argument("--lr_scheduler", default='it_linear', help="patience for early stopping")
    parser.add_argument('--clip_grads', action='store_true', default=True, help='augment scenes')

    # Model Paramters
    parser.add_argument('--input_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument('--goal_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument('--output_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--min_obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--num_nbr", type=int, default=4, help="number of neighbors")
    parser.add_argument("--view_range", type=float, default=20.0, help="accessible range of target pedestrian")
    parser.add_argument("--view_angle", type=float, default=2.09, help="accessible range of target pedestrian")
    parser.add_argument("--social_range", type=float, default=2.0, help="socially-aware range")

    parser.add_argument("--env_range", type=float, default=10.0, help="socially-aware range")
    parser.add_argument("--env_resol", type=float, default=0.2, help="socially-aware range")
    parser.add_argument("--patch_size", type=int, default=16, help="socially-aware range")
    parser.add_argument('--scene', action='store_true', help='glob expression for data files')

    # Ablation Setting Parameters
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layer", type=int, default=4, help="number of layers")
    parser.add_argument("--head", type=int, default=4, help="number of attention heads")
    parser.add_argument('--act_fn', default='relu', help='glob expression for data files')
    parser.add_argument('--sip', action='store_true', help='augment scenes')
    parser.add_argument("--col_weight", type=float, default=0.0, help="learning rate of adam")
    parser.add_argument("--traj_weight", type=float, default=1.0, help="learning rate of adam")
    # Embedding & Loss Parameters
    parser.add_argument("--sampling", type=float, default=1, help="sampling dataset")

    # Train Only
    parser.add_argument('--viz', action='store_true', help='augment scenes')
    parser.add_argument("--seed", type=int, default=0, help="embedding size")
    parser.add_argument('--binary_scene', action='store_true', help='augment scenes')


    args = parser.parse_args()
    print(args)
    if args.seed > 0:
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
    if args.dataset_name == 'ethucy':
        print("Original ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYDataset(split="train", args=args)
        # val_dataset = ETHUCYDataset(split="val", args=test_args)
        # indices = list(range(len(dataset)))
        # indices = indices[:int(len(indices) * args.sampling)]
        # val_split = int(np.floor(0.2 * len(indices)))
        # train_indices, val_indices = indices[val_split:], indices[:val_split]
        # train_sampler = SubsetRandomSampler(train_indices)
        # valid_sampler = SubsetRandomSampler(val_indices)
        # train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_worker,
        #                               sampler=train_sampler)
        # val_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_worker,
        #                             sampler=valid_sampler)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None # DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
        if args.test:
            test_dataset = ETHUCYDataset(split="test", args=test_args)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    elif args.dataset_name == 'ethucy_tpp':
        print("Trajectron++ ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYTPPDataset(split="train", args=args)
        # val_dataset = ETHUCYTPPDataset(split="val", args=test_args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None #DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
        # if args.test:
        #     test_dataset = ETHUCYTPPDataset(split="test", args=test_args)
        #     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
        #                                   shuffle=False)
    elif args.dataset_name == 'ethucy_star':
        print("STAR ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYSTARDataset(split="train", args=args)
        val_dataset = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        # if args.test:
        #     test_dataset = ETHUCYSTARDataset(split="test", args=test_args)
        #     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    elif args.dataset_name == 'ethucy_ynet':
        print("YNet ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYYNetDataset(split="train", args=args)
        val_dataset = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
    elif args.dataset_name == 'sdd_ynet':
        print("YNet SDD Dataset Loading...")
        args.dataset_split = 'default'
        train_dataset = SDDYNetDataset(split="train", args=args)
        val_dataset = None # test_argsSDDYNetDataset(split="test", args=test_args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None # DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    else:
        print("Dataset is not loaded.")


    cfgs = Config(args)
    cfgs_path = cfgs.get_path(mode=args.mode)
    ckpt_path = os.path.join(args.output_path, cfgs_path)
    os.makedirs(ckpt_path, exist_ok=True)
    cfgs.save_yml_config(ckpt_path)
    tb_writer = SummaryWriter(os.path.join("logs", cfgs_path))

    trainer = SBertPlusPTTrainer(train_dataloader=train_dataloader, val_dataloader=val_dataloader, args=args, tb_writer=tb_writer)
    stopper = EarlyStopping(patience=args.patience, ckpt_path=ckpt_path, parallel=trainer.parallel)
    for epoch in range(args.epoch):
        val_loss = 0
        train_loss, params = trainer.train(epoch)
        tb_writer.add_scalar('sbert/loss/train', train_loss, epoch)
        tb_writer.add_scalar('sbert/param/lr', params['lr'], epoch)
        if val_dataloader is None:
            if stopper(train_loss):
                if trainer.parallel:
                    save_model(ckpt_path, trainer.model.module.sbert, model_name="sbert_model", parallel=trainer.parallel)
                else:
                    save_model(ckpt_path, trainer.model.sbert, model_name="sbert_model", parallel=trainer.parallel)
        else:
            val_loss, _ = trainer.val(epoch)
            tb_writer.add_scalar('sbert/loss/val', val_loss, epoch)
            if stopper(val_loss):
                if trainer.parallel:
                    save_model(ckpt_path, trainer.model.module.sbert, model_name="sbert_model", parallel=trainer.parallel)
                else:
                    save_model(ckpt_path, trainer.model.sbert, model_name="sbert_model", parallel=trainer.parallel)

        if stopper.early_stop and args.patience != -1:
            break

        print("train_loss({:.4f}), val_loss({:.4f})".format(train_loss, val_loss))
    tb_writer.close()


if __name__=='__main__':
    trainer()