import argparse
import os
import copy
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from SocialBERT.sbert.model.finetuner import SBertFTTrainer
from SocialBERT.sbert.dataset.ethucy_star import ETHUCYSTARDataset
from SocialBERT.sbert.dataset.ethucy_ynet import ETHUCYYNetDataset
from SocialBERT.sbert.dataset.sdd_ynet import SDDYNetDataset
from SocialBERT.sbert.util.stopper import EarlyStopping, save_model
from SocialBERT.sbert.util.config import Config


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

    # Embedding & Loss Parameters
    parser.add_argument("--sampling", type=float, default=1, help="sampling dataset")

    # Train Only
    parser.add_argument('--test', action='store_true', help='augment scenes')
    parser.add_argument('--viz', action='store_true', help='augment scenes')
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
        train_dataset = ETHUCYSTARDataset(split="train", args=args)
        val_dataset = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        if args.test:
            test_dataset = ETHUCYSTARDataset(split="test", args=test_args)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    elif args.dataset_name == 'ethucy_ynet':
        print("YNet ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYYNetDataset(split="train", args=args)
        val_dataset = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        if args.test:
            test_dataset = ETHUCYYNetDataset(split="test", args=test_args)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    elif args.dataset_name == 'sdd_ynet':
        print("YNet SDD Dataset Loading...")
        args.dataset_split = 'default'
        train_dataset = SDDYNetDataset(split="train", args=args)
        val_dataset = None # = SDDYNetDataset(split="test", args=test_args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None # = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
        if args.test:
            test_dataset = SDDYNetDataset(split="test", args=test_args)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
            val_dataloader = None
    else:
        print("Dataset is not loaded.")

    cfgs = Config(args)
    cfgs_path = cfgs.get_path(mode=args.mode)
    cfgs_path = os.path.join(args.output_path, cfgs_path)
    os.makedirs(cfgs_path, exist_ok=True)
    cfgs.save_yml_config(cfgs_path)
    tb_writer = SummaryWriter(os.path.join("logs", cfgs_path))


    trainer = SBertFTTrainer(train_dataloader=train_dataloader, val_dataloader=val_dataloader, args=args, tb_writer=tb_writer)
    stopper = EarlyStopping(patience=args.patience, ckpt_path=cfgs_path, parallel=trainer.parallel)
    min_aderror = 10000
    min_fderror = 10000
    for epoch in range(args.epoch):
        test_loss = 0
        val_loss = 0
        train_loss, params = trainer.train(epoch)
        tb_writer.add_scalar('sbert/loss/train', train_loss, epoch)
        tb_writer.add_scalar('sbert/param/lr', params['lr'], epoch)
        # validation model save
        if val_dataloader is None:
            if stopper(train_loss):
                if trainer.parallel:
                    save_model(cfgs_path, trainer.model.module, model_name="full_model")
                else:
                    save_model(cfgs_path, trainer.model, model_name="full_model")
            if stopper.early_stop and args.patience != -1:
                break
        else:
            val_loss, _ = trainer.val(epoch)
            tb_writer.add_scalar('sbert/loss/val', val_loss, epoch)
            if stopper(val_loss):
                if trainer.parallel:
                    save_model(cfgs_path, trainer.model.module, model_name="full_model")
                else:
                    save_model(cfgs_path, trainer.model, model_name="full_model")

            if stopper.early_stop and args.patience != -1:
                break
        # test
        if test_dataloader is not None and (epoch+1) % 10 == 0:
            aderror, fderror = trainer.test(epoch, test_dataloader)
            print("aderror({:.4f}), fderror({:.4f})".format(aderror, fderror))
            min_aderror = min(min_aderror, aderror)
            min_fderror = min(min_fderror, fderror)
            print("min_aderror({:.4f}), min_fderror({:.4f})".format(min_aderror, min_fderror))
            tb_writer.add_scalar('sbert/test/ade', aderror, epoch)
            tb_writer.add_scalar('sbert/test/fde', fderror, epoch)
        print("train_loss({:.4f}), val_loss({:.4f}), test_loss({:.4f})".format(train_loss, val_loss, test_loss))


    tb_writer.close()


if __name__=='__main__':
    trainer()