import argparse
import os
import copy
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from SocialBERT.sbertplus.dataset.ethucy import ETHUCYDataset
from SocialBERT.sbertplus.dataset.ethucy_tpp import ETHUCYTPPDataset
from SocialBERT.sbertplus.dataset.ethucy_ynet import ETHUCYYNetDataset
from SocialBERT.sbertplus.dataset.ethucy_star import ETHUCYSTARDataset
from SocialBERT.sbertplus.dataset.sdd_ynet import SDDYNetDataset
from SocialBERT.sbertplus.model.finetuner import SBertPlusFTTrainer

from SocialBERT.sbertplus.util.stopper import EarlyStopping, save_model
from SocialBERT.sbertplus.util.config import Config
def trainer():

    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='ethucy_ynet', help='dataset name (eth, sdd)')
    parser.add_argument("--dataset_split", default='univ', help='dataset split (eth-eth, hotel, univ, zara1, zara2')
    parser.add_argument("--mode", default='finetune', help='dataset split (eth, hotel, univ, zara1, zara2')
    parser.add_argument("--train_mode", default='fs', help='mtp_sep, mtp_shr, tgp, mgp, tgp_mgp')
    parser.add_argument('--output_path', default='./output/sbertplus', help='glob expression for data files')
    parser.add_argument("--cuda", action='store_true', help="training with CUDA: true, or false")
    parser.add_argument("--share", action='store_true', help='augment scenes')
    parser.add_argument('--freeze', action='store_true', help='augment scenes')

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
    parser.add_argument("--param_last_epoch", type=float, default=0, help="learning rate of adam")
    parser.add_argument('--clip_grads', action='store_true', default=True, help='augment scenes')

    # Model Paramters
    parser.add_argument('--input_dim', type=int, default=2, help="number of batch_size")
    parser.add_argument('--goal_dim', type=int, default=2, help="--number of batch_size")
    parser.add_argument('--output_dim', type=int, default=2, help="number of batch_size")
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

    # Ablation Setting Parameters
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layer", type=int, default=4, help="number of layers")
    parser.add_argument("--head", type=int, default=4, help="number of attention heads")
    # parser.add_argument("--goal_layer", type=int, default=4, help="number of layers")
    # parser.add_argument("--goal_head", type=int, default=4, help="number of attention heads")

    parser.add_argument("--goal_hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--goal_latent", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--act_fn', default='relu', help='glob expression for data files')

    # Hyperparameters
    parser.add_argument('--sip', action='store_true', help='augment scenes')
    parser.add_argument("--cvae_sigma", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--kld_clamp", type=float, default=None, help="learning rate of adam")
    parser.add_argument("--col_weight", type=float, default=0.0, help="learning rate of adam")
    parser.add_argument("--traj_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--goal_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--kld_weight", type=float, default=1.0, help="learning rate of adam")
    parser.add_argument("--num_cycle", type=int, default=0, help="learning rate of adam")
    parser.add_argument('--normal', action='store_true', help='augment scenes')

    # Embedding & Loss Parameters
    parser.add_argument("--sampling", type=float, default=1, help="sampling dataset")
    parser.add_argument("--k_sample", type=int, default=20, help="embedding size")
    parser.add_argument("--d_sample", type=int, default=400, help="embedding size")

    # Train Only
    parser.add_argument('--test', action='store_true', help='augment scenes')
    parser.add_argument('--viz', action='store_true', help='augment scenes')
    parser.add_argument("--seed", type=int, default=0, help="embedding size")
    parser.add_argument('--binary_scene', action='store_true', help='augment scenes')

    args = parser.parse_args()
    print(args)
    # random_seed = args.seed
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    test_args = copy.copy(args)
    test_args.aug = False
    test_dataloader = None
    if args.dataset_name == 'ethucy':
        print("Original ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYDataset(split="train", args=args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        test_dataset = ETHUCYDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    elif args.dataset_name == 'ethucy_tpp':
        print("Trajectron++ ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYTPPDataset(split="train", args=args)
        val_dataset = ETHUCYTPPDataset(split="val", args=test_args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
        if args.test:
            test_dataset = ETHUCYTPPDataset(split="test", args=test_args)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                              shuffle=False)
    elif args.dataset_name == 'ethucy_star':
        print("STAR ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYSTARDataset(split="train", args=args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        # if args.test:
        test_dataset = ETHUCYSTARDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    elif args.dataset_name == 'ethucy_ynet':
        print("YNet ETH/UCY Dataset Loading...")
        train_dataset = ETHUCYYNetDataset(split="train", args=args)
        val_dataset = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None
        # if args.test:
        test_dataset = ETHUCYYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
    elif args.dataset_name == 'sdd_ynet':
        print("YNet SDD Dataset Loading...")
        args.dataset_split = 'default'
        train_dataset = SDDYNetDataset(split="train", args=args)
        # val_dataset = SDDYNetDataset(split="test", args=test_args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker,
                                      shuffle=True)
        val_dataloader = None # DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)
        # if args.test:
        test_dataset = SDDYNetDataset(split="test", args=test_args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=False)

    else:
        print("Dataset is not loaded.")

    if args.seed > 0:
        random_seed = args.seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)


    cfgs = Config(args)
    cfgs_path = cfgs.get_path(mode=args.mode)
    cfgs_path = os.path.join(args.output_path, cfgs_path)
    os.makedirs(cfgs_path, exist_ok=True)
    cfgs.save_yml_config(cfgs_path)
    tb_writer = SummaryWriter(os.path.join("logs", cfgs_path))


    trainer = SBertPlusFTTrainer(train_dataloader=train_dataloader, val_dataloader=val_dataloader, args=args, tb_writer=tb_writer)
    stopper = EarlyStopping(patience=args.patience, ckpt_path=cfgs_path, parallel=trainer.parallel, verbose=True)
    min_aderror = 10000
    min_fderror = 10000
    min_gderror = 10000
    for epoch in range(args.epoch):
        test_loss = 0
        val_loss = 0
        train_loss, params = trainer.train(epoch)
        tb_writer.add_scalar('loss/train/total', train_loss, epoch)
        tb_writer.add_scalar('loss/train/kld', params['kld_loss'], epoch)
        tb_writer.add_scalar('loss/train/gde', params['gde_loss'], epoch)
        tb_writer.add_scalar('loss/train/mgp_col', params['mgp_col_loss'], epoch)
        tb_writer.add_scalar('loss/train/ade', params['ade_loss'], epoch)
        tb_writer.add_scalar('loss/train/fde', params['fde_loss'], epoch)
        tb_writer.add_scalar('loss/train/tgp_col', params['tgp_col_loss'], epoch)
        tb_writer.add_scalar('lr/mgp_lr', params['mgp_lr'], epoch)
        tb_writer.add_scalar('lr/tgp_lr', params['tgp_lr'], epoch)
        tb_writer.add_scalar('weight/kld', params['kld_weight'], epoch)
        tb_writer.add_scalar('weight/ade', params['traj_weight'], epoch)
        tb_writer.add_scalar('weight/gde', params['goal_weight'], epoch)

        # validation model save
        if test_dataloader is not None and (epoch+1) % 1 == 0 and args.test and epoch > args.epoch/2:
            aderror, fderror, gderror = trainer.test(epoch, test_dataloader, args.d_sample, args.k_sample)
            print("[TEST] ADE({:.4f}), FDE({:.4f}), GDE({:.4f})".format(aderror, fderror, gderror))
            # min_aderror = min(min_aderror, aderror)
            # min_fderror = min(min_fderror, fderror)
            # min_gderror = min(min_gderror, gderror)
            # print("min_aderror({:.4f}), min_fderror({:.4f}), min_gderror({:.4f})".format(min_aderror, min_fderror, min_gderror))
            tb_writer.add_scalar('test/ade', aderror, epoch)
            tb_writer.add_scalar('test/fde', fderror, epoch)
            tb_writer.add_scalar('test/gde', gderror, epoch)
            if stopper(aderror):
                if trainer.parallel:
                    stopper.save_model(cfgs_path, trainer.model.module, model_name="full_model")
                else:
                    stopper.save_model(cfgs_path, trainer.model, model_name="full_model")

            if stopper.early_stop and args.patience != -1:
                break

        elif not args.test and val_dataloader is None:
            if stopper(train_loss):
                if trainer.parallel:
                    stopper.save_model(cfgs_path, trainer.model.module, model_name="full_model")
                else:
                    stopper.save_model(cfgs_path, trainer.model, model_name="full_model")

            if stopper.early_stop and args.patience != -1:
                break

        elif val_dataloader is not None and not args.test:
            val_loss, params = trainer.val(epoch)
            if 'epl_' in args.lr_scheduler:
                trainer.mgp_lr_scheduler.step(val_loss)
                trainer.tgp_lr_scheduler.step(val_loss)
            tb_writer.add_scalar('loss/val/total', val_loss, epoch)
            tb_writer.add_scalar('loss/val/kld', params['kld_loss'], epoch)
            tb_writer.add_scalar('loss/val/gde', params['gde_loss'], epoch)
            tb_writer.add_scalar('loss/val/mgp_col', params['mgp_col_loss'], epoch)
            tb_writer.add_scalar('loss/val/ade', params['ade_loss'], epoch)
            tb_writer.add_scalar('loss/val/fde', params['fde_loss'], epoch)
            tb_writer.add_scalar('loss/val/tgp_col', params['tgp_col_loss'], epoch)
            if stopper(val_loss):
                if trainer.parallel:
                    stopper.save_model(cfgs_path, trainer.model.module, model_name="full_model")
                else:
                    stopper.save_model(cfgs_path, trainer.model, model_name="full_model")

            if stopper.early_stop and args.patience != -1:
                break
        else:
            if 'epl_' in args.lr_scheduler:
                print("[ERROR] lr scheduler with val loss should be with validation loader.")
                break
    tb_writer.close()


if __name__=='__main__':
    trainer()