import os
import tqdm
import copy

import torch
import torch.nn as nn
import transformers
from SPUBERT.model.trainer import SPUBERTTrainer
from SPUBERT.model.spubert import (
    SPUBERTMGPConfig,
    SPUBERTTGPConfig,
    SPUBERTFTConfig,
    SPUBERTFTModel,
)
from SPUBERT.dataset.grid_map_numpy import estimate_map_length, estimate_num_patch
from SPUBERT.util.config import Config
from SPUBERT.model.loss import bom_loss_1, bom_loss_2, bom_loss_3
from SPUBERT.util.scheduler import CyclicScherduler, frange_cycle_sigmoid, frange_cycle_cosine
transformers.logging.set_verbosity_info()

from SPUBERT.util.viz import *
torch.autograd.set_detect_anomaly(True)

class SPUBERTFTTrainer(SPUBERTTrainer):
    """
    PreTrainer make the pretrained BERT model with two STM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer, args=args)
        """
        :param bert: BERT model which you want to train
        :param spatial_size: total spatial size to embed positional space
        :param train_dataloader: train gen data loader
        :param val_dataloader: test gen data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        num_patch = estimate_num_patch(estimate_map_length(args.env_range*2, args.env_resol), args.patch_size)
        self.sbert_tgp_cfgs = SPUBERTTGPConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch, dropout_prob=args.dropout_prob,
            patch_size=args.patch_size, col_weight=args.col_weight, traj_weight=args.traj_weight, act_fn=args.act_fn,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

        self.sbert_mgp_cfgs = SPUBERTMGPConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, k_sample=args.k_sample,
            goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
            dropout_prob=args.dropout_prob, patch_size=args.patch_size, kld_weight=args.kld_weight, col_weight=args.col_weight,
            kld_clamp=args.kld_clamp, cvae_sigma=args.cvae_sigma, goal_weight=args.goal_weight, act_fn=args.act_fn,
            normal=args.normal, view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

        self.sbert_cfgs = SPUBERTFTConfig(traj_cfgs=self.sbert_tgp_cfgs, goal_cfgs=self.sbert_mgp_cfgs, share=args.share)
        self.model = SPUBERTFTModel(self.sbert_tgp_cfgs, self.sbert_mgp_cfgs, self.sbert_cfgs) # enc share // mode freeze

        pretrain_args = copy.copy(args)
        pretrain_args.mode = "pretrain"
        if args.train_mode == "pt":
            pretrain_cfgs = Config(pretrain_args)
            pretrain_path = pretrain_cfgs.get_path(mode="pretrain")
            pretrain_modelpath = os.path.join(args.output_path, pretrain_path, "sbert_model.pth")
            self.model.tgp_model.sbert.load_state_dict(torch.load(pretrain_modelpath))
            self.model.mgp_model.sbert.load_state_dict(torch.load(pretrain_modelpath))
        else:
            print("from_scatch")

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if args.cuda and torch.cuda.device_count() > 1:
            self.parallel = True
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.parallel.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            self.parallel = False
        self.model.to(self.device)


        self.mgp_optim = transformers.AdamW(self.model.mgp_model.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01) #, no_deprecation_warning=True) #, correct_bias=False)
        self.tgp_optim = transformers.AdamW(self.model.tgp_model.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01) #, no_deprecation_warning=True) #, correct_bias=False)
        # self.mgp_optim = transformers.AdamW(self.model.mgp_model.parameters(), lr=args.lr, no_deprecation_warning=True)
        # self.tgp_optim = transformers.AdamW(self.model.tgp_model.parameters(), lr=args.lr, no_deprecation_warning=True)

        if args.lr_scheduler == "it_linear":
            self.mgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.mgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch * len(self.train_dataloader))
            self.tgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.tgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch * len(self.train_dataloader))
        elif args.lr_scheduler == "it_cosine":
            self.mgp_lr_scheduler = transformers.get_scheduler("cosine", optimizer=self.mgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch * len(self.train_dataloader))
            self.tgp_lr_scheduler = transformers.get_scheduler("cosine", optimizer=self.tgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch * len(self.train_dataloader))
        elif args.lr_scheduler == "ep_linear":
            self.mgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.mgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch)
            self.tgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.tgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch)
        elif args.lr_scheduler == "ep_cosine":
            self.mgp_lr_scheduler = transformers.get_scheduler("cosine", optimizer=self.mgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch)
            self.tgp_lr_scheduler = transformers.get_scheduler("cosine", optimizer=self.tgp_optim,
                                                           num_warmup_steps=args.warm_up,
                                                           num_training_steps=args.epoch)
        elif args.lr_scheduler == "epl_reduce":
            self.mgp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.mgp_optim, factor=0.2, patience=5, min_lr=1e-10)
            self.tgp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.tgp_optim, factor=0.2, patience=5, min_lr=1e-10)

        # elif args.lr_scheduler == "cyclic":
        #     self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        #         optimizer=self.optim, num_warmup_steps=args.warm_up, num_cycles=args.num_cycle, num_training_steps=args.epoch*len(self.train_dataloader))
        # elif args.lr_scheduler == "exp":
        #     self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.999)
        # elif args.lr_scheduler == "lambda":
        #     self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optim, lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            assert False

        self.kld_weight_scheduler = CyclicScherduler(name="kld_weight", end_val=args.kld_weight, epoch=args.epoch, #*len(self.train_dataloader),
                                                     num_gpu=torch.cuda.device_count(), func=frange_cycle_sigmoid, num_cycle=0) #args.num_cycle)
        self.traj_weight_scheduler = CyclicScherduler(name="traj_weight", end_val=args.traj_weight, epoch=args.epoch, # *len(self.train_dataloader),
                                                     num_gpu=torch.cuda.device_count(), func=frange_cycle_sigmoid, num_cycle=0) #args.num_cycle)
        self.goal_weight_scheduler = CyclicScherduler(name="goal_weight", end_val=args.goal_weight, epoch=args.epoch, #*len(self.train_dataloader),
                                                     num_gpu=torch.cuda.device_count(), func=frange_cycle_cosine, num_cycle=0, reverse=True) #args.num_cycle)

    def test(self, epoch, data_loader, d_sample, k_sample):
        self.model.eval()
        with torch.no_grad():
            total_aderror = 0
            total_fderror = 0
            total_gderror = 0
            total_data = 0
            data_iter = tqdm.tqdm(enumerate(data_loader),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=len(data_loader),
                                  bar_format="{l_bar}{r_bar}")
            for i, it_data in data_iter:
                data = copy.deepcopy(it_data)
                del it_data
                data = {key: value.to(self.device) for key, value in data.items()}
                if self.args.scene:
                    outputs = self.model.inference(
                        mgp_spatial_ids=data["mgp_spatial_ids"], mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"],
                        mgp_attn_mask=data["mgp_attn_mask"], tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"],
                        tgp_attn_mask=data["tgp_attn_mask"], env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                        env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"], d_sample=d_sample)
                else:
                    outputs = self.model.inference(
                        mgp_spatial_ids=data["mgp_spatial_ids"], mgp_temporal_ids=data["mgp_temporal_ids"], mgp_segment_ids=data["mgp_segment_ids"],
                        mgp_attn_mask=data["mgp_attn_mask"], tgp_temporal_ids=data["tgp_temporal_ids"], tgp_segment_ids=data["tgp_segment_ids"],
                        tgp_attn_mask=data["tgp_attn_mask"], d_sample=d_sample)

                outputs["pred_trajs"] = torch.einsum('bkts,b->bkts', outputs["pred_trajs"], data["scales"])
                outputs["pred_goals"] = torch.einsum('bks,b->bks', outputs["pred_goals"], data["scales"])
                data["traj_lbl"] = torch.einsum('bts,b->bts', data["traj_lbl"], data["scales"])
                data["goal_lbl"] = torch.einsum('bs,b->bs', data["goal_lbl"], data["scales"])
                gderror, aderror, fderror = bom_loss_3(outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"], k_sample, output_dim=self.args.output_dim)
                total_aderror += aderror
                total_fderror += fderror
                total_gderror += gderror
                total_data += len(data["mgp_spatial_ids"])

            total_aderror = total_aderror/total_data
            total_fderror = total_fderror/total_data
            total_gderror = total_gderror/total_data

            return total_aderror, total_fderror, total_gderror

    def train_iteration(self, epoch, data_loader):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: epoch_loss
        """
        # str_code = "train" if train else "val"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % ("train", epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # total_loss = 0
        total_mgp_loss = 0
        total_tgp_loss = 0
        total_tgp_col_loss = 0
        total_mgp_col_loss = 0
        total_kld_loss = 0
        total_gde_loss = 0
        total_ade_loss = 0
        total_fde_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            kld_weight = self.kld_weight_scheduler.get_param(train=True).to(self.device)
            traj_weight = self.traj_weight_scheduler.get_param(train=True).to(self.device)
            goal_weight = self.goal_weight_scheduler.get_param(train=True).to(self.device)
            data = {key: value.to(self.device) for key, value in data.items()}
            self.mgp_optim.zero_grad()
            self.tgp_optim.zero_grad()
            if self.args.scene:
                outputs = self.model(mgp_spatial_ids=data["mgp_spatial_ids"],
                                     mgp_temporal_ids=data["mgp_temporal_ids"],
                                     mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                     tgp_spatial_ids=data["tgp_spatial_ids"], tgp_temporal_ids=data["tgp_temporal_ids"],
                                     tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                     env_spatial_ids=data["env_spatial_ids"],
                                     env_temporal_ids=data["env_temporal_ids"],
                                     env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"],
                                     traj_lbl=data["traj_lbl"], goal_lbl=data["goal_lbl"], envs=data["envs"],
                                     envs_params=data["envs_params"], kld_weight=kld_weight,
                                     traj_weight=traj_weight, goal_weight=goal_weight)
                if self.args.col_weight > 0:
                    total_mgp_col_loss += outputs["mgp_col_loss"].mean().item()
                    total_tgp_col_loss += outputs["tgp_col_loss"].mean().item()
            else:
                outputs = self.model(mgp_spatial_ids=data["mgp_spatial_ids"],
                                     mgp_temporal_ids=data["mgp_temporal_ids"],
                                     mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                     tgp_spatial_ids=data["tgp_spatial_ids"], tgp_temporal_ids=data["tgp_temporal_ids"],
                                     tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                     traj_lbl=data["traj_lbl"], goal_lbl=data["goal_lbl"], kld_weight=kld_weight,
                                     traj_weight=traj_weight, goal_weight=goal_weight)

            mgp_loss = outputs["mgp_loss"].mean()
            tgp_loss = outputs["tgp_loss"].mean()
            total_kld_loss += outputs["kld_loss"].mean().item()  # kld_loss.item()
            total_ade_loss += outputs["ade_loss"].mean().item()  # traj_ade_loss.item()
            total_fde_loss += outputs["fde_loss"].mean().item()  # traj_fde_loss.item()
            total_gde_loss += outputs["gde_loss"].mean().item()  # gde_loss.item()
            total_mgp_loss += outputs["mgp_loss"].mean().item()
            total_tgp_loss += outputs["tgp_loss"].mean().item()
            mgp_loss.backward()
            tgp_loss.backward()
            if self.args.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.mgp_optim.step()
            self.tgp_optim.step()
            if 'it_' in self.args.lr_scheduler:
                self.mgp_lr_scheduler.step()
                self.tgp_lr_scheduler.step()
        if 'ep_' in self.args.lr_scheduler:
            self.mgp_lr_scheduler.step()
            self.tgp_lr_scheduler.step()

        self.kld_weight_scheduler.step()
        self.traj_weight_scheduler.step()
        self.goal_weight_scheduler.step()
        total_kld_loss = total_kld_loss / len(data_iter)
        total_ade_loss = total_ade_loss / len(data_iter)
        total_fde_loss = total_fde_loss / len(data_iter)
        total_gde_loss = total_gde_loss / len(data_iter)
        total_mgp_loss = total_mgp_loss / len(data_iter)
        total_tgp_loss = total_tgp_loss / len(data_iter)
        if self.args.scene and self.args.col_weight > 0:
            total_mgp_col_loss = total_mgp_col_loss / len(data_iter)
            total_tgp_col_loss = total_tgp_col_loss / len(data_iter)
        total_loss = total_mgp_loss + total_tgp_loss
        # total_loss = total_loss / len(data_iter)
        print("[MGP] total_mgp=%f, kld=%f, gde=%f, col=%f" % (total_mgp_loss, total_kld_loss, total_gde_loss, total_mgp_col_loss))
        print("[TGP] total_tgp=%f, ade=%f, fde=%f, col=%f" % (total_tgp_loss, total_ade_loss, total_fde_loss, total_tgp_col_loss))
        params = {"total_loss": total_loss,
                  "traj_weight": traj_weight, "goal_weight": goal_weight, "kld_weight": kld_weight,
                  "kld_loss": total_kld_loss, "gde_loss": total_gde_loss, "mgp_col_loss": total_mgp_col_loss,
                  "ade_loss": total_ade_loss, "fde_loss": total_fde_loss, "tgp_col_loss": total_tgp_col_loss,
                  "mgp_lr": self.mgp_optim.param_groups[0]['lr'], "tgp_lr": self.tgp_optim.param_groups[0]['lr']}
        return total_loss, params

    def val_iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: epoch_loss
        """
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % ("val", epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_mgp_loss = 0
        total_tgp_loss = 0
        total_tgp_col_loss = 0
        total_mgp_col_loss = 0
        total_kld_loss = 0
        total_gde_loss = 0
        total_ade_loss = 0
        total_fde_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            kld_weight = self.kld_weight_scheduler.get_param(train=False).to(self.device)
            traj_weight = self.traj_weight_scheduler.get_param(train=False).to(self.device)
            goal_weight = self.goal_weight_scheduler.get_param(train=False).to(self.device)
            data = {key: value.to(self.device) for key, value in data.items()}
            if self.args.scene:
                outputs = self.model(mgp_spatial_ids=data["mgp_spatial_ids"], mgp_temporal_ids=data["mgp_temporal_ids"],
                                     mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                     tgp_spatial_ids=data["tgp_spatial_ids"], tgp_temporal_ids=data["tgp_temporal_ids"],
                                     tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                     env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                     env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"],
                                     traj_lbl=data["traj_lbl"], goal_lbl=data["goal_lbl"], envs=data["envs"],
                                     envs_params=data["envs_params"], kld_weight=kld_weight,
                                     traj_weight=traj_weight, goal_weight=goal_weight)
                if self.args.col_weight > 0:
                    total_mgp_col_loss += outputs["mgp_col_loss"].mean().item()  # goal_col_loss.item()
                    total_tgp_col_loss += outputs["tgp_col_loss"].mean().item()  # traj_col_loss.item()
            else:
                outputs = self.model(mgp_spatial_ids=data["mgp_spatial_ids"], mgp_temporal_ids=data["mgp_temporal_ids"],
                                     mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                     tgp_spatial_ids=data["tgp_spatial_ids"], tgp_temporal_ids=data["tgp_temporal_ids"],
                                     tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                     traj_lbl=data["traj_lbl"], goal_lbl=data["goal_lbl"], kld_weight=kld_weight,
                                     traj_weight=traj_weight, goal_weight=goal_weight)

            total_kld_loss += outputs["kld_loss"].mean().item()  # kld_loss.item()
            total_ade_loss += outputs["ade_loss"].mean().item()  # traj_ade_loss.item()
            total_fde_loss += outputs["fde_loss"].mean().item()  # traj_fde_loss.item()
            total_gde_loss += outputs["gde_loss"].mean().item()  # gde_loss.item()
            total_mgp_loss += outputs["mgp_loss"].mean().item()
            total_tgp_loss += outputs["tgp_loss"].mean().item()

        total_kld_loss = total_kld_loss / len(data_iter)
        total_ade_loss = total_ade_loss / len(data_iter)
        total_fde_loss = total_fde_loss / len(data_iter)
        total_gde_loss = total_gde_loss / len(data_iter)
        total_mgp_loss = total_mgp_loss / len(data_iter)
        total_tgp_loss = total_tgp_loss / len(data_iter)
        if self.args.scene and self.args.col_weight > 0:
            total_mgp_col_loss = total_mgp_col_loss / len(data_iter)
            total_tgp_col_loss = total_tgp_col_loss / len(data_iter)
        total_loss = total_mgp_loss + total_tgp_loss
        print("[MGP] total_mgp=%f, kld=%f, gde=%f, col=%f" % (total_mgp_loss, total_kld_loss, total_gde_loss, total_mgp_col_loss))
        print("[TGP] total_tgp=%f, ade=%f, fde=%f, col=%f" % (total_tgp_loss, total_ade_loss, total_fde_loss, total_tgp_col_loss))
        params = {"total_loss": total_loss,
                  "traj_weight": traj_weight, "goal_weight": goal_weight, "kld_weight": kld_weight,
                  "kld_loss": total_kld_loss, "gde_loss": total_gde_loss, "mgp_col_loss": total_mgp_col_loss,
                  "ade_loss": total_ade_loss, "fde_loss": total_fde_loss, "tgp_col_loss": total_tgp_col_loss}
        return total_loss, params
