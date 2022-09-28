import os
import tqdm
import copy
import torch
import torch.nn as nn
import transformers
from SocialBERT.sbert.util.config import Config
from SocialBERT.sbert.model.sbert import (
SBertConfig,
SBertFTModel
)
from SocialBERT.sbert.model.trainer import SBertTrainer
from SocialBERT.sbert.model.loss import ADError, FDError

class SBertFTTrainer(SBertTrainer):
    """
    PreTrainer make the pretrained BERT model with two STM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        super().__init__(train_dataloader=train_dataloader, val_dataloader=val_dataloader, tb_writer=tb_writer,
                         args=args)
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
        self.sbert_cfgs = SBertConfig(
            input_dim=args.input_dim, output_dim=args.output_dim,
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, dropout_prob=args.dropout_prob, act_fn=args.act_fn,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range, sip=args.sip)

        self.model = SBertFTModel(self.sbert_cfgs)  # enc share // mode freeze
        if args.train_mode == "pt":
            pretrain_args = copy.copy(args)
            pretrain_args.mode = "pretrain"
            pretrain_cfgs = Config(pretrain_args)
            pretrain_path = pretrain_cfgs.get_path(mode="pretrain")
            pretrain_modelpath = os.path.join(args.output_path, pretrain_path, "sbert_model.pth")
            self.model.sbert.load_state_dict(torch.load(pretrain_modelpath))
            print("Learning SocialBERT from the pre-trained.")
        elif args.train_mode == "fs":
            print("Learning SocialBERT from scratch.")
        else:
            print("Learning SocialBERT from scratch.")

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if args.cuda and torch.cuda.device_count() > 1:
            self.parallel = True
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.parallel.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            self.parallel = False
        self.model.to(self.device)

        # for param in self.model.sbert.parameters():
        #     print(param)

        self.optim = transformers.AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.999),
                                        weight_decay=0.01)
        if args.lr_scheduler == "linear":
            self.lr_scheduler = transformers.get_scheduler(
                "linear", optimizer=self.optim, num_warmup_steps=args.warm_up, num_training_steps=len(self.train_dataloader)*args.epoch)
        elif args.lr_scheduler == "cosine":
            self.lr_scheduler = transformers.get_scheduler(
                "cosine", optimizer=self.optim, num_warmup_steps=args.warm_up, num_training_steps=len(self.train_dataloader)*args.epoch)
        elif args.lr_scheduler == "cyclic":
            self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optim, num_warmup_steps=args.warm_up, num_cycles=args.num_cycle,
                num_training_steps=args.epoch*len(self.train_dataloader))
        else:
            assert False

    def test(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            total_aderror = 0
            total_fderror = 0
            total_data = 0
            data_iter = tqdm.tqdm(enumerate(data_loader),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=len(data_loader),
                                  bar_format="{l_bar}{r_bar}")
            for i, it_data in data_iter:
                data = copy.deepcopy(it_data)
                del it_data
                data = {key: value.to(self.device) for key, value in data.items()}
                outputs = self.model(train=False, spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                     segment_ids=data["segment_ids"], attn_mask=data["attn_mask"], traj_lbl=data["traj_lbl"])

                data["traj_lbl"] = torch.einsum('ijk,i->ijk', data["traj_lbl"], data["scales"])
                outputs["pred_traj"] = torch.einsum('ijk,i->ijk', outputs["pred_traj"], data["scales"])
                total_aderror += ADError(pred_traj=outputs["pred_traj"], gt_traj=data["traj_lbl"])
                total_fderror += FDError(pred_final_pos=outputs["pred_traj"][:, -1, :].squeeze(dim=1), gt_final_pos=data["traj_lbl"][:, -1, :].squeeze(dim=1))
                total_data += len(data["spatial_ids"])
            # print(total_aderror, total_fderror, total_data)
            total_aderror = total_aderror / total_data
            total_fderror = total_fderror / total_data

            return total_aderror, total_fderror



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
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % ("train", epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        total_ade_loss = 0
        total_fde_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            data = {key: value.to(self.device) for key, value in data.items()}
            self.model.zero_grad()
            outputs = self.model(train=True, spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                 segment_ids=data["segment_ids"], attn_mask=data["attn_mask"], traj_lbl=data["traj_lbl"])
            loss = outputs["total_loss"].mean()
            total_ade_loss += outputs["ade_loss"].mean().item()
            total_fde_loss += outputs["fde_loss"].mean().item()
            total_loss += outputs["total_loss"].mean().item()
            loss.backward()
            if self.args.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.lr_scheduler.step()

        total_ade_loss = total_ade_loss / len(data_iter)
        total_fde_loss = total_fde_loss / len(data_iter)
        total_loss = total_loss / len(data_iter)
        print("total=%f, ade=%f, fde=%f" % (total_loss, total_ade_loss, total_fde_loss))
        params = {"lr": self.optim.param_groups[0]['lr']}
        return total_loss, params

    def val_iteration(self, epoch, data_loader):
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
        total_loss = 0
        total_ade_loss = 0
        total_fde_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            data = {key: value.to(self.device) for key, value in data.items()}
            outputs = self.model(spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                 segment_ids=data["segment_ids"],
                                 attn_mask=data["attn_mask"], traj_lbl=data["traj_lbl"])
            total_ade_loss += outputs["ade_loss"].mean().item()  # traj_ade_loss.item()
            total_fde_loss += outputs["fde_loss"].mean().item()  # traj_fde_loss.item()
            total_loss += outputs["total_loss"].mean().item()
        total_ade_loss = total_ade_loss / len(data_iter)
        total_fde_loss = total_fde_loss / len(data_iter)
        total_loss = total_loss / len(data_iter)
        print("total=%f, ade=%f, fde=%f" % (total_loss, total_ade_loss, total_fde_loss))
        params = {"lr": self.optim.param_groups[0]['lr']}
        return total_loss, params
