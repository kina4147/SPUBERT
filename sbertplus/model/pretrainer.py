import torch
import torch.nn as nn
import transformers
import tqdm
import copy

from SocialBERT.sbertplus.model.trainer import SBertTrainer
from SocialBERT.sbertplus.model.sbertplus import (
SBertPlusPTConfig, SBertPlusPTModel
)
from SocialBERT.sbertplus.dataset.grid_map_numpy import estimate_map_length, estimate_num_patch

class SBertPlusPTTrainer(SBertTrainer):
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
        self.sbert_cfgs = SBertPlusPTConfig(
            input_dim=args.input_dim, output_dim=args.output_dim, goal_dim=args.goal_dim,
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, dropout_prob=args.dropout_prob, act_fn=args.act_fn,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range, sip=args.sip,
            scene = args.scene, num_patch = num_patch, patch_size = args.patch_size,
            col_weight = args.col_weight, traj_weight = args.traj_weight)

        self.model = SBertPlusPTModel(self.sbert_cfgs)



        if args.cuda and torch.cuda.device_count() > 1:
            self.parallel = True
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.parallel.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            self.parallel = False
        self.model.to(self.device)

        self.optim = transformers.AdamW(self.model.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01) #, no_deprecation_warning=True) # , correct_bias=False)
        # self.optim = transformers.AdamW(self.model.parameters(), lr=args.lr, no_deprecation_warning=True)
        if args.lr_scheduler == "it_linear":
            self.lr_scheduler = transformers.get_scheduler("linear", optimizer=self.optim, num_warmup_steps=args.warm_up, num_training_steps=args.epoch*len(self.train_dataloader))
        elif args.lr_scheduler == "it_cosine":
            self.lr_scheduler = transformers.get_scheduler("cosine", optimizer=self.optim, num_warmup_steps=args.warm_up, num_training_steps=args.epoch*len(self.train_dataloader))
        # elif args.lr_scheduler == "cyclic":
        #     self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        #         optimizer=self.optim, num_warmup_steps=args.warm_up, num_cycles=args.num_cycle, num_training_steps=args.epoch*len(self.train_dataloader))
        else:
            assert False


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

        total_loss = 0
        total_mtp_loss = 0
        total_sip_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            data = {key: value.to(self.device) for key, value in data.items()}
            self.model.zero_grad()
            if self.sbert_cfgs.scene:
                outputs = self.model(spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                     segment_ids=data["segment_ids"], attn_mask=data["attn_mask"],
                                     env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                     env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"],
                                     envs=data["envs"], envs_params=data["envs_params"],
                                     traj_mask=data["traj_mask"], traj_lbl=data["traj_lbl"], near_lbl=data["near_lbl"])
            else:
                outputs = self.model(spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                     segment_ids=data["segment_ids"], attn_mask=data["attn_mask"],
                                     traj_mask=data["traj_mask"], traj_lbl=data["traj_lbl"], near_lbl=data["near_lbl"])

            loss = outputs['total_loss'].mean()
            total_mtp_loss += outputs['mtp_loss'].mean().item()
            if self.sbert_cfgs.sip:
                total_sip_loss += outputs['sip_loss'].mean().item()
            total_loss += outputs['total_loss'].mean().item()
            loss.backward()
            if self.args.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.lr_scheduler.step()
        total_loss = total_loss / len(data_iter)
        total_mtp_loss = total_mtp_loss / len(data_iter)
        total_sip_loss = total_sip_loss / len(data_iter)
        print("total=%f, mtp=%f, sip=%f" % (total_loss, total_mtp_loss, total_sip_loss))
        params = {"lr": self.optim.param_groups[0]['lr']}
        return total_loss, params


    def val_iteration(self, epoch, data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % ("val", epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        total_mtp_loss = 0
        total_sip_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
            data = {key: value.to(self.device) for key, value in data.items()}
            if self.sbert_cfgs.scene:
                outputs = self.model(spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                     segment_ids=data["segment_ids"], attn_mask=data["attn_mask"],
                                     env_spatial_ids=data["env_spatial_ids"], env_temporal_ids=data["env_temporal_ids"],
                                     env_segment_ids=data["env_segment_ids"], env_attn_mask=data["env_attn_mask"],
                                     traj_mask=data["traj_mask"], traj_lbl=data["traj_lbl"], near_lbl=data["near_lbl"])
            else:
                outputs = self.model(spatial_ids=data["spatial_ids"], temporal_ids=data["temporal_ids"],
                                     segment_ids=data["segment_ids"], attn_mask=data["attn_mask"],
                                     traj_mask=data["traj_mask"], traj_lbl=data["traj_lbl"], near_lbl=data["near_lbl"])
            total_mtp_loss += outputs['mtp_loss'].mean().item()
            if self.sbert_cfgs.sip:
                total_sip_loss += outputs['sip_loss'].mean().item()
            total_loss += outputs['total_loss'].mean().item()
        total_loss = total_loss / len(data_iter)
        total_mtp_loss = total_mtp_loss / len(data_iter)
        total_sip_loss = total_sip_loss / len(data_iter)
        print("total=%f, mtp=%f, sip=%f" % (total_loss, total_mtp_loss, total_sip_loss))
        params = {"lr": self.optim.param_groups[0]['lr']}
        return total_loss, params