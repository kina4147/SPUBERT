import torch
import copy
import tqdm
import transformers
from SPUBERT.model.spubert import (
    SPUBERTMGPConfig,
    SPUBERTTGPConfig,
    SPUBERTConfig,
    SPUBERTModel,
)
from SPUBERT.dataset.grid_map_numpy import estimate_map_length, estimate_num_patch
from SPUBERT.model.loss import bom_loss
transformers.logging.set_verbosity_info()
torch.autograd.set_detect_anomaly(True)

class SPUBERTTrainer(object):
    def __init__(self, train_dataloader=None, args=None):
        self.train_dataloader = train_dataloader
        self.optim = None
        self.lr_scheduler = None
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        num_patch = estimate_num_patch(estimate_map_length(args.env_range*2, args.env_resol), args.patch_size)
        self.sbert_tgp_cfgs = SPUBERTTGPConfig(
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch,
            patch_size=args.patch_size, view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

        self.sbert_mgp_cfgs = SPUBERTMGPConfig(
            hidden_size=args.hidden, num_layer=args.layer, num_head=args.head, k_sample=args.k_sample,
            goal_hidden_size=args.goal_hidden, goal_latent_size=args.goal_latent, obs_len=args.obs_len,
            pred_len=args.pred_len, num_nbr=args.num_nbr, scene=args.scene, num_patch=num_patch, patch_size=args.patch_size,
            view_range=args.view_range, view_angle=args.view_angle, social_range=args.social_range)

        self.sbert_cfgs = SPUBERTConfig(traj_cfgs=self.sbert_tgp_cfgs, goal_cfgs=self.sbert_mgp_cfgs)
        self.model = SPUBERTModel(self.sbert_tgp_cfgs, self.sbert_mgp_cfgs, self.sbert_cfgs)
        self.model.to(self.device)

        self.mgp_optim = transformers.AdamW(self.model.mgp_model.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
        self.tgp_optim = transformers.AdamW(self.model.tgp_model.parameters(), lr=args.lr, eps=1e-6, betas=(0.9, 0.999), weight_decay=0.01)

        self.mgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.mgp_optim,
                                                       num_warmup_steps=args.warm_up,
                                                       num_training_steps=args.epoch * len(self.train_dataloader))
        self.tgp_lr_scheduler = transformers.get_scheduler("linear", optimizer=self.tgp_optim,
                                                       num_warmup_steps=args.warm_up,
                                                       num_training_steps=args.epoch * len(self.train_dataloader))

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
                gderror, aderror, fderror = bom_loss(outputs["pred_goals"], outputs["pred_trajs"], data["goal_lbl"], data["traj_lbl"], k_sample)
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
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % ("train", epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")


        total_mgp_loss = 0
        total_tgp_loss = 0
        total_kld_loss = 0
        total_gde_loss = 0
        total_ade_loss = 0
        total_fde_loss = 0
        for i, it_data in data_iter:
            data = copy.deepcopy(it_data)
            del it_data
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
                                     envs_params=data["envs_params"])
            else:
                outputs = self.model(mgp_spatial_ids=data["mgp_spatial_ids"],
                                     mgp_temporal_ids=data["mgp_temporal_ids"],
                                     mgp_segment_ids=data["mgp_segment_ids"], mgp_attn_mask=data["mgp_attn_mask"],
                                     tgp_spatial_ids=data["tgp_spatial_ids"], tgp_temporal_ids=data["tgp_temporal_ids"],
                                     tgp_segment_ids=data["tgp_segment_ids"], tgp_attn_mask=data["tgp_attn_mask"],
                                     traj_lbl=data["traj_lbl"], goal_lbl=data["goal_lbl"])

            mgp_loss = outputs["mgp_loss"].mean()
            tgp_loss = outputs["tgp_loss"].mean()
            total_kld_loss += outputs["kld_loss"].mean().item()
            total_ade_loss += outputs["ade_loss"].mean().item()
            total_fde_loss += outputs["fde_loss"].mean().item()
            total_gde_loss += outputs["gde_loss"].mean().item()
            total_mgp_loss += outputs["mgp_loss"].mean().item()
            total_tgp_loss += outputs["tgp_loss"].mean().item()
            mgp_loss.backward()
            tgp_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.mgp_optim.step()
            self.tgp_optim.step()
            self.mgp_lr_scheduler.step()
            self.tgp_lr_scheduler.step()
        total_kld_loss = total_kld_loss / len(data_iter)
        total_ade_loss = total_ade_loss / len(data_iter)
        total_fde_loss = total_fde_loss / len(data_iter)
        total_gde_loss = total_gde_loss / len(data_iter)
        total_mgp_loss = total_mgp_loss / len(data_iter)
        total_tgp_loss = total_tgp_loss / len(data_iter)
        print("[MGP] total_mgp=%f, kld=%f, gde=%f" % (total_mgp_loss, total_kld_loss, total_gde_loss))
        print("[TGP] total_tgp=%f, ade=%f, fde=%f" % (total_tgp_loss, total_ade_loss, total_fde_loss))

    def step_lr_scheduler(self, loss):
        if self.args.lr_scheduler == 'loss':
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def train(self, epoch):
        self.model.train()
        return self.train_iteration(epoch, self.train_dataloader)

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model.from_pretrained(path)
