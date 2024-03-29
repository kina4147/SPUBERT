# coding=utf-8
# Copyright (c) 2023 Electronics and Telecommunications Research Institute (ETRI).
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from SPUBERT.model.modeling_spubert import SPUBERTEncoder, SPUBERTModelBase
from typing import Optional, Tuple
from collections import OrderedDict
from SPUBERT.model.embedding import SPUBERTEmbeddings, SPUBERTMMEmbeddings
from SPUBERT.model.cvae import GoalRecognitionNet, GoalEncoder#, GoalPriorNet
from SPUBERT.model.decoder import SpatialDecoder, GoalDecoder
from SPUBERT.model.pooler import GoalPooler, FutureTrajPooler
from SPUBERT.model.loss import (
    ADELoss, FDELoss, MGPCVAELoss,
)
from SPUBERT.model.kmean_sampler import MultiKMeans

class SPUBERTTGPConfig:
    def __init__(
            self,
            hidden_size=512,
            num_layer=4,
            num_head=4,
            act_fn="relu",
            dropout_prob=0.1,
            input_dim=2,
            output_dim=2,
            goal_dim=2,
            view_range=20.0,
            view_angle=1.0,
            social_range=2.0,
            obs_len=8,
            pred_len=12,
            num_nbr=5,
            scene=False,
            num_patch=32,
            patch_size=32,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.goal_dim = goal_dim
        self.act_fn = act_fn
        self.dropout_prob=dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.num_layer = num_layer
        self.num_head = num_head
        self.view_range = view_range
        self.view_angle = view_angle
        self.social_range = social_range
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nbr = num_nbr
        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range

class SPUBERTMGPConfig:
    def __init__(
            self,
            hidden_size=512,
            num_layer=4,
            num_head=4,
            act_fn="relu",
            dropout_prob=0.1,
            input_dim=2,
            output_dim=2,
            goal_dim=2,
            view_range=20.0,
            view_angle=1.0,
            social_range=2.0,
            obs_len=8,
            pred_len=12,
            num_nbr=5,
            scene=False,
            num_patch=32,
            patch_size=32,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            k_sample=20,
            goal_hidden_size=64,
            goal_latent_size=64,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.goal_dim = goal_dim
        self.act_fn = act_fn
        self.dropout_prob=dropout_prob
        self.layer_norm_eps = layer_norm_eps

        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size*4
        self.num_layer = num_layer
        self.num_head = num_head

        self.view_range = view_range
        self.view_angle = view_angle
        self.social_range = social_range
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nbr = num_nbr

        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range

        self.k_sample = k_sample
        self.goal_hidden_size = goal_hidden_size
        self.goal_latent_size = goal_latent_size



class SPUBERTConfig:
    def __init__(self, traj_cfgs, goal_cfgs):
        # Matching Parameters
        if traj_cfgs.input_dim == goal_cfgs.input_dim:
            self.input_dim = goal_cfgs.input_dim
        else:
            raise ValueError("input_dim")
        if traj_cfgs.view_range == goal_cfgs.view_range:
            self.view_range = goal_cfgs.view_range
        else:
            raise ValueError("view_range")
        if traj_cfgs.view_angle == goal_cfgs.view_angle:
            self.view_angle = goal_cfgs.view_angle
        else:
            raise ValueError("input_dim")
        if traj_cfgs.social_range == goal_cfgs.social_range:
            self.social_range = goal_cfgs.social_range
        else:
            raise ValueError("input_dim")
        if traj_cfgs.obs_len == goal_cfgs.obs_len:
            self.obs_len = goal_cfgs.obs_len
        else:
            raise ValueError("input_dim")
        if traj_cfgs.pred_len == goal_cfgs.pred_len:
            self.pred_len = goal_cfgs.pred_len
        else:
            raise ValueError("input_dim")
        if traj_cfgs.num_nbr == goal_cfgs.num_nbr:
            self.num_nbr = goal_cfgs.num_nbr
        else:
            raise ValueError("input_dim")
        if traj_cfgs.scene == goal_cfgs.scene:
            self.scene = goal_cfgs.scene
        else:
            raise ValueError("input_dim")
        if traj_cfgs.num_patch == goal_cfgs.num_patch:
            self.num_patch = goal_cfgs.num_patch
        else:
            raise ValueError("input_dim")
        if traj_cfgs.patch_size == goal_cfgs.patch_size:
            self.patch_size = goal_cfgs.patch_size
        else:
            raise ValueError("input_dim")
        if traj_cfgs.pad_token_id == goal_cfgs.pad_token_id:
            self.pad_token_id = goal_cfgs.pad_token_id
        else:
            raise ValueError
        if traj_cfgs.chunk_size_feed_forward == goal_cfgs.chunk_size_feed_forward:
            self.chunk_size_feed_forward = goal_cfgs.chunk_size_feed_forward
        else:
            raise ValueError
        if traj_cfgs.initializer_range == goal_cfgs.initializer_range:
            self.initializer_range = goal_cfgs.initializer_range
        else:
            raise ValueError("initializer_range")
        if traj_cfgs.hidden_size == goal_cfgs.hidden_size:
            self.hidden_size = goal_cfgs.hidden_size
            self.intermediate_size = goal_cfgs.intermediate_size
        else:
            raise ValueError("hidden_size")
        if traj_cfgs.initializer_range == goal_cfgs.initializer_range:
            self.initializer_range = goal_cfgs.initializer_range
        else:
            raise ValueError("initializer_range")
        if traj_cfgs.act_fn == goal_cfgs.act_fn:
            self.act_fn = goal_cfgs.act_fn
        else:
            raise ValueError("act_fn")
        if traj_cfgs.dropout_prob == goal_cfgs.dropout_prob:
            self.dropout_prob = goal_cfgs.dropout_prob
        else:
            raise ValueError("dropout_prob")
        if traj_cfgs.layer_norm_eps == goal_cfgs.layer_norm_eps:
            self.layer_norm_eps = goal_cfgs.layer_norm_eps
        else:
            raise ValueError("layer_norm_eps")

        if traj_cfgs.output_dim == goal_cfgs.output_dim:
            self.output_dim = goal_cfgs.output_dim
        else:
            raise ValueError("output_dim")
        if traj_cfgs.goal_dim == goal_cfgs.goal_dim:
            self.goal_dim = goal_cfgs.goal_dim
        else:
            raise ValueError("goal_dim")

        self.k_sample = goal_cfgs.k_sample
        self.goal_hidden_size = goal_cfgs.goal_hidden_size
        self.goal_latent_size = goal_cfgs.goal_latent_size
        self.num_traj_layer = traj_cfgs.num_layer
        self.num_traj_head = traj_cfgs.num_head
        self.num_goal_layer = goal_cfgs.num_layer
        self.num_goal_head = goal_cfgs.num_head


class SPUBERTEncoderModel(SPUBERTModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        if self.cfgs.scene:
            self.embeddings = SPUBERTMMEmbeddings(cfgs)
        else:
            self.embeddings = SPUBERTEmbeddings(cfgs)
        self.sbert_encoder = SPUBERTEncoder(cfgs)
        self.init_weights()

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            output_attentions=False
    ):
        device = spatial_ids.device
        traj_batch_size, traj_seq_len, traj_spatial_dim = spatial_ids.size()
        if self.cfgs.scene:
            env_batch_size, env_seq_len, env_spatial_dim = env_spatial_ids.size()
            emb_h = self.embeddings(spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids,
                                    env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids,
                                    env_segment_ids=env_segment_ids)
            all_attn_mask = torch.cat([attn_mask, env_attn_mask], dim=1)
            ext_all_attn_mask: torch.Tensor = self.get_extended_attention_mask(all_attn_mask, (env_batch_size, traj_seq_len + env_seq_len), device)
        else:
            ext_all_attn_mask: torch.Tensor = self.get_extended_attention_mask(attn_mask, (traj_batch_size, traj_seq_len), device)
            emb_h = self.embeddings(spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids)
        enc_h = self.sbert_encoder(hidden_states=emb_h, attention_mask=ext_all_attn_mask, output_attentions=output_attentions)

        return enc_h

class SPUBERTTGPOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    ade_loss: torch.FloatTensor = None
    fde_loss: torch.FloatTensor = None
    pred_trajs: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTTGPModel(SPUBERTModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SPUBERTEncoderModel(cfgs=cfgs)
        self.traj_pooler = FutureTrajPooler(hidden_size=cfgs.hidden_size, obs_len=cfgs.obs_len, pred_len=cfgs.pred_len)
        self.sbert_decoder = SpatialDecoder(cfgs.hidden_size, cfgs.output_dim, layer_norm_eps=cfgs.layer_norm_eps, act_fn=cfgs.act_fn)
        self.init_weights()


    def inference(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            output_attentions=False):
        enc_h = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions
        )
        traj_h = self.traj_pooler(enc_h["last_hidden_state"])
        pred_trajs = self.sbert_decoder(traj_h)  # (B*K, Seq, Sdim)
        return SPUBERTTGPOutput(
            pred_trajs=pred_trajs,
            attentions=enc_h["attentions"],
        )

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            traj_lbl=None,
            goal_lbl=None,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            envs=None,
            envs_params=None,
            output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions
        )
        traj_h = self.traj_pooler(enc_h["last_hidden_state"])
        pred_trajs = self.sbert_decoder(traj_h)  # (B*K, Seq, Sdim)
        loss_ade = ADELoss()
        loss_fde = FDELoss()
        ade_loss = loss_ade(pred_trajs, traj_lbl)
        fde_loss = loss_fde(pred_trajs[:, -1], goal_lbl)
        ade_loss = ade_loss
        fde_loss = fde_loss
        total_loss = ade_loss


        return SPUBERTTGPOutput(
            total_loss=total_loss,
            ade_loss=ade_loss,
            fde_loss=fde_loss,
            pred_trajs=pred_trajs,
            attentions=enc_h["attentions"],
        )


class SPUBERTMGPOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    gde_loss: torch.FloatTensor = None
    kld_loss: torch.FloatTensor = None
    pred_goals: torch.FloatTensor = None
    best_idx = torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTMGPModel(SPUBERTModelBase):
    def __init__(self, goal_cfgs):
        super().__init__(goal_cfgs)
        self.sbert = SPUBERTEncoderModel(goal_cfgs)

        # POOLER
        self.enc_hidden_pooler = GoalPooler(hidden_size=goal_cfgs.hidden_size, obs_len=goal_cfgs.obs_len, pred_len=goal_cfgs.pred_len, act_fn=goal_cfgs.act_fn)

        # GT GOAL ENCODER
        self.gt_goal_encoder = GoalEncoder(goal_cfgs.goal_dim, goal_cfgs.goal_hidden_size, layer_norm_eps=goal_cfgs.layer_norm_eps,
                                           dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)

        # CVAE NETWORK
        self.goal_recog_net = GoalRecognitionNet(
            enc_hidden_size=goal_cfgs.hidden_size, goal_hidden_size=goal_cfgs.goal_hidden_size,
            goal_latent_size=goal_cfgs.goal_latent_size, act_fn=goal_cfgs.act_fn)

        self.sbert_decoder = GoalDecoder(goal_cfgs.hidden_size, goal_cfgs.goal_latent_size, goal_cfgs.goal_dim, act_fn=goal_cfgs.act_fn)
        self.init_weights()

    def goal_predict(self,
                     spatial_ids=None,
                     temporal_ids=None,
                     segment_ids=None,
                     attn_mask=None,
                     env_spatial_ids=None,
                     env_temporal_ids=None,
                     env_segment_ids=None,
                     env_attn_mask=None,
                     num_goal=10000,
                     ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask
        )

        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        pred_goals = self.goal_predictor(goal_h, k_sample=num_goal)
        return pred_goals

    def goal_predictor(self, pred_goal_h, k_sample, d_sample=0):
        if d_sample < k_sample:
            d_sample = k_sample

        p_goal_mu = torch.zeros(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).to(pred_goal_h.device)
        p_goal_std = torch.ones(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).to(pred_goal_h.device)
        k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, d_sample, 1)  # B * goal_hidden_size
        eps = torch.randn_like(p_goal_std)
        k_goal_latent = eps.mul(p_goal_std).add(p_goal_mu)
        k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
        bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)

        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)  # BK * S
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), d_sample, self.cfgs.goal_dim)  # BK * S => B*K*S

        if d_sample > k_sample:
            kmeans = MultiKMeans(n_clusters=k_sample, n_kmeans=pred_goals.shape[0], max_iter=10, verbose=False)
            pred_goals = kmeans.fit_predict(pred_goals)

        return pred_goals


    def goal_trainer(self, pred_goal_h, cur_pos, traj_lbl, goal_lbl, k_sample):
        gt_goal_h = self.gt_goal_encoder(goal_lbl)

        r_goal_out = self.goal_recog_net(torch.cat([pred_goal_h, gt_goal_h], dim=-1))
        r_goal_mu = r_goal_out[:, :self.cfgs.goal_latent_size]
        r_goal_logvar = r_goal_out[:, self.cfgs.goal_latent_size:]
        # KLD Loss
        alpha = 1.0
        kld_loss = alpha * (-0.5 * torch.sum(1 + r_goal_logvar - r_goal_logvar.exp() - r_goal_mu.pow(2), dim=1).mean())

        # B * K * Lat
        r_goal_std = r_goal_logvar.mul(0.5).exp()
        k_r_goal_mu = r_goal_mu.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
        k_r_goal_std = r_goal_std.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size

        k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
        eps = torch.randn_like(k_r_goal_std)
        k_goal_latent = eps.mul(k_r_goal_std).add(k_r_goal_mu) # z
        k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
        bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)  # K*B*L => B*K*L => BK * L

        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)  # BK * S
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), k_sample, self.cfgs.goal_dim)

        return pred_goals, kld_loss


    def inference(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            output_attentions=False,
            d_sample=0
    ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions
        )
        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        pred_goals = self.goal_predictor(goal_h, k_sample=self.cfgs.k_sample, d_sample=d_sample)
        return SPUBERTMGPOutput(
            pred_goals=pred_goals,
            attentions=goal_enc_out["attentions"],
        )

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            traj_lbl=None,
            goal_lbl=None,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            envs=None,
            envs_params=None,
            output_attentions=False,
            # kld_weight=None,
    ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions
        )

        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        pred_goals, kld_loss = self.goal_trainer(goal_h, spatial_ids[:, self.cfgs.obs_len+self.cfgs.pred_len, :], traj_lbl, goal_lbl, k_sample=self.cfgs.k_sample)

        mgp_cvae_loss = MGPCVAELoss()
        gde_loss, best_idx = mgp_cvae_loss(pred_goals, goal_lbl, self.cfgs.k_sample, output_dim=self.cfgs.output_dim, best=True)
        kld_loss = kld_loss
        gde_loss = gde_loss
        total_loss = kld_loss + gde_loss


        return SPUBERTMGPOutput(
            total_loss=total_loss,
            gde_loss=gde_loss,
            kld_loss=kld_loss,
            pred_goals=pred_goals,
            best_idx=best_idx,
            attentions=goal_enc_out["attentions"],
        )

class SPUBERTOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    gde_loss: torch.FloatTensor = None
    ade_loss: torch.FloatTensor = None
    fde_loss: torch.FloatTensor = None
    kld_loss: torch.FloatTensor = None
    pred_trajs: torch.FloatTensor = None
    pred_goals: torch.FloatTensor = None
    mgp_attentions: Optional[Tuple[torch.FloatTensor]] = None
    tgp_attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTModel(SPUBERTModelBase):

    def __init__(self, tgp_cfgs, mgp_cfgs, cfgs):
        super().__init__(cfgs)
        self.tgp_model = SPUBERTTGPModel(tgp_cfgs)
        self.mgp_model = SPUBERTMGPModel(mgp_cfgs)
        self.init_weights()

    def add_goals(self, spatial_ids, goals, mask_val, pad_val):
        """
        # spatial ids: b * seq * spatial_dim
        # => spatial_ids: b * k * seq * spatial_dim (repeat)
        # goals: b * k * spatial_dim
        # goals: b * k * 1 * spatial_dim
        # add_goals : k * b * seq * spatial_dim
        """
        spatial_ids = spatial_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1, 1)
        spatial_ids[:, :, 1 + self.cfgs.obs_len:self.cfgs.obs_len + self.cfgs.pred_len, :] = mask_val
        spatial_ids[:, :, self.cfgs.obs_len + self.cfgs.pred_len, :self.cfgs.goal_dim] = goals
        spatial_ids[:, :, self.cfgs.obs_len + self.cfgs.pred_len, self.cfgs.goal_dim:] = pad_val
        return spatial_ids

    def add_best_goals(self, spatial_ids, pred_goals, best_idx, mask_val, pad_val):
        """
        # spatial ids: b * seq * spatial_dim
        # => spatial_ids: b * 1 * seq * spatial_dim (repeat)
        # goals: b * 1 * spatial_dim
        # goals: b * 1 * 1 * spatial_dim
        # add_goals : k * b * seq * spatial_dim
        """
        best_goals = pred_goals[range(len(best_idx)), best_idx]
        spatial_ids[:, 1 + self.cfgs.obs_len:self.cfgs.obs_len + self.cfgs.pred_len, :] = mask_val
        spatial_ids[:, self.cfgs.obs_len + self.cfgs.pred_len, :self.cfgs.goal_dim] = best_goals
        spatial_ids[:, self.cfgs.obs_len + self.cfgs.pred_len, self.cfgs.goal_dim:] = pad_val
        return spatial_ids

    def inference(
            self,
            mgp_spatial_ids,
            mgp_temporal_ids,
            mgp_segment_ids,
            mgp_attn_mask,
            tgp_temporal_ids,
            tgp_segment_ids,
            tgp_attn_mask,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            output_attentions=False,
            d_sample=0,
    ):

        traj_batch_size, traj_seq_len, traj_spatial_dim = mgp_spatial_ids.size()
        mgp_out = self.mgp_model.inference(
            spatial_ids=mgp_spatial_ids, segment_ids=mgp_segment_ids, temporal_ids=mgp_temporal_ids, attn_mask=mgp_attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions, d_sample=d_sample)

        k_goal_spatial_ids = self.add_goals(mgp_spatial_ids, mgp_out["pred_goals"], mask_val=self.cfgs.view_range, pad_val=-self.cfgs.view_range).view(-1, traj_seq_len, self.cfgs.input_dim)
        k_goal_segment_ids = tgp_segment_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)  # B * K * Sdim -> BK * Seq
        k_goal_temporal_ids = tgp_temporal_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)  # B * K * Sdim -> BK * Seq
        k_goal_attn_mask = tgp_attn_mask.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, traj_seq_len)  # B * K * Sdim -> BK * Seq
        if self.cfgs.scene:
            env_batch_size, env_seq_len, env_spatial_dim = env_spatial_ids.size()
            k_goal_env_spatial_ids = env_spatial_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1, 1).view(-1, env_seq_len, env_spatial_dim)
            k_goal_env_segment_ids = env_segment_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
            k_goal_env_temporal_ids = env_temporal_ids.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
            k_goal_env_attn_mask = env_attn_mask.unsqueeze(1).repeat(1, self.cfgs.k_sample, 1).view(-1, env_seq_len)
        else:
            k_goal_env_spatial_ids = None
            k_goal_env_segment_ids = None
            k_goal_env_temporal_ids = None
            k_goal_env_attn_mask = None

        tgp_out = self.tgp_model.inference(
            spatial_ids=k_goal_spatial_ids, segment_ids=k_goal_segment_ids, temporal_ids=k_goal_temporal_ids,
            attn_mask=k_goal_attn_mask, env_spatial_ids=k_goal_env_spatial_ids, env_segment_ids=k_goal_env_segment_ids,
            env_temporal_ids=k_goal_env_temporal_ids, env_attn_mask=k_goal_env_attn_mask, output_attentions=output_attentions)
        pred_trajs = tgp_out["pred_trajs"].reshape(traj_batch_size, self.cfgs.k_sample, self.cfgs.pred_len, self.cfgs.output_dim)
        pred_goals = mgp_out["pred_goals"].reshape(traj_batch_size, self.cfgs.k_sample, self.cfgs.goal_dim)  # BK * S => B*K*S
        return SPUBERTOutput(
            pred_trajs=pred_trajs,
            pred_goals=pred_goals,
            goal_attentions=mgp_out["attentions"],
            traj_attentions=tgp_out["attentions"])

    def forward(
            self,
            mgp_spatial_ids,
            mgp_temporal_ids,
            mgp_segment_ids,
            mgp_attn_mask,
            tgp_spatial_ids,
            tgp_temporal_ids,
            tgp_segment_ids,
            tgp_attn_mask,
            traj_lbl=None,
            goal_lbl=None,
            env_spatial_ids=None,
            env_temporal_ids=None,
            env_segment_ids=None,
            env_attn_mask=None,
            envs=None,
            envs_params=None,
            output_attentions=False,
    ):
        mgp_out = self.mgp_model(
            spatial_ids=mgp_spatial_ids, segment_ids=mgp_segment_ids, temporal_ids=mgp_temporal_ids, attn_mask=mgp_attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, envs=envs, envs_params=envs_params,
            traj_lbl=traj_lbl, goal_lbl=goal_lbl, output_attentions=output_attentions)

        tgp_out = self.tgp_model(
            spatial_ids=tgp_spatial_ids, segment_ids=tgp_segment_ids, temporal_ids=tgp_temporal_ids, attn_mask=tgp_attn_mask,
            env_spatial_ids=env_spatial_ids, env_segment_ids=env_segment_ids, env_temporal_ids=env_temporal_ids,
            env_attn_mask=env_attn_mask, envs=envs, envs_params=envs_params,
            traj_lbl=traj_lbl, goal_lbl=goal_lbl, output_attentions=output_attentions)


        return SPUBERTOutput(
            mgp_loss=mgp_out["total_loss"],
            tgp_loss=tgp_out["total_loss"],
            ade_loss=tgp_out["ade_loss"],
            fde_loss=tgp_out["fde_loss"],
            gde_loss=mgp_out["gde_loss"],
            kld_loss=mgp_out["kld_loss"],
            pred_trajs=tgp_out["pred_trajs"],
            pred_goals=mgp_out["pred_goals"],
            goal_attentions=mgp_out["attentions"],
            traj_attentions=tgp_out["attentions"])

