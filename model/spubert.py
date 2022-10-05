
import torch
import torch.nn as nn
from SPUBERT.model.modeling_spubert import (
    SPUBERTEncoder,
    SPUBERTModelBase,
)

from typing import Optional, Tuple
from collections import OrderedDict
from SPUBERT.model.embedding import SPUBERTEmbeddings, SPUBERTMMEmbeddings
from SPUBERT.model.cvae import (
    GoalRecognitionNet, GoalPriorNet, GoalEncoder, TrajGRUEncoder, TrajMLPEncoder, TrajHiddenGRUEncoder)
from SPUBERT.model.decoder import SpatialDecoder, GoalDecoder, SIPDecoder, MTPDecoder
from SPUBERT.model.pooler import GoalPooler, FullTrajPooler, PastTrajPooler, FutureTrajPooler, MTPPooler, SIPPooler
from SPUBERT.model.loss import (
    goal_collision_loss, pos_collision_loss, beta_tcvae_loss, info_vae_loss, beta_tcvae_loss_normal,
    MaskedADELoss, ADELoss, FDELoss, MGPCVAELoss, reparametrize, BetaTCVAELoss
    # tgp_cvae_loss, cvae_loss, mgp_cvae_loss, traj_loss, attn_traj_loss, traj_collision_loss,mgp_tgp_fde_loss,
)
from SPUBERT.model.kmean_sampler import MultiKMeans
class SPUBERTPTConfig:
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
            col_weight=10,
            traj_weight=1.0,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            sip=False
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

        self.col_weight = col_weight
        self.traj_weight = traj_weight

        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range

        self.sip = sip

class SPUBERTTGPConfig:
    def __init__(
            self,
            hidden_size=512,
            # intermediate_size=3072, # FFNN
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
            col_weight=10,
            traj_weight=1.0,
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

        self.col_weight = col_weight
        self.traj_weight = traj_weight

        self.scene = scene
        self.num_patch = num_patch
        self.patch_size = patch_size

        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range

class SPUBERTMGPConfig:
    # model_type = "sbertplus_mgp"
    def __init__(
            self,
            hidden_size=512,
            num_layer=4,
            num_head=4,
            # intermediate_size=3072, # FFNN
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
            kld_weight=10,
            col_weight=10,
            goal_weight=1.0,
            cvae_sigma=1.0,
            kld_clamp=None,
            share=False,
            normal=False,
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
        self.kld_weight = kld_weight
        self.col_weight = col_weight
        self.goal_weight = goal_weight
        self.share = share

        self.cvae_sigma = cvae_sigma
        self.kld_clamp = kld_clamp
        self.normal = normal


class SPUBERTFTConfig:
    def __init__(self, traj_cfgs, goal_cfgs, share=False):
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

        if traj_cfgs.col_weight == goal_cfgs.col_weight:
            self.col_weight = goal_cfgs.col_weight
        else:
            raise ValueError("col_weight")

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
        self.kld_weight = goal_cfgs.kld_weight
        self.cvae_sigma = goal_cfgs.cvae_sigma
        self.kld_clamp = goal_cfgs.kld_clamp
        self.traj_weight = traj_cfgs.traj_weight
        self.goal_weight = goal_cfgs.goal_weight
        self.num_traj_layer = traj_cfgs.num_layer
        self.num_traj_head = traj_cfgs.num_head
        self.num_goal_layer = goal_cfgs.num_layer
        self.num_goal_head = goal_cfgs.num_head
        self.normal = goal_cfgs.normal

        # self.pretrained_models = ["RMTP", "RMTP_SHARE", "TGP", "MGP", "TGP_MGP", "FROM_SCRATCH"]
        # self.train_mode = train_mode
        self.share = share
        if share:
            if self.num_traj_head != self.num_goal_layer or self.num_goal_head != self.num_traj_head:
                raise ValueError("TGP and MGP Encoders don't have the same size.")
            else:
                self.num_layer = self.num_traj_layer
                self.num_head = self.num_traj_head

class SPUBERTModel(SPUBERTModelBase):
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



class SPUBERTPTOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    # col_loss: torch.FloatTensor = None
    mtp_loss: torch.FloatTensor = None
    sip_loss: torch.FloatTensor = None
    mtp_output: torch.FloatTensor = None
    sip_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class SPUBERTPTModel(SPUBERTModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SPUBERTModel(cfgs=cfgs)
        self.sip_pooler = SIPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len, num_nbr=cfgs.num_nbr)
        self.sip_decoder = SIPDecoder(hidden_size=cfgs.hidden_size)
        self.mtp_pooler = MTPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len, num_nbr=cfgs.num_nbr)
        self.mtp_decoder = MTPDecoder(hidden_size=cfgs.hidden_size, out_dim=cfgs.output_dim, layer_norm_eps=cfgs.layer_norm_eps, act_fn=cfgs.act_fn)
        self.init_weights()

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            traj_lbl=None,
            goal_lbl=None,
            traj_mask=None,
            near_lbl=None,
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
        mtp_h = self.mtp_pooler(enc_h["last_hidden_state"])
        sip_h = self.sip_pooler(enc_h["last_hidden_state"])
        mtp_out = self.mtp_decoder(mtp_h)
        sip_out = self.sip_decoder(sip_h)

        loss_mtp = MaskedADELoss()
        mtp_loss = loss_mtp(mtp_out, traj_lbl, traj_mask)
        total_loss = mtp_loss
        if self.cfgs.sip:
            loss_sip = nn.NLLLoss(ignore_index=0, reduction='mean')
            sip_loss = loss_sip(sip_out.view(-1, 3), near_lbl.view(-1))
            total_loss += sip_loss
        else:
            sip_loss = None

        # if self.cfgs.scene and self.cfgs.col_weight > 0:
        #     col_loss = self.cfgs.col_weight*pos_collision_loss(pred_trajs, envs, envs_params)
        #     total_loss += col_loss
        # else:
        #     col_loss = None

        return SPUBERTPTOutput(
            total_loss=total_loss,
            mtp_loss=mtp_loss,
            # col_loss=col_loss,
            sip_loss=sip_loss,
            mtp_output=mtp_out,
            sip_output=sip_out,
            attentions=enc_h["attentions"],
        )


class SPUBERTTGPOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    ade_loss: torch.FloatTensor = None
    fde_loss: torch.FloatTensor = None
    col_loss: torch.FloatTensor = None
    pred_trajs: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTTGPModel(SPUBERTModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SPUBERTModel(cfgs=cfgs)
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
        ade_loss = self.cfgs.traj_weight * ade_loss
        fde_loss = self.cfgs.traj_weight * fde_loss
        total_loss = ade_loss # + fde_loss
        if self.cfgs.scene and self.cfgs.col_weight > 0:
            col_loss = self.cfgs.col_weight*pos_collision_loss(pred_trajs, envs, envs_params)
            total_loss += col_loss
        else:
            col_loss = None

        return SPUBERTTGPOutput(
            total_loss=total_loss,
            ade_loss=ade_loss,
            fde_loss=fde_loss,
            col_loss=col_loss,
            pred_trajs=pred_trajs,
            attentions=enc_h["attentions"],
        )


class SPUBERTMGPOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    col_loss: torch.FloatTensor = None
    gde_loss: torch.FloatTensor = None
    kld_loss: torch.FloatTensor = None
    pred_goals: torch.FloatTensor = None
    best_idx = torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTMGPModel(SPUBERTModelBase):
    def __init__(self, goal_cfgs, share_enc=None):
        super().__init__(goal_cfgs)
        if share_enc:
            self.sbert = share_enc
        else:
            self.sbert = SPUBERTModel(goal_cfgs)

        # POOLER
        self.enc_hidden_pooler = GoalPooler(hidden_size=goal_cfgs.hidden_size, obs_len=goal_cfgs.obs_len, pred_len=goal_cfgs.pred_len, act_fn=goal_cfgs.act_fn)
        # self.enc_hidden_pooler = FutureTrajPooler(hidden_size=goal_cfgs.hidden_size, obs_len=goal_cfgs.obs_len, pred_len=goal_cfgs.pred_len)


        # MGP HIDDEN ENCODER
        # self.obs_traj_h_encoder = TrajGRUEncoder(input_dim=goal_cfgs.hidden_size, hidden_size=goal_cfgs.hidden_size,
        #         layer_norm_eps=goal_cfgs.layer_norm_eps, dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)
        # self.enc_hidden_gru_encoder = TrajHiddenGRUEncoder(
        #     mgp_hidden_size=goal_cfgs.hidden_size, hidden_size=goal_cfgs.hidden_size, layer_norm_eps=goal_cfgs.layer_norm_eps,
        #     dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)
        # self.obs_traj_h_encoder = TrajMLPEncoder(input_dim=goal_cfgs.hidden_size, hidden_size=goal_cfgs.hidden_size, traj_len=goal_cfgs.obs_len,
        #         layer_norm_eps=goal_cfgs.layer_norm_eps, dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)

        # GT GOAL ENCODER
        self.gt_goal_encoder = GoalEncoder(goal_cfgs.goal_dim, goal_cfgs.goal_hidden_size, layer_norm_eps=goal_cfgs.layer_norm_eps,
                                           dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)
        # self.pred_traj_embeddings = SpatialEmbedding(input_dim=goal_cfgs.goal_dim, embedding_dim=goal_cfgs.goal_hidden_size, act_fn=goal_cfgs.act_fn)
        # self.gt_traj_encoder = TrajGRUEncoder(input_dim=goal_cfgs.goal_dim, hidden_size=goal_cfgs.goal_hidden_size,
        #         layer_norm_eps=goal_cfgs.layer_norm_eps, dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn, init=True)
        # self.gt_traj_encoder = TrajMLPEncoder(input_dim=goal_cfgs.goal_dim, hidden_size=goal_cfgs.goal_hidden_size, traj_len=goal_cfgs.pred_len,
        #         layer_norm_eps=goal_cfgs.layer_norm_eps, dropout_prob=goal_cfgs.dropout_prob, act_fn=goal_cfgs.act_fn)

        # CVAE NETWORK
        self.goal_recog_net = GoalRecognitionNet(
            enc_hidden_size=goal_cfgs.hidden_size, goal_hidden_size=goal_cfgs.goal_hidden_size,
            goal_latent_size=goal_cfgs.goal_latent_size, act_fn=goal_cfgs.act_fn)
        self.goal_prior_net = GoalPriorNet(
            enc_hidden_size=goal_cfgs.hidden_size, goal_latent_size=goal_cfgs.goal_latent_size, act_fn=goal_cfgs.act_fn)


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
        # if self.cfgs.scene:
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask
        )

        # traj_h = self.traj_pooler(goal_enc_out["last_hidden_state"])
        # goal_h = self.obs_traj_h_encoder(traj_h)
        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        # goal_h = self.enc_hidden_gru_encoder(goal_h)
        pred_goals = self.goal_predictor(goal_h, k_sample=num_goal)
        return pred_goals

    def goal_predictor(self, pred_goal_h, k_sample, d_sample=0):
        # normal = False
        # pred_goal_h
        # torch.tensor(len(pred_goal_h), self.cfgs.goal_latent_size)
        # k_goal_latent.normal_(0, 1)
        if d_sample < k_sample:
            d_sample = k_sample

        if self.cfgs.normal:
            p_goal_mu = torch.zeros(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).to(pred_goal_h.device)
            p_goal_std = torch.ones(pred_goal_h.size(0), d_sample, self.cfgs.goal_latent_size).mul(self.cfgs.cvae_sigma).to(pred_goal_h.device)
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, d_sample, 1)  # B * goal_hidden_size
            eps = torch.randn_like(p_goal_std)
            k_goal_latent = eps.mul(p_goal_std).add(p_goal_mu)
            k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
            bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)
        else:
            p_goal_out = self.goal_prior_net(pred_goal_h) # B X 2L
            p_goal_mu = p_goal_out[:, :self.cfgs.goal_latent_size]
            p_goal_logvar = p_goal_out[:, self.cfgs.goal_latent_size:]

            p_goal_std = p_goal_logvar.mul(0.5).exp()
            p_goal_mu = p_goal_mu.unsqueeze(1).repeat(1, d_sample, 1)  # B * goal_hidden_size
            p_goal_std = p_goal_std.unsqueeze(1).repeat(1, d_sample, 1)  # B * goal_hidden_size
            eps = torch.randn_like(p_goal_std)
            k_goal_latent = eps.mul(p_goal_std).add(p_goal_mu)  # B * K * H
            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, d_sample, 1)  # B * goal_hidden_size
            k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
            bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)  # K*B*L => B*K*L => BK * L

        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)  # BK * S
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), d_sample, self.cfgs.goal_dim)  # BK * S => B*K*S

        if d_sample > k_sample:
            kmeans = MultiKMeans(n_clusters=k_sample, n_kmeans=pred_goals.shape[0], max_iter=10, verbose=False)
            pred_goals = kmeans.fit_predict(pred_goals)

        return pred_goals


    def goal_trainer(self, pred_goal_h, cur_pos, traj_lbl, goal_lbl, k_sample):
        gt_goal_h = self.gt_goal_encoder(goal_lbl)

        # traj_lbl: B * T * INPUT_DIM => B * HIDDEN_DIM
        # traj_lbl = self.pred_traj_embeddings(traj_lbl)
        # gt_goal_h = self.pred_traj_encoder(traj_lbl)
        # if self.cfgs.scale:
        #     traj_lbl /= self.cfgs.scale
        # gt_goal_h = self.pred_traj_encoder(traj_lbl, cur_pos)
        # gt_goal_h = self.sbert.embeddings(goal_lbl.unsqueeze(1), torch.full((goal_lbl.size(0), 1), self.cfgs.obs_len+self.cfgs.pred_len, dtype=torch.long).to(goal_lbl.device), torch.ones((goal_lbl.size(0), 1), dtype=torch.long).to(goal_lbl.device))
        # gt_goal_h = gt_goal_h.squeeze(1)

        # Hidden state Dropout
        # gt_goal_h = nn.functional.dropout(gt_goal_h, p=0.25,  training=True)
        r_goal_out = self.goal_recog_net(torch.cat([pred_goal_h, gt_goal_h], dim=-1))
        r_goal_mu = r_goal_out[:, :self.cfgs.goal_latent_size]
        r_goal_logvar = r_goal_out[:, self.cfgs.goal_latent_size:]
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        # k_sample = 1000
        if self.cfgs.normal: # B * L
            #BetaTCVAE Loss
            # z_r = reparametrize(r_goal_mu, r_goal_logvar)
            # kld_loss = beta_tcvae_loss_normal(z_r, r_goal_mu, r_goal_logvar, alpha, beta, gamma)

            # KLD Loss
            kld_loss = alpha * (-0.5 * torch.sum(1 + r_goal_logvar - r_goal_logvar.exp() - r_goal_mu.pow(2), dim=1).mean())

            # INFOVAE
            # z_p = torch.randn_like(z_r)
            # kld_loss += beta * info_vae_loss(z_p, z_r)

            # B * K * Lat
            r_goal_std = r_goal_logvar.mul(0.5).exp()
            k_r_goal_mu = r_goal_mu.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
            k_r_goal_std = r_goal_std.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size

            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
            eps = torch.randn_like(k_r_goal_std)
            k_goal_latent = eps.mul(k_r_goal_std).add(k_r_goal_mu) # z
            k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
            bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)  # K*B*L => B*K*L => BK * L
        else:
            # r_dist = torch.distributions.Normal(r_goal_mu, r_goal_logvar.mul(0.5).exp_())
            p_goal_out = self.goal_prior_net(pred_goal_h)
            p_goal_mu = p_goal_out[:, :self.cfgs.goal_latent_size]
            p_goal_logvar = p_goal_out[:, self.cfgs.goal_latent_size:]

            # z_p = reparametrize(p_goal_mu, p_goal_logvar)
            # z_r = reparametrize(r_goal_mu, r_goal_logvar)
            # BetaTCVAE
            # kld_loss = beta_tcvae_loss(z_r, r_goal_mu, r_goal_logvar, z_p, p_goal_mu, p_goal_logvar, alpha, beta, gamma)
            # InfoVAE
            # kld_loss = 0.5 * ((r_goal_logvar.exp() / p_goal_logvar.exp()) + (p_goal_mu - r_goal_mu).pow(2) / p_goal_logvar.exp() - 1 + (p_goal_logvar - r_goal_logvar))
            # kld_loss = 0.5 * ((p_goal_logvar.exp() / r_goal_logvar.exp()) + (r_goal_mu - p_goal_mu).pow(2) / r_goal_logvar.exp() - 1 + (r_goal_logvar - p_goal_logvar))
            # kld_loss = alpha * kld_loss.sum(dim=-1).mean()
            # kld_loss = 0.5 * ((p_goal_logvar.exp() / r_goal_logvar.exp()) + (r_goal_mu - p_goal_mu).pow(2) / r_goal_logvar.exp() - 1 + (r_goal_logvar - p_goal_logvar))
            kld_loss = alpha * (0.5 * torch.sum((r_goal_logvar.exp() / p_goal_logvar.exp()) + (p_goal_mu - r_goal_mu).pow(2) / p_goal_logvar.exp() - 1 + (p_goal_logvar - r_goal_logvar), dim=-1).mean())

            # if self.cfgs.kld_clamp:
            #     kld_loss = torch.clamp(kld_loss, min=self.cfgs.kld_clamp)
            # kld_loss += beta * info_vae_loss(z_p, z_r)
            r_goal_std = r_goal_logvar.mul(0.5).exp()
            k_r_goal_mu = r_goal_mu.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
            k_r_goal_std = r_goal_std.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size

            k_pred_goal_h = pred_goal_h.unsqueeze(1).repeat(1, k_sample, 1)  # B * goal_hidden_size
            eps = torch.randn_like(k_r_goal_std)
            k_goal_latent = eps.mul(k_r_goal_std).add(k_r_goal_mu) # B * K * H
            k_pred_goal_h = torch.cat([k_goal_latent, k_pred_goal_h], dim=-1)
            bk_pred_goal_h = k_pred_goal_h.reshape(-1, self.cfgs.goal_latent_size + self.cfgs.hidden_size)  # K*B*L => B*K*L => BK * L

        bk_pred_goals = self.sbert_decoder(bk_pred_goal_h)  # BK * S
        pred_goals = bk_pred_goals.reshape(len(pred_goal_h), k_sample, self.cfgs.goal_dim) # self.cfgs.spatial_dim)  # BK * S => B*K*S
        # if self.cfgs.kmean:
        #     kmeans = MultiKMeans(n_clusters=k_sample, n_kmeans=pred_goals.shape[0], max_iter=6, verbose=True)
        #     pred_goals = kmeans.fit_predict(pred_goals)
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
        # goal_h = self.enc_hidden_gru_encoder(goal_h)
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
            kld_weight=None,
    ):
        goal_enc_out = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, output_attentions=output_attentions
        )

        goal_h = self.enc_hidden_pooler(goal_enc_out["last_hidden_state"])
        # goal_h = self.enc_hidden_gru_encoder(goal_h)
        pred_goals, kld_loss = self.goal_trainer(goal_h, spatial_ids[:, self.cfgs.obs_len+self.cfgs.pred_len, :], traj_lbl, goal_lbl, k_sample=self.cfgs.k_sample)

        mgp_cvae_loss = MGPCVAELoss()
        gde_loss, best_idx = mgp_cvae_loss(pred_goals, goal_lbl, self.cfgs.k_sample, output_dim=self.cfgs.output_dim, best=True)
        kld_loss = kld_weight * kld_loss
        gde_loss = self.cfgs.goal_weight * gde_loss
        total_loss = kld_loss + gde_loss
        if self.cfgs.scene and self.cfgs.col_weight > 0:
            col_loss = self.cfgs.col_weight*goal_collision_loss(pred_goals, envs, envs_params)
            total_loss += col_loss
        else:
            col_loss = None

        return SPUBERTMGPOutput(
            total_loss=total_loss,
            gde_loss=gde_loss,
            col_loss=col_loss,
            kld_loss=kld_loss,
            pred_goals=pred_goals,
            best_idx=best_idx,
            attentions=goal_enc_out["attentions"],
        )



class SPUBERTFTOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    gde_loss: torch.FloatTensor = None
    ade_loss: torch.FloatTensor = None
    fde_loss: torch.FloatTensor = None
    kld_loss: torch.FloatTensor = None
    mgp_col_loss: torch.FloatTensor = None
    tgp_col_loss: torch.FloatTensor = None
    pred_trajs: torch.FloatTensor = None
    pred_goals: torch.FloatTensor = None
    mgp_attentions: Optional[Tuple[torch.FloatTensor]] = None
    tgp_attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTFTModel(SPUBERTModelBase):

    def __init__(self, tgp_cfgs, mgp_cfgs, cfgs):
        super().__init__(cfgs)

        if cfgs.share:
            self.tgp_model = SPUBERTTGPModel(cfgs)
            self.mgp_model = SPUBERTMGPModel(cfgs, share_enc=self.tgp_model.sbert)
        else:
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
        return SPUBERTFTOutput(
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
            kld_weight=1.0,
            traj_weight=1.0,
            goal_weight=1.0,
    ):
        mgp_out = self.mgp_model(
            spatial_ids=mgp_spatial_ids, segment_ids=mgp_segment_ids, temporal_ids=mgp_temporal_ids, attn_mask=mgp_attn_mask,
            env_spatial_ids=env_spatial_ids, env_temporal_ids=env_temporal_ids, env_segment_ids=env_segment_ids,
            env_attn_mask=env_attn_mask, envs=envs, envs_params=envs_params,
            traj_lbl=traj_lbl, goal_lbl=goal_lbl, output_attentions=output_attentions, kld_weight=kld_weight)

        tgp_out = self.tgp_model(
            spatial_ids=tgp_spatial_ids, segment_ids=tgp_segment_ids, temporal_ids=tgp_temporal_ids, attn_mask=tgp_attn_mask,
            env_spatial_ids=env_spatial_ids, env_segment_ids=env_segment_ids, env_temporal_ids=env_temporal_ids,
            env_attn_mask=env_attn_mask, envs=envs, envs_params=envs_params,
            traj_lbl=traj_lbl, goal_lbl=goal_lbl, output_attentions=output_attentions)


        return SPUBERTFTOutput(
            mgp_loss=mgp_out["total_loss"],
            tgp_loss=tgp_out["total_loss"],
            ade_loss=tgp_out["ade_loss"],
            fde_loss=tgp_out["fde_loss"],
            gde_loss=mgp_out["gde_loss"],
            tgp_col_loss=tgp_out["col_loss"],
            mgp_col_loss=mgp_out["col_loss"],
            kld_loss=mgp_out["kld_loss"],
            pred_trajs=tgp_out["pred_trajs"],
            pred_goals=mgp_out["pred_goals"],
            goal_attentions=mgp_out["attentions"],
            traj_attentions=tgp_out["attentions"])

