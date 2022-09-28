
import torch
from torch import nn

from typing import Optional, Tuple
from transformers.activations import ACT2FN
from collections import OrderedDict
from torch.nn import functional as F

from SocialBERT.sbert.model.modeling_sbert import SBertEncoder, SBertModelBase
from SocialBERT.sbert.model.embedding import SBertEmbeddings
from SocialBERT.sbert.model.decoder import SIPDecoder, MTPDecoder
from SocialBERT.sbert.model.pooler import SIPPooler, MTPPooler, FutureTrajPooler
from SocialBERT.sbert.model.loss import ADELoss, FDELoss, MaskedADELoss

class SBertConfig:
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
            view_range=20.0,
            view_angle=1.0,
            social_range=2.0,
            obs_len=8,
            pred_len=12,
            num_nbr=5,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            sip=False
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
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

        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = 0
        self.initializer_range = initializer_range
        self.sip = sip

        # self.scale = scale


class SBertModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.embeddings = SBertEmbeddings(cfgs)
        self.sbert_encoder = SBertEncoder(cfgs)
        self.init_weights()

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            output_attentions=False
    ):
        device = spatial_ids.device
        traj_batch_size, traj_seq_len, traj_spatial_dim = spatial_ids.size()

        ext_all_attn_mask: torch.Tensor = self.get_extended_attention_mask(
            attn_mask, (traj_batch_size, traj_seq_len), device)
        emb_h = self.embeddings(spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids)
        enc_h = self.sbert_encoder(hidden_states=emb_h, attention_mask=ext_all_attn_mask, output_attentions=output_attentions)

        return enc_h


class SBertPTOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    mtp_loss: torch.FloatTensor = None
    sip_loss: torch.FloatTensor = None
    mtp_output: torch.FloatTensor = None
    sip_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SBertPTModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertModel(cfgs=cfgs)
        self.sip_pooler = SIPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len)
        self.sip_decoder = SIPDecoder(hidden_size=cfgs.hidden_size)
        self.mtp_pooler = MTPPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len, num_nbr=cfgs.num_nbr)
        self.mtp_decoder = MTPDecoder(hidden_size=cfgs.hidden_size, out_dim=cfgs.output_dim, layer_norm_eps=cfgs.layer_norm_eps, act_fn=cfgs.act_fn)
        self.init_weights()

    def forward(
            self,
            train,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            traj_mask=None,
            traj_lbl=None,
            near_lbl=None,
            output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask, output_attentions=output_attentions
        )
        mtp_h = self.mtp_pooler(enc_h["last_hidden_state"])
        sip_h = self.sip_pooler(enc_h["last_hidden_state"])
        mtp_out = self.mtp_decoder(mtp_h)
        sip_out = self.sip_decoder(sip_h)
        if self.cfgs.sip:
            loss_mtp = MaskedADELoss()
            mtp_loss = loss_mtp(mtp_out, traj_lbl, traj_mask)
            loss_sip = nn.NLLLoss(ignore_index=0, reduction='mean')
            sip_loss = loss_sip(sip_out.view(-1, 3), near_lbl.view(-1))
            total_loss = mtp_loss + sip_loss
        else:
            loss_mtp = MaskedADELoss()
            mtp_loss = loss_mtp(mtp_out, traj_lbl, traj_mask)
            total_loss = mtp_loss
            sip_loss = None
        return SBertPTOutput(
            total_loss=total_loss,
            mtp_loss=mtp_loss,
            sip_loss=sip_loss,
            mtp_output=mtp_out,
            sip_output=sip_out,
            attentions=enc_h["attentions"],
        )

class SBertFTOutput(OrderedDict):
    total_loss: torch.FloatTensor = None
    ade_loss: torch.FloatTensor = None
    fde_loss: torch.FloatTensor = None
    pred_traj: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SBertFTModel(SBertModelBase):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.sbert = SBertModel(cfgs=cfgs)
        self.traj_pooler = FutureTrajPooler(obs_len=cfgs.obs_len, pred_len=cfgs.pred_len)
        self.sbert_decoder = MTPDecoder(hidden_size=cfgs.hidden_size, out_dim=cfgs.output_dim, layer_norm_eps=cfgs.layer_norm_eps, act_fn=cfgs.act_fn)
        self.init_weights()

    def forward(
            self,
            spatial_ids,
            temporal_ids,
            segment_ids,
            attn_mask,
            traj_lbl=None,
            train=False,
            output_attentions=False,
    ):
        enc_h = self.sbert(
            spatial_ids=spatial_ids, segment_ids=segment_ids, temporal_ids=temporal_ids, attn_mask=attn_mask, output_attentions=output_attentions
        )
        traj_h = self.traj_pooler(enc_h["last_hidden_state"])
        pred_traj = self.sbert_decoder(traj_h)  # (B*K, Seq, Sdim)
        if train:
            loss_ade = ADELoss()
            loss_fde = FDELoss()
            ade_loss = loss_ade(pred_traj, traj_lbl)
            fde_loss = loss_fde(pred_traj, traj_lbl)
            total_loss = ade_loss # + fde_loss
            return SBertFTOutput(
                total_loss=total_loss,
                ade_loss=ade_loss,
                fde_loss=fde_loss,
                pred_traj=pred_traj,
                attentions=enc_h["attentions"],
            )
        else:
            return SBertFTOutput(
                pred_traj=pred_traj,
                attentions=enc_h["attentions"],
            )