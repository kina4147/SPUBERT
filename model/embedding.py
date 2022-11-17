import torch.nn as nn
import torch
from transformers.activations import ACT2FN

class TemporalEmbedding(nn.Embedding):
    def __init__(self, frame_size, embedding_dim=512, padding_idx=0):
        super().__init__(num_embeddings=frame_size+1, embedding_dim=embedding_dim, padding_idx=padding_idx)


# Agent ID: 0 is target trajectory
class SegmentEmbedding(nn.Embedding):
    def __init__(self, segment_size, embedding_dim=512, padding_idx=0):
        super().__init__(num_embeddings=segment_size+1, embedding_dim=embedding_dim, padding_idx=padding_idx)


class ModalEmbedding(nn.Embedding):
    def __init__(self, modal_size, embedding_dim=512):
        super().__init__(num_embeddings=modal_size, embedding_dim=embedding_dim)


class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=512, act_fn='relu'):
        super(SpatialEmbedding, self).__init__()
        self.linear1 = nn.Linear(input_dim, embedding_dim)
        self.act_fn = ACT2FN[act_fn]


    def forward(self, x):
        x = self.act_fn(self.linear1(x))
        return x

# Map Embeddings: CGridMap / GridMap / RayCasting
# patch or map => flatten()
class GridMapEmbeddings(nn.Module):
    def __init__(self, width, height, embedding_dim=512, act_fn='relu'):
        super(GridMapEmbeddings, self).__init__()
        self.linear1 = nn.Linear(width*height, embedding_dim)
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x): # patch # matrix input? # vector input?
        x = self.act_fn(self.linear1(x))
        return x


class SPUBERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, cfgs):
        super().__init__()
        self.spatial_embeddings = SpatialEmbedding(input_dim=cfgs.input_dim, embedding_dim=cfgs.hidden_size, act_fn=cfgs.act_fn)
        self.temporal_embeddings = TemporalEmbedding(frame_size=cfgs.obs_len+cfgs.pred_len, embedding_dim=cfgs.hidden_size, padding_idx=cfgs.pad_token_id)
        self.segment_embeddings = SegmentEmbedding(segment_size=cfgs.num_nbr+1, embedding_dim=cfgs.hidden_size, padding_idx=cfgs.pad_token_id)

        self.LayerNorm = nn.LayerNorm(cfgs.hidden_size, eps=cfgs.layer_norm_eps)
        self.dropout = nn.Dropout(cfgs.dropout_prob)

    def forward(
        self, spatial_ids=None, temporal_ids=None, segment_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        spatial_embeddings = self.spatial_embeddings(spatial_ids)
        temporal_embeddings = self.temporal_embeddings(temporal_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = spatial_embeddings + temporal_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SPUBERTMMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, cfgs):
        super().__init__()
        self.spatial_embeddings = SpatialEmbedding(input_dim=cfgs.input_dim, embedding_dim=cfgs.hidden_size,
                                                   act_fn=cfgs.act_fn)
        self.temporal_embeddings = TemporalEmbedding(frame_size=cfgs.obs_len + cfgs.pred_len,
                                                     embedding_dim=cfgs.hidden_size, padding_idx=cfgs.pad_token_id)
        # num_map_patch: 4 / 9 / 16
        self.segment_embeddings = SegmentEmbedding(segment_size=cfgs.num_nbr + cfgs.num_patch + 1, embedding_dim=cfgs.hidden_size,
                                                   padding_idx=cfgs.pad_token_id)

        # self.modal_embeddings = ModalEmbedding(modal_size=2, embedding_dim=cfgs.hidden_size)
        self.env_spatial_embeddings = GridMapEmbeddings(width=cfgs.patch_size, height=cfgs.patch_size, embedding_dim=cfgs.hidden_size, act_fn=cfgs.act_fn)

        self.LayerNorm = nn.LayerNorm(cfgs.hidden_size, eps=cfgs.layer_norm_eps)
        self.dropout = nn.Dropout(cfgs.dropout_prob)


    def forward(
            self, spatial_ids=None, temporal_ids=None, segment_ids=None, env_spatial_ids=None, env_temporal_ids=None, env_segment_ids=None, modal_ids=None, inputs_embeds=None,
            past_key_values_length=0
    ):
        spatial_embeddings = self.spatial_embeddings(spatial_ids)
        temporal_embeddings = self.temporal_embeddings(temporal_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        modal_ids = torch.zeros_like(segment_ids)
        # modal_embeddings = self.modal_embeddings(modal_ids)
        trajectory_embeddings = spatial_embeddings + temporal_embeddings + segment_embeddings# + modal_embeddings

        env_spatial_embeddings = self.env_spatial_embeddings(env_spatial_ids)
        env_temporal_embeddings = self.temporal_embeddings(env_temporal_ids)
        env_segment_embeddings = self.segment_embeddings(env_segment_ids)
        env_modal_ids = torch.ones_like(env_segment_ids)
        # env_modal_embeddings = self.modal_embeddings(env_modal_ids)
        scene_embeddings = env_spatial_embeddings + env_temporal_embeddings + env_segment_embeddings# + env_modal_embeddings
        total_embeddings = torch.cat([trajectory_embeddings, scene_embeddings], dim=1)

        total_embeddings = self.LayerNorm(total_embeddings)
        total_embeddings = self.dropout(total_embeddings)
        return total_embeddings
