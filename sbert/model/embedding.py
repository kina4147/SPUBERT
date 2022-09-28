import torch.nn as nn
from transformers.activations import ACT2FN


class TemporalEmbedding(nn.Embedding):
    def __init__(self, frame_size, embedding_dim=512, padding_idx=0):
        super().__init__(num_embeddings=frame_size+1, embedding_dim=embedding_dim, padding_idx=padding_idx)


class SegmentEmbedding(nn.Embedding):
    """ # Agent ID: 0 is target trajectory """
    def __init__(self, segment_size, embedding_dim=512, padding_idx=0):
        super().__init__(num_embeddings=segment_size+1, embedding_dim=embedding_dim, padding_idx=padding_idx)


class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=512, act_fn='relu'):
        super(SpatialEmbedding, self).__init__()
        self.linear1 = nn.Linear(input_dim, embedding_dim)
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x):
        x = self.act_fn(self.linear1(x))
        return x


class SBertEmbeddings(nn.Module):
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

