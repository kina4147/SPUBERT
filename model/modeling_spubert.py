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
from typing import Optional, Tuple, List
import inspect
from torch import Tensor, device, dtype
import torch.utils.checkpoint
from torch import nn
from typing import Callable
from collections import OrderedDict
from SPUBERT.model.activations import *

ACT2FN = {
    "relu": F.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class SPUBERTSelfAttention(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        if cfgs.hidden_size % cfgs.num_head != 0 and not hasattr(cfgs, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfgs.hidden_size, cfgs.num_head)
            )

        self.num_attention_heads = cfgs.num_head
        self.attention_head_size = int(cfgs.hidden_size / cfgs.num_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfgs.hidden_size, self.all_head_size)
        self.key = nn.Linear(cfgs.hidden_size, self.all_head_size)
        self.value = nn.Linear(cfgs.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cfgs.dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class SPUBERTSelfOutput(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dense = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfgs.hidden_size, eps=cfgs.layer_norm_eps)
        self.dropout = nn.Dropout(cfgs.dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SPUBERTAttention(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.self = SPUBERTSelfAttention(cfgs)
        self.output = SPUBERTSelfOutput(cfgs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SPUBERTIntermediate(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dense = nn.Linear(cfgs.hidden_size, cfgs.intermediate_size)
        self.intermediate_act_fn = ACT2FN[cfgs.act_fn]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SPUBERTOutput(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dense = nn.Linear(cfgs.intermediate_size, cfgs.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfgs.hidden_size, eps=cfgs.layer_norm_eps)
        self.dropout = nn.Dropout(cfgs.dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SPUBERTLayer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.chunk_size_feed_forward = cfgs.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SPUBERTAttention(cfgs)
        self.intermediate = SPUBERTIntermediate(cfgs)
        self.output = SPUBERTOutput(cfgs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SPUBERTEncoderOutput(OrderedDict):
    last_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SPUBERTEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.cfgs.act_fn = 'gelu'
        self.cfgs.dropout_prob = 0.1
        self.cfgs.layer_norm_eps = 1e-4
        self.layer = nn.ModuleList([SPUBERTLayer(cfgs) for _ in range(cfgs.num_layer)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        return SPUBERTEncoderOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attentions,
        )


class SPUBERTPooler(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dense = nn.Linear(cfgs.hidden_size, cfgs.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class SPUBERTModelBase(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    model.
    """

    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.cfgs.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
