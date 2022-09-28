# coding=utf-8
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
"""PyTorch BERT model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List
import inspect

from torch import Tensor, device, dtype
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Callable
from collections import OrderedDict
from SocialBERT.sbertplus.model.activations import *

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


# logger = logging.get_logger(__name__)


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


class SBertSelfAttention(nn.Module):
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


class SBertSelfOutput(nn.Module):
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


class SBertAttention(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.self = SBertSelfAttention(cfgs)
        self.output = SBertSelfOutput(cfgs)

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


class SBertIntermediate(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dense = nn.Linear(cfgs.hidden_size, cfgs.intermediate_size)
        self.intermediate_act_fn = ACT2FN[cfgs.act_fn]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SBertOutput(nn.Module):
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


class SBertLayer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.chunk_size_feed_forward = cfgs.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SBertAttention(cfgs)
        self.intermediate = SBertIntermediate(cfgs)
        self.output = SBertOutput(cfgs)

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


class SBertEncoderOutput(OrderedDict):
    last_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SBertEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.cfgs.act_fn = 'gelu'
        self.cfgs.dropout_prob = 0.1
        self.cfgs.layer_norm_eps = 1e-4
        self.layer = nn.ModuleList([SBertLayer(cfgs) for _ in range(cfgs.num_layer)])

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

        return SBertEncoderOutput(
            last_hidden_state=hidden_states,
            attentions=all_self_attentions,
        )


class SBertPooler(nn.Module):
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



class SBertModelBase(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    model.
    """

    # cfgs_class = SberBertConfig
    # load_tf_weights = load_tf_weights_in_bert
    # base_model_prefix = "bert"
    # _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

    def init_weights(self):
        """
        Initializes and prunes weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)

    #     # Tie weights if needed
    #     self.tie_weights()
    #
    # def tie_weights(self):
    #     """
    #     Tie the weights between the input embeddings and the output embeddings.
    #
    #     If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
    #     the weights instead.
    #     """
    #     output_embeddings = self.get_output_embeddings()
    #     if output_embeddings is not None and self.config.tie_word_embeddings:
    #         self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
    #
    #     if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
    #         if hasattr(self, self.base_model_prefix):
    #             self = getattr(self, self.base_model_prefix)
    #         self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

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

    # def save_pretrained(self, save_directory: Union[str, os.PathLike]):
    #     """
    #     Save a model and its configuration file to a directory, so that it can be re-loaded using the
    #     `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
    #
    #     Arguments:
    #         save_directory (:obj:`str` or :obj:`os.PathLike`):
    #             Directory to which to save. Will be created if it doesn't exist.
    #     """
    #     if os.path.isfile(save_directory):
    #         logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
    #         return
    #     os.makedirs(save_directory, exist_ok=True)
    #
    #     # Only save the model itself if we are using distributed training
    #     model_to_save = self.module if hasattr(self, "module") else self
    #
    #     # Attach architecture to the config
    #     model_to_save.config.architectures = [model_to_save.__class__.__name__]
    #
    #     state_dict = model_to_save.state_dict()
    #
    #     # Handle the case where some state_dict keys shouldn't be saved
    #     if self._keys_to_ignore_on_save is not None:
    #         state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}
    #
    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    #
    #     if getattr(self.config, "xla_device", False) and is_torch_tpu_available():
    #         import torch_xla.core.xla_model as xm
    #
    #         if xm.is_master_ordinal():
    #             # Save configuration file
    #             model_to_save.config.save_pretrained(save_directory)
    #         # xm.save takes care of saving only from master
    #         xm.save(state_dict, output_model_file)
    #     else:
    #         model_to_save.config.save_pretrained(save_directory)
    #         torch.save(state_dict, output_model_file)
    #
    #     logger.info("Model weights saved in {}".format(output_model_file))
    #
    #
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    #     r"""
    #     Instantiate a pretrained pytorch model from a pre-trained model configuration.
    #
    #     The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
    #     train the model, you should first set it back in training mode with ``model.train()``.
    #
    #     The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
    #     pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    #     task.
    #
    #     The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
    #     weights are discarded.
    #
    #     Parameters:
    #         pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
    #             Can be either:
    #
    #                 - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
    #                   Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
    #                   a user or organization name, like ``dbmdz/bert-base-german-cased``.
    #                 - A path to a `directory` containing model weights saved using
    #                   :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
    #                 - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
    #                   this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
    #                   as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
    #                   a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
    #                 - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
    #                   arguments ``config`` and ``state_dict``).
    #         model_args (sequence of positional arguments, `optional`):
    #             All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
    #         config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
    #             Can be either:
    #
    #                 - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
    #                 - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.
    #
    #             Configuration for the model to use instead of an automatically loaded configuation. Configuration can
    #             be automatically loaded when:
    #
    #                 - The model is a model provided by the library (loaded with the `model id` string of a pretrained
    #                   model).
    #                 - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
    #                   by supplying the save directory.
    #                 - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
    #                   configuration JSON file named `config.json` is found in the directory.
    #         state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
    #             A state dictionary to use instead of a state dictionary loaded from saved weights file.
    #
    #             This option can be used if you want to create a model from a pretrained configuration but load your own
    #             weights. In this case though, you should check if using
    #             :func:`~transformers.PreTrainedModel.save_pretrained` and
    #             :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
    #         cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
    #             Path to a directory in which a downloaded pretrained model configuration should be cached if the
    #             standard cache should not be used.
    #         from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
    #             Load the model weights from a TensorFlow checkpoint save file (see docstring of
    #             ``pretrained_model_name_or_path`` argument).
    #         force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
    #             Whether or not to force the (re-)download of the model weights and configuration files, overriding the
    #             cached versions if they exist.
    #         resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
    #             Whether or not to delete incompletely received files. Will attempt to resume the download if such a
    #             file exists.
    #         proxies (:obj:`Dict[str, str], `optional`):
    #             A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
    #             'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
    #         output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
    #             Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
    #         local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
    #             Whether or not to only look at local files (i.e., do not try to download the model).
    #         use_auth_token (:obj:`str` or `bool`, `optional`):
    #             The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
    #             generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
    #         revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
    #             The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
    #             git-based system for storing model and other artifacts on huggingface.co, so ``revision`` can be any
    #             identifier allowed by git.
    #         mirror(:obj:`str`, `optional`, defaults to :obj:`None`):
    #             Mirror source to accelerate downloads in China. If you are from China and have an accessibility
    #             problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
    #             Please refer to the mirror site for more information.
    #         kwargs (remaining dictionary of keyword arguments, `optional`):
    #             Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
    #             :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
    #             automatically loaded:
    #
    #                 - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
    #                   underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
    #                   already been done)
    #                 - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
    #                   initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
    #                   ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
    #                   with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
    #                   attribute will be passed to the underlying model's ``__init__`` function.
    #
    #     .. note::
    #
    #         Passing :obj:`use_auth_token=True` is required when you want to use a private model.
    #
    #     Examples::
    #
    #         >>> from transformers import BertConfig, BertModel
    #         >>> # Download model and configuration from huggingface.co and cache.
    #         >>> model = BertModel.from_pretrained('bert-base-uncased')
    #         >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
    #         >>> model = BertModel.from_pretrained('./test/saved_model/')
    #         >>> # Update configuration during loading.
    #         >>> model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    #         >>> assert model.config.output_attentions == True
    #         >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
    #         >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
    #         >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
    #     """
    #     config = kwargs.pop("config", None)
    #     state_dict = kwargs.pop("state_dict", None)
    #     cache_dir = kwargs.pop("cache_dir", None)
    #     from_tf = kwargs.pop("from_tf", False)
    #     force_download = kwargs.pop("force_download", False)
    #     resume_download = kwargs.pop("resume_download", False)
    #     proxies = kwargs.pop("proxies", None)
    #     output_loading_info = kwargs.pop("output_loading_info", False)
    #     local_files_only = kwargs.pop("local_files_only", False)
    #     use_auth_token = kwargs.pop("use_auth_token", None)
    #     revision = kwargs.pop("revision", None)
    #     mirror = kwargs.pop("mirror", None)
    #
    #     # Load config if we don't provide a configuration
    #     if not isinstance(config, PretrainedConfig):
    #         config_path = config if config is not None else pretrained_model_name_or_path
    #         config, model_kwargs = cls.config_class.from_pretrained(
    #             config_path,
    #             *model_args,
    #             cache_dir=cache_dir,
    #             return_unused_kwargs=True,
    #             force_download=force_download,
    #             resume_download=resume_download,
    #             proxies=proxies,
    #             local_files_only=local_files_only,
    #             use_auth_token=use_auth_token,
    #             revision=revision,
    #             **kwargs,
    #         )
    #     else:
    #         model_kwargs = kwargs
    #
    #     # Load model
    #     if pretrained_model_name_or_path is not None:
    #         pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    #         if os.path.isdir(pretrained_model_name_or_path):
    #             if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
    #                 # Load from a TF 1.0 checkpoint in priority if from_tf
    #                 archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
    #             elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
    #                 # Load from a TF 2.0 checkpoint in priority if from_tf
    #                 archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
    #             elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
    #                 # Load from a PyTorch checkpoint
    #                 archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
    #             else:
    #                 raise EnvironmentError(
    #                     "Error no file named {} found in directory {} or `from_tf` set to False".format(
    #                         [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
    #                         pretrained_model_name_or_path,
    #                     )
    #                 )
    #         elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
    #             archive_file = pretrained_model_name_or_path
    #         elif os.path.isfile(pretrained_model_name_or_path + ".index"):
    #             assert (
    #                 from_tf
    #             ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
    #                 pretrained_model_name_or_path + ".index"
    #             )
    #             archive_file = pretrained_model_name_or_path + ".index"
    #         else:
    #             archive_file = hf_bucket_url(
    #                 pretrained_model_name_or_path,
    #                 filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
    #                 revision=revision,
    #                 mirror=mirror,
    #             )
    #
    #         try:
    #             # Load from URL or cache if already cached
    #             resolved_archive_file = cached_path(
    #                 archive_file,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 resume_download=resume_download,
    #                 local_files_only=local_files_only,
    #                 use_auth_token=use_auth_token,
    #             )
    #         except EnvironmentError as err:
    #             logger.error(err)
    #             msg = (
    #                 f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
    #                 f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
    #                 f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
    #             )
    #             raise EnvironmentError(msg)
    #
    #         if resolved_archive_file == archive_file:
    #             logger.info("loading weights file {}".format(archive_file))
    #         else:
    #             logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
    #     else:
    #         resolved_archive_file = None
    #
    #     config.name_or_path = pretrained_model_name_or_path
    #     # Instantiate model.
    #     model = cls(config, *model_args, **model_kwargs)
    #
    #     if state_dict is None and not from_tf:
    #         try:
    #             state_dict = torch.load(resolved_archive_file, map_location="cpu")
    #         except Exception:
    #             raise OSError(
    #                 f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
    #                 f"at '{resolved_archive_file}'"
    #                 "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
    #             )
    #
    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []
    #
    #     if from_tf:
    #         if resolved_archive_file.endswith(".index"):
    #             # Load from a TensorFlow 1.X checkpoint - provided by original authors
    #             model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
    #         else:
    #             # Load from our TensorFlow 2.0 checkpoints
    #             try:
    #                 from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model
    #
    #                 model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
    #             except ImportError:
    #                 logger.error(
    #                     "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
    #                     "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
    #                 )
    #                 raise
    #     else:
    #         # Convert old format to new format if needed from a PyTorch state_dict
    #         old_keys = []
    #         new_keys = []
    #         for key in state_dict.keys():
    #             new_key = None
    #             if "gamma" in key:
    #                 new_key = key.replace("gamma", "weight")
    #             if "beta" in key:
    #                 new_key = key.replace("beta", "bias")
    #             if new_key:
    #                 old_keys.append(key)
    #                 new_keys.append(new_key)
    #         for old_key, new_key in zip(old_keys, new_keys):
    #             state_dict[new_key] = state_dict.pop(old_key)
    #
    #         # copy state_dict so _load_from_state_dict can modify it
    #         metadata = getattr(state_dict, "_metadata", None)
    #         state_dict = state_dict.copy()
    #         if metadata is not None:
    #             state_dict._metadata = metadata
    #
    #         # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    #         # so we need to apply the function recursively.
    #         def load(module: nn.Module, prefix=""):
    #             local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    #             module._load_from_state_dict(
    #                 state_dict,
    #                 prefix,
    #                 local_metadata,
    #                 True,
    #                 missing_keys,
    #                 unexpected_keys,
    #                 error_msgs,
    #             )
    #             for name, child in module._modules.items():
    #                 if child is not None:
    #                     load(child, prefix + name + ".")
    #
    #         # Make sure we are able to load base model as well as derived model (with heads)
    #         start_prefix = ""
    #         model_to_load = model
    #         has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
    #         if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
    #             start_prefix = cls.base_model_prefix + "."
    #         if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
    #             model_to_load = getattr(model, cls.base_model_prefix)
    #
    #         load(model_to_load, prefix=start_prefix)
    #
    #         if model.__class__.__name__ != model_to_load.__class__.__name__:
    #             base_model_state_dict = model_to_load.state_dict().keys()
    #             head_model_state_dict_without_base_prefix = [
    #                 key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
    #             ]
    #             missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)
    #
    #         # Some model may have keys that are not in the state by design, removing them before needlessly warning
    #         # the user.
    #         if cls._keys_to_ignore_on_load_missing is not None:
    #             for pat in cls._keys_to_ignore_on_load_missing:
    #                 missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    #
    #         if cls._keys_to_ignore_on_load_unexpected is not None:
    #             for pat in cls._keys_to_ignore_on_load_unexpected:
    #                 unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    #
    #         if len(unexpected_keys) > 0:
    #             logger.warning(
    #                 f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
    #                 f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
    #                 f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
    #                 f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
    #                 f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
    #                 f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
    #             )
    #         else:
    #             logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    #         if len(missing_keys) > 0:
    #             logger.warning(
    #                 f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
    #                 f"and are newly initialized: {missing_keys}\n"
    #                 f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
    #             )
    #         else:
    #             logger.info(
    #                 f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
    #                 f"If your task is similar to the task the model of the checkpoint was trained on, "
    #                 f"you can already use {model.__class__.__name__} for predictions without further training."
    #             )
    #         if len(error_msgs) > 0:
    #             raise RuntimeError(
    #                 "Error(s) in loading state_dict for {}:\n\t{}".format(
    #                     model.__class__.__name__, "\n\t".join(error_msgs)
    #                 )
    #             )
    #     # make sure token embedding weights are still tied if needed
    #     model.tie_weights()
    #
    #     # Set model in evaluation mode to deactivate DropOut modules by default
    #     model.eval()
    #
    #     if output_loading_info:
    #         loading_info = {
    #             "missing_keys": missing_keys,
    #             "unexpected_keys": unexpected_keys,
    #             "error_msgs": error_msgs,
    #         }
    #         return model, loading_info
    #
    #     if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
    #         import torch_xla.core.xla_model as xm
    #
    #         model = xm.send_cpu_data_to_device(model, xm.xla_device())
    #         model.to(xm.xla_device())
    #
    #     return model


    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.cfgs.initializer_range)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    # def get_input_embeddings(self):
    #     return self.embeddings.spatial_embeddings
    #
    # def set_input_embeddings(self, value):
    #     self.embeddings.spatial_embeddings = value