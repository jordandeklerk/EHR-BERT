from __future__ import absolute_import, division, print_function

import os
import math
import logging

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .bert_config import BertConfig

logger = logging.getLogger(__name__)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit activation function."""
    return F.gelu(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        """Construct a LayerNorm module with given hidden size and epsilon."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNorm."""
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadedAttention(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize multi-headed attention with config."""
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for multi-headed attention."""
        batch_size = query.size(0)

        query = rearrange(self.query(query), 'b t (h d) -> b h t d', h=self.h)
        key = rearrange(self.key(key), 'b t (h d) -> b h t d', h=self.h)
        value = rearrange(self.value(value), 'b t (h d) -> b h t d', h=self.h)

        x, attn = self.attention(query, key, value, mask, self.dropout)

        x = rearrange(x, 'b h t d -> b t (h d)')
        return self.output_linear(x)


class Attention(nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scores = torch.einsum('bhqd, bhkd -> bhqk', query, key) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        x = torch.einsum('bhqk, bhvd -> bhqd', p_attn, value)
        return x, p_attn


class SublayerConnection(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize sublayer connection with LayerNorm and dropout."""
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize position-wise feed-forward layer."""
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for position-wise feed-forward layer."""
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize transformer block with attention and feed-forward layers."""
        super().__init__()
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.input_sublayer = SublayerConnection(config)
        self.output_sublayer = SublayerConnection(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer block."""
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize BERT embeddings from word and token type embeddings."""
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for BERT embeddings."""
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings + self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PreTrainedBertModel(nn.Module):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        """Abstract class for pre-trained BERT model."""
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class `BertConfig`. "
                f"To create a model from a Google pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def init_bert_weights(self, module: nn.Module):
        """Initialize weights for BERT model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, state_dict: dict = None, cache_dir: str = '', *inputs, **kwargs):
        """Load a pre-trained BERT model."""
        CONFIG_NAME = "bert_config.json"
        serialization_dir = os.path.join(cache_dir, pretrained_model_name)
        
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info(f"Model config {config}")
        
        model = cls(config, *inputs, **kwargs)
        
        if state_dict is None:
            WEIGHTS_NAME = "pytorch_model.bin"
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        state_dict = cls._update_state_dict_keys(state_dict)

        missing_keys, unexpected_keys, error_msgs = [], [], []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        cls._load_state_dict_into_model(model, state_dict, metadata, missing_keys, unexpected_keys, error_msgs)

        cls._log_model_loading_info(model, missing_keys, unexpected_keys)
        return model

    @staticmethod
    def _update_state_dict_keys(state_dict: dict) -> dict:
        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        return state_dict

    @staticmethod
    def _load_state_dict_into_model(model: nn.Module, state_dict: dict, metadata: dict, missing_keys: list, unexpected_keys: list, error_msgs: list):
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    @staticmethod
    def _log_model_loading_info(model: nn.Module, missing_keys: list, unexpected_keys: list):
        if missing_keys:
            logger.info(f"Weights of {model.__class__.__name__} not initialized from pretrained model: {missing_keys}")
        if unexpected_keys:
            logger.info(f"Weights from pretrained model not used in {model.__class__.__name__}: {unexpected_keys}")


class BERT(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, proc_voc=None):
        """Initialize BERT model with embedding and transformer blocks."""
        super().__init__(config)
        self.embedding = BertEmbeddings(config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.apply(self.init_bert_weights)

    def forward(self, x: torch.Tensor, token_type_ids: torch.Tensor = None, input_positions: torch.Tensor = None, input_sides: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for BERT model."""
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, token_type_ids)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return x, x[:, 0]


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize BERT pooler with dense and activation layers."""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for BERT pooler."""
        first_token_tensor = hidden_states[:, 0]
        return self.activation(self.dense(first_token_tensor))


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: BertConfig):
        """Initialize prediction head transform with dense and LayerNorm."""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for prediction head transform."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        return self.LayerNorm(hidden_states)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size: int = None):
        """Initialize language model prediction head with transform and decoder."""
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size if voc_size is None else voc_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for language model prediction head."""
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)