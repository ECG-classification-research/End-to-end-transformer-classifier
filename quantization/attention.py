"""Explicit self-attention preserving the original FP32 model."""
import copy
import math
import torch
from torch import nn


class ExplicitAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        if (not attention.batch_first or attention.in_proj_weight is None
                or attention.bias_k is not None or attention.bias_v is not None
                or attention.add_zero_attn or attention.dropout != 0):
            raise ValueError('Unsupported attention configuration for ECG export.')
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        for i, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
            layer = nn.Linear(self.embed_dim, self.embed_dim, bias=attention.in_proj_bias is not None)
            layer.to(device=attention.in_proj_weight.device, dtype=attention.in_proj_weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(attention.in_proj_weight[i*self.embed_dim:(i+1)*self.embed_dim])
                if layer.bias is not None:
                    layer.bias.copy_(attention.in_proj_bias[i*self.embed_dim:(i+1)*self.embed_dim])
            setattr(self, name, layer)
        self.out_proj = copy.deepcopy(attention.out_proj)

    def forward(self, query, key, value, need_weights=False):
        batch, tokens, _ = query.shape
        def heads(x):
            return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.q_proj(query)), heads(self.k_proj(key)), heads(self.v_proj(value))
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        result = torch.matmul(weights, v).transpose(1, 2).reshape(batch, tokens, self.embed_dim)
        return self.out_proj(result), weights.mean(dim=1) if need_weights else None


def explicit_model(model):
    result = copy.deepcopy(model)
    for block in result.transformer_encoder.blocks:
        block.attention = ExplicitAttention(block.attention)
    return result
