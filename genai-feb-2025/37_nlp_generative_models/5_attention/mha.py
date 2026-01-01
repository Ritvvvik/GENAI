import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class SingleheadSelfAttention(nn.Module):
    def __init__(self, emb_dim, k_dim, v_dim):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, k_dim)
        self.W_k = nn.Linear(emb_dim, k_dim)
        self.W_v = nn.Linear(emb_dim, v_dim)

    def forward(self, x, mask=False):
        print(x, x.shape)
        Q = self.W_q(x)
        print(Q, Q.shape)
        K = self.W_k(x)
        print(K, K.shape)
        V = self.W_v(x)
        print(V, V.shape)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[2])
        print(attn_scores, attn_scores.shape)
        if mask:
            mask_matrix = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask_matrix.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print(attn_weights, attn_weights.shape)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert emb_dim % n_heads == 0

        self.attention_heads = nn.ModuleList(
            [
                SingleheadSelfAttention(emb_dim, emb_dim // n_heads, emb_dim // n_heads)
                for _ in range(n_heads)
            ]
        )
        self.W_o = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=False):
        head_outputs = [attn_head(x, mask) for attn_head in self.attention_heads]
        print(head_outputs, len(head_outputs), head_outputs[0].shape)
        concat_output = torch.cat(head_outputs, dim=-1)
        print(concat_output, concat_output.shape)
        output = self.W_o(concat_output)
        return output
