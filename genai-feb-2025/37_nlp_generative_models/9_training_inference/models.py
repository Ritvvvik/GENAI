import torch
import torch.nn as nn
import torch.nn.functional as F


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# GenModel with token & positional embeddings
class TextGenerationModel3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.hidden_layer = nn.Linear(config["emb_dim"], config["hidden_dim"])
        self.output_layer = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.config = config
        print(f"Model initialized with {param_count(self)} parameters")

    def forward(self, x):
        # print(x, x.shape)
        tok_embeds = self.tok_embedding(x)
        # print(tok_embeds, tok_embeds.shape)
        positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        # print(pos_embeds, pos_embeds.shape)
        embeds = tok_embeds + pos_embeds
        x = F.relu(self.hidden_layer(embeds))
        # print(x, x.shape)
        logits = self.output_layer(x)
        # print(x, x.shape)
        return logits


# GenModel with token & positional embeddings + masked multi head attention
class TextGenerationModel4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.attn = MultiheadMaskedSelfAttention(config)
        self.hidden_layer = nn.Linear(config["emb_dim"], config["hidden_dim"])
        self.output_layer = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.config = config
        print(f"Model initialized with {param_count(self)} parameters")

    def forward(self, x):
        # print(x, x.shape)
        tok_embeds = self.tok_embedding(x)
        # print(tok_embeds, tok_embeds.shape)
        positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        # print(pos_embeds, pos_embeds.shape)
        embeds = tok_embeds + pos_embeds
        x = self.attn(embeds)
        # print(x, x.shape)
        x = x + embeds
        # print(x, x.shape)
        x = F.relu(self.hidden_layer(x))
        # print(x, x.shape)
        logits = self.output_layer(x)
        # print(x, x.shape)
        return logits


# GenModel with token & positional embeddings + masked multi head attention + layer normalization
class TextGenerationModel5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.ln_1 = nn.LayerNorm(config["emb_dim"])
        self.attn = MultiheadMaskedSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["emb_dim"])
        self.hidden_layer = nn.Linear(config["emb_dim"], config["hidden_dim"])
        self.output_layer = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.config = config
        print(f"Model initialized with {param_count(self)} parameters")

    def forward(self, x):
        # print(x, x.shape)
        tok_embeds = self.tok_embedding(x)
        # print(tok_embeds, tok_embeds.shape)
        positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        # print(pos_embeds, pos_embeds.shape)
        embeds = tok_embeds + pos_embeds
        x = self.ln_1(embeds)
        x = self.attn(x)
        # print(x, x.shape)
        x = x + embeds
        x = self.ln_2(x)
        # print(x, x.shape)
        x = F.relu(self.hidden_layer(x))
        # print(x, x.shape)
        logits = self.output_layer(x)
        # print(x, x.shape)
        return logits


# GenModel with token & positional embeddings + masked multi head attention + layer normalization + dropout + ffnn
class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_1 = nn.Dropout(p=config["drop_rate"])
        self.ln_1 = nn.LayerNorm(config["emb_dim"])
        self.attn = MultiheadMaskedSelfAttention(config)
        self.drop_2 = nn.Dropout(p=config["drop_rate"])
        self.ln_2 = nn.LayerNorm(config["emb_dim"])
        self.ff = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )
        self.drop_3 = nn.Dropout(p=config["drop_rate"])
        self.ln_3 = nn.LayerNorm(config["emb_dim"])
        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"])
        self.config = config
        print(f"Model initialized with {param_count(self)} parameters")

    def forward(self, x):
        tok_embeds = self.tok_embedding(x)
        positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        embeds = tok_embeds + pos_embeds
        x = self.drop_1(embeds)
        x_shortcut = x

        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_2(x)
        x = x + x_shortcut
        x_shortcut = x

        x = self.ln_2(x)
        x = self.ff(x)
        x = self.drop_3(x)
        x = x + x_shortcut

        x = self.ln_3(x)
        logits = self.output_layer(x)
        return logits


class MultiheadMaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config["emb_dim"] % config["n_heads"] == 0
        ), f"embedding dim should be divisible by number of heads"
        self.num_heads = config["n_heads"]
        self.embd_size = config["emb_dim"]
        # batched key, query, and value projections for all heads
        self.c_attn = nn.Linear(config["emb_dim"], 3 * config["emb_dim"])
        self.c_proj = nn.Linear(config["emb_dim"], config["emb_dim"])

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.embd_size, dim=-1)  # (B,T,C), (B,T,C), (B,T,C)
        q = q.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(
            1, 2
        )  # (B,nh,T,hs)
        k = k.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(
            1, 2
        )  # (B,nh,T,hs)
        v = v.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(
            1, 2
        )  # (B,nh,T,hs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,nh,T,hs)
        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B,nh,T,hs) --> (B,T,nh,hs) --> (B,T,C=nh*hs)
        out = self.c_proj(out)  # (B,T,C) --> (B,T,C)
        return out
