import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["emb_dim"], 4 * config["emb_dim"])
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config["emb_dim"], config["emb_dim"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.attn = MultiheadMaskedSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config["emb_dim"])
        self.ln_2 = nn.LayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config["vocab_size"], config["emb_dim"]),
                wpe=nn.Embedding(config["context_length"], config["emb_dim"]),
                drop=nn.Dropout(config["drop_rate"]),
                h=nn.ModuleList(
                    [DecoderBlock(config) for _ in range(config["n_layers"])]
                ),
                ln_f=nn.LayerNorm(config["emb_dim"]),
            )
        )

        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.config = config
        self.apply(self._init_weights)
        print(f"Model parameter count:{param_count(self)}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # print(x, x.shape)
        tok_embeds = self.transformer.wte(x)
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_embeds = self.transformer.wpe(positions)
        x = tok_embeds + pos_embeds
        # print(x, x.shape)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        # print(x, x.shape)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # print(logits, logits.shape)
        return logits

    @classmethod
    def from_pretrained(cls, type, vocab_size):
        assert type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print(f">>> Begin Loading Weights of pretrained model:{type}")
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, emb_dim=768),  # 124M params
            "gpt2-medium": dict(n_layers=24, n_heads=16, emb_dim=1024),  # 350M params
            "gpt2-large": dict(n_layers=36, n_heads=20, emb_dim=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, emb_dim=1600),  # 1558M params
        }
        config = config_args[type]
        config["vocab_size"] = vocab_size
        config["context_length"] = 1024
        config["drop_rate"] = 0.0

        model = GPT2Model(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys) == len(
            sd_keys_hf
        ), f"mismatched keys {len(sd_keys)} != {len(sd_keys_hf)}"

        # copy while ensuring all parameters are aligned in names and shape
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # need to transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print(f">>> End Loading Weights of pretrained model:{type}")
        return model
