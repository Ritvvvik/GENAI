import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
from mha import *


class TextGenerationModel4(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, hidden_dim, n_heads):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(context_length, emb_dim)
        self.attn = MultiheadSelfAttention(emb_dim=emb_dim, n_heads=n_heads)
        self.hidden_layer = nn.Linear(emb_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        print(x, x.shape)
        tok_embeds = self.tok_embedding(x)
        print(tok_embeds, tok_embeds.shape)
        positions = torch.arange(0, x.shape[1]).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        print(pos_embeds, pos_embeds.shape)
        embeds = tok_embeds + pos_embeds
        print(embeds, embeds.shape)
        x = self.attn(embeds, True)
        print(x, x.shape)
        x = x + embeds
        print(x, x.shape)
        x = F.relu(self.hidden_layer(x))
        print(x, x.shape)
        logits = self.output_layer(x)
        print(logits, logits.shape)
        return logits


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    context_length = 10
    model = TextGenerationModel4(tokenizer.vocab_size, context_length, 128, 64, 1)
    prompt = "Life is short, eat dessert first"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    response = []
    input = inputs["input_ids"]
    for i in range(1):
        input = input[:, -context_length:]
        logits = model(input)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        id_next = torch.argmax(probs, dim=-1)
        # print(id_next)
        next_word = tokenizer.decode(id_next, skip_special_tokens=True)
        response.append(next_word)
        input = torch.cat((input, id_next.unsqueeze(0)), dim=1)
    print(" ".join(response))
