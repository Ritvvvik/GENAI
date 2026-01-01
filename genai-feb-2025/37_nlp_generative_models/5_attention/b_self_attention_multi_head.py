import torch
import torch.nn as nn
from transformers import AutoTokenizer
from mha import *

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
sentence = "A fluffy blue creature roamed the verdant forest."
inputs = tokenizer(
    sentence, padding="max_length", truncation=True, max_length=10, return_tensors="pt"
)

torch.manual_seed(123)
embed_dim = 6
embed = nn.Embedding(tokenizer.vocab_size, embed_dim)
embedded_sentence = embed(inputs["input_ids"])

attention = MultiheadSelfAttention(emb_dim=embed_dim, n_heads=2)
output = attention(embedded_sentence)
print(output, output.shape)
