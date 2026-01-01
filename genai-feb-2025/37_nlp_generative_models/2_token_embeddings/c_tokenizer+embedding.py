import torch
from torch.nn import Embedding
from transformers import AutoTokenizer

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

torch.manual_seed(123)
embed_dim = 50
embed = Embedding(tokenizer.vocab_size, embed_dim)
print(embed.state_dict())

sentences = [
    "Life is short, eat dessert first",
    "I love ice cream",
    "I love chocolate cake",
]
inputs = tokenizer(
    sentences, padding="max_length", truncation=True, max_length=10, return_tensors="pt"
)
print(inputs["input_ids"], inputs["input_ids"].shape)
output = embed(inputs["input_ids"])
print(output, output.shape)
