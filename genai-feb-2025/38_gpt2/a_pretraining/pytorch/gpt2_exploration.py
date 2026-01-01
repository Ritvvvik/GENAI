from dataloader import *
from models import GPT2Model
from trainer import *
from inference import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(
    {
        "pad_token": tokenizer.eos_token,
    }
)

prompts = [
    "Every effort moves you",
    "The quick brown fox jumps over",
    "the capital of France is",
    "Scientists have made a new discovery in Antarctica",
    "The reason football is more popular than baseball is because",
    "Cricket is most popular in india because",
    "What is the capital of India?",
    "How many teeth do humans have?",
    "Write a short story about a robot who dreams.",
]

model = GPT2Model.from_pretrained("gpt2")  # gpt2 gpt2-medium, gpt2-large, gpt2-xl
inferencer = Inferencer(tokenizer=tokenizer, model=model)
for prompt in prompts:
    print(inferencer.generate(prompt=prompt, num_words=20))
    print()
