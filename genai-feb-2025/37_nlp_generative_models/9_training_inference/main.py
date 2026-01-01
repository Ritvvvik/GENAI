import os
from dataloader import *
from models import (
    TextGenerationModel3,
    TextGenerationModel4,
    TextGenerationModel5,
    NanoGPT,
)
from trainer import *
from inference import *
from transformers import AutoTokenizer

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens(
    {
        "pad_token": tokenizer.eos_token,
    }
)

# model configuration
# config = {
#     "vocab_size": len(tokenizer),
#     "context_length": 64,
#     "emb_dim": 128,
#     "hidden_dim": 64,
#     "n_heads": 4,
#     "drop_out": 0.25,
# }
# model = TextGenerationModel3(config)
# model = TextGenerationModel4(config)
# model = TextGenerationModel5(config)
config = {
    "vocab_size": len(tokenizer),
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.25,
}
model = NanoGPT(config)

# train configuration
epochs = 10
batch_size = 8
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
src_dir = "C:/Users/pc/Documents/nn/pretraining/books"
train_dir = os.path.join(src_dir, "train")
train_dataset = CustomDataset(
    tokenizer=tokenizer, data_dir=train_dir, context_length=config["context_length"]
)
val_dir = os.path.join(src_dir, "val")
val_dataset = CustomDataset(
    tokenizer=tokenizer, data_dir=val_dir, context_length=config["context_length"]
)
output_dir = os.path.join(src_dir, "nano_gpt")

# train the model
trainer = Trainer(model, tokenizer)
trainer.train(
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs,
    batch_size=batch_size,
    train_shuffle=True,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    val_shuffle=False,
    eval_freq=1,
    output_dir=output_dir,
    check_point_id=0,
    check_point_freq=5,
)


# Inference with trained model
prompt = "I was telling her that"
inferencer = Inferencer(output_dir=output_dir)
print(inferencer.generate(prompt=prompt))
print()
