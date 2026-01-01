from datasets import load_from_disk
import os
from huggingface_hub import login
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)

# Authenticate your token for model download
login(token=os.getenv("HUGGING_FACE_TOKEN"), add_to_git_credential=True)
src_dir = "F:/nn/pretraining/bookcorpus2"

# load datasets
dataset = load_from_disk(os.path.join(src_dir, "chunked_ds"))
print(dataset)

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# Create model
config = GPT2Config()
model = GPT2LMHeadModel(config)
print(model)

# Training arguments
output_dir = os.path.join(src_dir, "gpt2_hf")
log_dir = os.path.join(src_dir, "logs")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    save_steps=1,
    eval_strategy="epoch",
    eval_steps=1,
    #logging_strategy="epoch",
    #logging_steps = 1,
    #logging_dir=log_dir,
    #report_to="tensorboard",
    #push_to_hub=True
)

# Initialize Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

# Train the model
trainer.train()


