from datasets import load_from_disk
import os
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    setup_chat_format,
    SFTTrainer,
    SFTConfig,
)

# load data
src_dir = "F:/nn/fine-tuning/translation"
dataset = load_from_disk(os.path.join(src_dir, "preprocessed_ds"))

# load tokenizer and model
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
print(model.get_memory_footprint() / 1e6)

# required if tokenizer doesnot have default chat template
print(tokenizer.chat_template)
model, tokenizer = setup_chat_format(model, tokenizer)
print(tokenizer.chat_template)


# Training arguments
output_dir = os.path.join(src_dir, "gpt3_it_full")
log_dir = os.path.join(src_dir, "logs")
training_args = SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    #save_steps=5,
    eval_strategy="epoch",
    #eval_steps=1,
    #dataset_text_field="messages",
    # logging_strategy="epoch",
    # logging_steps = 1,
    # logging_dir=log_dir,
    # report_to="tensorboard",
    # push_to_hub=True
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print(batch["input_ids"][0], batch["labels"][0])

# Train the model
trainer.train(resume_from_checkpoint=True)

# push the model to hub
login(token=os.getenv("HUGGING_FACE_TOKEN"), add_to_git_credential=True)
#trainer.push_to_hub()