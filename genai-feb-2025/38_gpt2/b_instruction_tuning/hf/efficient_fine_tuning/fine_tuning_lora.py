from datasets import load_from_disk
import os
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
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

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto"
)

print(tokenizer.chat_template)
model, tokenizer = setup_chat_format(model, tokenizer)
print(tokenizer.chat_template)


# Training arguments
output_dir = os.path.join(src_dir, "gpt2_it_lora")
log_dir = os.path.join(src_dir, "logs")
training_args = SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    save_steps=5,
    eval_strategy="epoch",
    eval_steps=1,
    # logging_strategy="epoch",
    # logging_steps = 1,
    # logging_dir=log_dir,
    # report_to="tensorboard",
    # push_to_hub=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config
)
print(trainer.model.get_memory_footprint() / 1e6)
trainer.model.print_trainable_parameters()

# Train the model
trainer.train(resume_from_checkpoint=True)


# push the model to hub
login(token=os.getenv("HUGGING_FACE_TOKEN"), add_to_git_credential=True)
#trainer.push_to_hub()