import os
import torch
from peft import AutoPeftModelForCausalLM

src_dir = "F:/nn/fine-tuning/translation"
peft_model_id = os.path.join(src_dir, "gpt2_it_lora/checkpoint-810")
output_dir = os.path.join(src_dir, "gpt2_it_lora_merged_model")

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir,safe_serialization=True)