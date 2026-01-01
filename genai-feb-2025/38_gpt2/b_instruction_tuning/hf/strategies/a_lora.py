from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto"
)
print(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["wte", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, peft_config)
print(peft_model)
peft_model.print_trainable_parameters()

print(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['gpt2'])