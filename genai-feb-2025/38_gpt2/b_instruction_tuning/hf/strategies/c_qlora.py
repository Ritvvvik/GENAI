import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def trainable_parms(model):
    parms = [(name, param.dtype) for name, param in model.named_parameters() if param.requires_grad]
    return parms

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
print(model)
print(trainable_parms(model))

# it freezes all layers
# it casts every non-quantized 16-bit layer to FP32 to improve training
# it enables gradient checkpointing
# unfreeze layers of your choice later on using the LoRA configuration
model = prepare_model_for_kbit_training(model)
print(trainable_parms(model))


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
print(trainable_parms(peft_model))
