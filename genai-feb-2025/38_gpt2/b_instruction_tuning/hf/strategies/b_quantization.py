from collections import Counter
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def get_parm_dtypes(iterable, top_k=3):
    return Counter([p.dtype for p in iterable]).most_common(top_k)

# unquantized model
model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto"
)
print(model)
print(model.get_memory_footprint() / 1e6)
print(get_parm_dtypes(model.parameters()))

#quantized model(8-bits)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.bfloat16,
)
model_q8 = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
print(model_q8)
print(model_q8.get_memory_footprint() / 1e6)
print(get_parm_dtypes(model_q8.parameters()))

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_q4 = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
print(model_q4)
print(model_q4.get_memory_footprint() / 1e6)
print(get_parm_dtypes(model_q4.parameters()))