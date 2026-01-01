import os
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM,BitsAndBytesConfig
import time


def infer(model, tokenizer, prompt):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    start_time = time.perf_counter()
    outputs = pipe(prompt, use_cache=True, max_new_tokens=400, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=tokenizer.eos_token_id,  pad_token_id=pipe.tokenizer.pad_token_id)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Generation took {elapsed_time:.6f} seconds")
    return outputs[0]['generated_text']

if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model_id = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "I was telling her that"
    print(infer(model, tokenizer, prompt))