import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
 


def infer(model, tokenizer, prompt):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = pipe.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=tokenizer.eos_token_id,  pad_token_id=pipe.tokenizer.pad_token_id)
    return outputs[0]['generated_text']

if __name__ == "__main__":

    src_dir = "F:/nn/fine-tuning/translation/gpt2_it_qlora"
    model_id = os.path.join(src_dir, "checkpoint-810")

    # Load Model with/without PEFT adapter
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     torch_dtype=torch.float16
    # )
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #prompt = [{"role": "user", "content":'The Force is strong in you!'}]
    prompt = [{"role": "user", "content":'The birch canoe slid on the smooth planks.'}]
    print(infer(model, tokenizer, prompt))