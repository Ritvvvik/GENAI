import os
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens(
    {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": tokenizer.eos_token,
    }
)

def chat_template(message):
    prompt = ""
    if message.get("system") != None and len(message["system"]) > 0:
        prompt += f"<|im_start|>system\n{message["system"]}<|im_end|>\n"
    if message.get("user") != None and len(message["user"]) > 0:
        prompt += f"<|im_start|>user\n{message["user"]}<|im_end|>\n"
    if message.get("assistant") != None and len(message["assistant"]) > 0:
        prompt += f"<|im_start|>assistant\n{message["assistant"]}<|im_end|>"
    return prompt

def format_item(item):
    chosen_message = {
         "system": item["system"],
         "user": item["question"],
         "assistant": item["chosen"]
    }
    templated_chosen_message = chat_template(chosen_message)

    rej_message = {
         "system": item["system"],
         "user": item["question"],
         "assistant": item["rejected"]
    }
    templated_rej_message = chat_template(rej_message)
    return {"chosen_message": templated_chosen_message,
            "rej_message": templated_rej_message
            }

def tokenize_function(example):
    context_length = 1024
    chosen_encoded = tokenizer(example["chosen_message"],  return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=context_length)
    example["chosen_input_ids"] = chosen_encoded["input_ids"]
    example["chosen_attention_mask"] = chosen_encoded["attention_mask"]

    rej_encoded = tokenizer(example["rej_message"],  return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=context_length)
    example["rej_input_ids"] = rej_encoded["input_ids"]
    example["rej_attention_mask"] = rej_encoded["attention_mask"]
    return example


if __name__ == '__main__':
    src_dir = "F:/nn/preference_tuning/orca"

    print(">>> DataLoading Begin...")
    dataset = (load_dataset("Intel/orca_dpo_pairs",trust_remote_code=True, split="train")
                .shuffle(seed=1)
                .select(range(200)))
    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset["train"][0]) 
    print(">>> DataLoading End...")

    # Format the items according to chat template
    print(">>> Chat Templating Dataset Begin...")
    remove_columns = dataset["train"].column_names
    chat_template_ds = dataset.map(format_item, remove_columns=remove_columns, num_proc=os.cpu_count())
    chat_template_ds.save_to_disk(os.path.join(src_dir, 'templated_ds'))
    print(chat_template_ds["train"][0]) 
    print(">>> Chat Templating Dataset End...")

    # tokenize the dataset
    print(">>> Tokenization Begin...")
    tokenized_ds = chat_template_ds.map(tokenize_function, remove_columns=["chosen_message", "rej_message"], num_proc=os.cpu_count())
    tokenized_ds.save_to_disk(os.path.join(src_dir, 'tokenized_ds'))
    print(tokenized_ds["train"][0])
    print(">>> Tokenization End...")


