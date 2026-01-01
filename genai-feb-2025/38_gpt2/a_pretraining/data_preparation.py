# pip install datasets
# pip install transformers
# pip install accelerate

import os
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token=tokenizer.eos_token

def tokenize_function(example):
    tokens = tokenizer(example["text"])
    return tokens

def concat(examples):    
    examples["input_ids"]=[list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["attention_mask"]=[list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples

def chunk(examples):
    chunk_size = 1024       
    input_ids = examples["input_ids"][0]     
    attention_mask = examples["attention_mask"][0] 
    input_ids_truncated = []
    attention_mask_truncated = []

    for i in range(0,len(input_ids),chunk_size):
        chunk = input_ids[i:i+chunk_size]
        if len(chunk)==chunk_size: # drop the last chunk if not equal
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])     
    examples['input_ids']=input_ids_truncated
    examples["attention_mask"]=attention_mask_truncated
        
    return examples

if __name__ == '__main__':
    src_dir = "F:/nn/pretraining/bookcorpus2"

    print(">>> DataLoading Begin...")
    dataset = (load_dataset("bookcorpus",trust_remote_code=True, split="train")
                .shuffle(seed=1)
                .select(range(20000)))
    print(dataset)
    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset) 
    print(">>> DataLoading End...")

    # tokenize the dataset
    print(">>> Tokenization Begin...")
    tokenized_ds = dataset.map(tokenize_function,batched=True,remove_columns='text', num_proc=16)
    tokenized_ds.save_to_disk(os.path.join(src_dir, 'tokenized_ds'))
    print(">>> Tokenization End...")

    # Make samples to a size of context window
    print(">>> Chunk Dataset Begin...")
    concated_ds = tokenized_ds.map(concat,batched=True,batch_size=10000,num_proc=16)
    chunked_ds = concated_ds.map(chunk,batched=True,batch_size=2,num_proc=16)
    chunked_ds.save_to_disk(os.path.join(src_dir, 'chunked_ds')) 
    print(">>> Chunk Dataset End...")
