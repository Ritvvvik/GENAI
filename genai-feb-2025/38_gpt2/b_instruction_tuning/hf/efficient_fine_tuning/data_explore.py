from datasets import load_from_disk, load_dataset
import os

if __name__ == "__main__":    
    src_dir = "F:/nn/fine-tuning/translation"

    print(">>> DataLoading Begin...")
    dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
    print(dataset[0:5])
   