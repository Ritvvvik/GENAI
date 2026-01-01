from datasets import load_from_disk, load_dataset
import os

def format_dataset(examples):
    if isinstance(examples["prompt"], list):
        output_texts = []
        for i in range(len(examples["prompt"])):
            converted_sample = [
                {"role": "user", "content": examples["prompt"][i]},
                {"role": "assistant", "content": examples["completion"][i]},
            ]
            output_texts.append(converted_sample)
        return {"messages": output_texts}
    else:
        converted_sample = [
            {"role": "user", "content": examples["prompt"]},
            {"role": "assistant", "content": examples["completion"]},
        ]
        return {"messages": converted_sample}
    
if __name__ == "__main__":    
    src_dir = "F:/nn/fine-tuning/translation"

    print(">>> DataLoading Begin...")
    dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
    print(dataset)
    dataset = dataset.train_test_split(0.1)
    print(dataset)
    print(dataset["train"][0])
    print(">>> DataLoading End...")

    print(">>> Preprocessing Begin...")
    dataset = dataset.rename_column("sentence", "prompt")
    dataset = dataset.rename_column("translation_extra", "completion")
    dataset = dataset.remove_columns(["translation"])
    print(dataset)
    print(dataset["train"][0])

    dataset = dataset.map(format_dataset, remove_columns=["prompt", "completion"], num_proc=os.cpu_count())
    print(dataset["train"][0])
    dataset.save_to_disk(os.path.join(src_dir, 'preprocessed_ds'))
    print(">>> Preprocessing End...")

