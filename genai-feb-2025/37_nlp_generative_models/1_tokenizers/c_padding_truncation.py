from transformers import AutoTokenizer

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer)

texts = [
    "how are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]
# padding true
for text in texts:
    input = tokenizer.encode(text, padding=True, truncation=True, max_length=10)
    print(input)
    output = tokenizer.decode(input)
    print(output)

# padding to max length
for text in texts:
    input = tokenizer.encode(text, padding="max_length", truncation=True, max_length=10)
    print(input)
    output = tokenizer.decode(input)
    print(output)

# skip special tokens
for text in texts:
    input = tokenizer.encode(text, padding="max_length", truncation=True, max_length=10)
    print(input)
    output = tokenizer.decode(input, skip_special_tokens=True)
    print(output)

# tensor return
for text in texts:
    input = tokenizer.encode(
        text,
        padding="max_length",
        truncation=True,
        max_length=10,
        return_tensors="pt",
    )
    print(input)
