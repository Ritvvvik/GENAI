from transformers import AutoTokenizer

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "how are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]


# batch encode plus
inputs = tokenizer.batch_encode_plus(
    texts, padding="max_length", truncation=True, max_length=10
)
print(inputs)


# preferred method(it decides automatically to invoke encode or batch_encode_plus)
inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=10)
print(inputs)


for input in inputs["input_ids"]:
    print(input)
    output = tokenizer.decode(input, skip_special_tokens=True)
    print(output)
