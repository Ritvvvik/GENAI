from transformers import AutoTokenizer

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(tokenizer)

texts = [
    "how are you?",
    "heoll are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]

for text in texts:
    print("------------------------------")
    inputs = tokenizer.encode(text)  # convert text to tokens + tokens to ids
    print(inputs)
    output = tokenizer.decode(inputs)  # convert ids to tokens + grouping tokens
    print(output)
    print("------------------------------")

# for text in texts:
#     print("------------------------------")
#     tokens = tokenizer.tokenize(text)
#     print(tokens)
#     ids = tokenizer.convert_tokens_to_ids(tokens)
#     print(ids)
#     decoded_tokens = tokenizer.convert_ids_to_tokens(ids)
#     print(decoded_tokens)
#     print("------------------------------")
