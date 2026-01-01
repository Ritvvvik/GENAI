import re
import os


def build_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item]
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    print(vocab)
    return vocab


class SimpleTokenizerV1:
    def __init__(self, src_dir, file):
        vocab = build_vocab(os.path.join(src_dir, file))
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


if __name__ == "__main__":
    tokenizer = SimpleTokenizerV1(
        "C:/Users/pc/Documents/nn",
        "sample.txt",
    )

    text = "the last he painted, you know, Mrs. Gisburn said with pardonable pride."
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
