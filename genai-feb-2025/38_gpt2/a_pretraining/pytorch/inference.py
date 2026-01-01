import os
import torch


class Inferencer:
    def __init__(
        self, output_dir=None, check_point_id=None, model=None, tokenizer=None
    ):
        if output_dir is not None:
            sub_id = "final" if check_point_id is None else str(check_point_id)
            check_point_dir = os.path.join(output_dir, "check_points", sub_id)
            if not os.path.exists(check_point_dir):
                print(f"Checkpoint directory {check_point_dir} does not exist.")
                return
            model = torch.load(
                os.path.join(check_point_dir, "model.pkl"), weights_only=False
            )
            tokenizer = torch.load(
                os.path.join(check_point_dir, "tokenizer.pkl"), weights_only=False
            )
        elif model is None or tokenizer is None:
            print("model/tokenizer is missing")
            return

        self.context_length = model.config["context_length"]
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)

    def generate(self, prompt, num_words=50):
        self.model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        input = inputs["input_ids"].to(self.device)

        self.model.to(self.device)
        response = []
        for _ in range(num_words):
            input = input[:, -self.context_length :]
            # print(input)
            with torch.no_grad():
                logits = self.model(input)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            id_next = torch.argmax(probs, dim=-1, keepdim=True)
            input = torch.cat((input, id_next), dim=1)
            # print(id_next)
            next_word = self.tokenizer.decode(
                id_next.squeeze(), skip_special_tokens=True
            )
            response.append(next_word)

        response = " ".join(response)
        return response.replace("\n", " ")

    def generate_creative(self, prompt, num_words=50, temperature=1.0, top_k=None):
        self.model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        input = inputs["input_ids"].to(self.device)

        self.model.to(self.device)
        response = []
        for _ in range(num_words):
            input = input[:, -self.context_length :]
            # print(input)
            with torch.no_grad():
                logits = self.model(input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, id_next), dim=1)
            # print(id_next)
            next_word = self.tokenizer.decode(
                id_next.squeeze(0), skip_special_tokens=True
            )
            response.append(next_word)

        response = " ".join(response)
        return response.replace("\n", " ")

if __name__ == "__main__":
    src_dir = "F:/nn/pretraining/bookcorpus2"
    output_dir = os.path.join(src_dir, "gpt2")
    prompt = "I was telling her that"
    inferencer = Inferencer(output_dir=output_dir, check_point_id=5)
    print(inferencer.generate_creative(prompt=prompt))
