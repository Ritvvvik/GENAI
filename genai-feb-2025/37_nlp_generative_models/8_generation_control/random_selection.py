import torch

token_probs = torch.tensor([0.1, 0.1, 0.3, 0.5])
for _ in range(50):
    max_Y = torch.argmax(token_probs, 0)
    print(max_Y)

cnt = [0, 0, 0, 0]
for _ in range(5000):
    sampled_Y = torch.multinomial(token_probs, 1)
    print(sampled_Y[0])
    cnt[sampled_Y[0]] += 1

print(cnt)
