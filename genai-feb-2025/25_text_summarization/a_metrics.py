from bert_score import score
from rouge import Rouge

reference = "it is not cold"
response = "it is freezing"
p, r, f = score([response], [reference], lang="en")
print(p.item(), r.item(), f.item())

scores = Rouge().get_scores(response, reference)
rouge_1 = scores[0]["rouge-1"]
print(rouge_1["p"], rouge_1["r"], rouge_1["f"])

rouge_2 = scores[0]["rouge-2"]
print(rouge_2["p"], rouge_2["r"], rouge_2["f"])

rouge_l = scores[0]["rouge-l"]
print(rouge_l["p"], rouge_l["r"], rouge_l["f"])
