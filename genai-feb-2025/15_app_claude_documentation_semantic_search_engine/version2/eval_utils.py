from tqdm import tqdm


def calculate_mrr(retrieved_links: list[str], correct_links) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0


def evaluate_retrieval(retrieved_links: list[str], correct_links: list[str]):
    true_positives = len(set(retrieved_links) & set(correct_links))
    precision = true_positives / len(retrieved_links) if retrieved_links else 0
    recall = true_positives / len(correct_links) if correct_links else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    mrr = calculate_mrr(retrieved_links, correct_links)
    return precision, recall, mrr, f1


def evaluate_retrieval_batch(eval_data, ss):
    precisions = []
    recalls = []
    mrrs = []
    f1s = []
    verdicts = []
    all_correct_links = []
    all_retrieved_links = []
    for i, item in enumerate(tqdm(eval_data, desc="Evaluating Retrieval")):
        correct_links = item["correct_chunks"]
        retrieved_links = ss.retrieve(item["question"])
        precision, recall, mrr, f1 = evaluate_retrieval(retrieved_links, correct_links)
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        f1s.append(f1)
        verdicts.append(True if f1 >= 0.3 else False)
        all_correct_links.append(correct_links)
        all_retrieved_links.append(retrieved_links)
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    avg_f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    return (
        avg_precision,
        avg_recall,
        avg_mrr,
        avg_f1,
        precisions,
        recalls,
        f1s,
        mrrs,
        verdicts,
        all_correct_links,
        all_retrieved_links,
    )
