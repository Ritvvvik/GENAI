from sklearn.metrics import classification_report, confusion_matrix
from config_reader import settings
import os
import json
from tqdm import tqdm
import pandas as pd
from classifier import TextClassifier
import matplotlib.pyplot as plt
import numpy as np


class ClassificationEvaluator:
    def __init__(self):
        self.src_dir = settings.file_paths.src_dir
        self.eval_file = settings.file_paths.eval_file
        self.eval_file_with_response = settings.file_paths.eval_file_with_response
        self.results_dir = os.path.join(self.src_dir, settings.file_paths.results_dir)
        self.results_report_file = settings.file_paths.results_report_file
        self.results_matrix_file = settings.file_paths.results_matrix_file

    def generate_responses(self):
        classifier = TextClassifier()
        print(">>> Begin: Generating responses for evaluation questions...")
        with open(os.path.join(self.src_dir, self.eval_file), "r") as f:
            eval_data = json.load(f)
        results = []
        for item in tqdm(eval_data, desc="Generating Answers Using Model"):
            response = classifier.predict(item["question"])
            tmp = {
                "id": item["id"],
                "user_input": item["question"],
                "response": response,
                "reference": item["answer"],
            }
            results.append(tmp)

        with open(os.path.join(self.src_dir, self.eval_file_with_response), "w") as f:
            json.dump(results, f, indent=2)
        print(">>> End: Generating responses for evaluation questions...")

    def save_confusion_matrix(self, cm, labels):
        # Visualize the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, cmap="Blues")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set tick labels and positions
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Add labels to each cell
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        # Set labels and title
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, self.results_matrix_file))

    def generate_metrics(self):
        print(">>> Begin: Evaluating metrics...")
        with open(os.path.join(self.src_dir, self.eval_file_with_response), "r") as f:
            eval_data = json.load(f)

        responses = []
        references = []
        for item in eval_data:
            responses.append(item["response"])
            references.append(item["reference"])

        report = classification_report(references, responses, output_dict=True)
        df = pd.DataFrame(report).transpose()
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        df.to_csv(os.path.join(self.results_dir, self.results_report_file))

        labels = sorted(list(set(references)))
        cm = confusion_matrix(references, responses, labels=labels)
        self.save_confusion_matrix(cm, labels)
        print(">>> End: Evaluating metrics...")


if __name__ == "__main__":
    evaluator = ClassificationEvaluator()
    evaluator.generate_responses()
    evaluator.generate_metrics()
