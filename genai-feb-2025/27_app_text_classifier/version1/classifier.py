from config_reader import *
from agent import text_classifier_agent
import logfire
import time


class TextClassifier:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_pydantic_ai()
        logfire.instrument_openai()

    def predict(self, query: str) -> str:
        result = text_classifier_agent.run_sync(query)
        return result.data


if __name__ == "__main__":
    classifier = TextClassifier()
    query = "I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    print(classifier.predict(query))
