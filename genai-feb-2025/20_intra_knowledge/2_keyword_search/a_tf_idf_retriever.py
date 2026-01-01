from langchain_community.retrievers import TFIDFRetriever

retriever = TFIDFRetriever.from_texts(
    [
        "Australia won the Cricket World Cup 2023",
        "India and Australia played in the finals",
        "Australia won the sixth time having last won in 2015",
    ]
)

results = retriever.invoke("world cup winners")
for result in results:
    print(result.page_content)
