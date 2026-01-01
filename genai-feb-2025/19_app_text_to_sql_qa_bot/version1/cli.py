from rich.console import Console
from rag import TextToSqlRAG


def main():
    rag = TextToSqlRAG()
    console = Console()
    console.print(
        "Welcome to TextToSql Bot.  How can i help you.",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_question = input(">>")
        if user_question == "q":
            break
        console.print()
        context = rag.retrieveContext(user_question)
        console.print(context, style="green", end="\n\n")
        answer = rag.execute(user_question)
        console.print(answer, style="cyan", end="\n\n")


if __name__ == "__main__":
    main()

""" 
Sample queries
[
    {
        "question": "How many albums are there?",
        "answer": "347",
    },
    {
        "question": "How many distinct genres are there?",
        "answer": "25",
    },
    {
        "question": "Which Employee has the Highest Total Number of Customers?",
        "answer": "Peacock Jane has the most customers (she has 21 customers)",
    },
    {
        "question": "Who are our top Customers according to Invoices?",
        "answer": "Helena Holy, Richard Cunningham, Luis Rojas, Ladislav Kovacs, and Hugh O\u2019Reilly are the top five customers who have spent the highest amount of money according to the invoice",
    },
    {
        "question": "How many Rock music listeners are there?",
        "answer": "We found out that all 59 customers in the database have listened to Rock Music.",
    },
    {
        "question": "What artists have written most rock music songs?",
        "answer": "Led Zeppelin tops the list of Artists who have written the most Rock Music with 114 songs followed Closely by U2 with 112 music.",
    },
    {
        "question": "Which artist has earned the most according to the Invoice Lines? How much is it?",
        "answer": "The Artist who has earned the most according to the invoice lines is Iron Maiden with a total of $138.6.",
    },
    {
        "question": "How many tracks have a song length greater than the average song length?",
        "answer": "Out of 3503 songs in the database, we found out that 494 of these songs have length more than the average music length of 393,599.21 milliseconds.",
    },
    {
        "question": "What is the most popular genre for Australia?",
        "answer": "Rock is the most popular song for Australia",
    },
] """
