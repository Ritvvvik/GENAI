# pip install logfire
# 10M events per month limit on its free tier
# monitoring->explore: select * from records
# debugging->live: events
import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

logfire.info("Hello, {name}!", name="Algorithmica")
