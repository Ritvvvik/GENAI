import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

logfire.notice("Hello, {name}!", name="Algorithmica")
logfire.info("Hello, {name}!", name="Algorithmica")
logfire.debug("Hello, {name}!", name="Algorithmica")
logfire.warn("Hello, {name}!", name="Algorithmica")
logfire.error("Hello, {name}!", name="Algorithmica")
logfire.fatal("Hello, {name}!", name="Algorithmica")
