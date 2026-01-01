from pydantic_ai import Agent
from httpx import Client
from dotenv import load_dotenv
import os
import logfire
import time
from geocode_agent import *
from weather_agent import *


def workflow_chain(
    user_prompt: str,
    geo_deps: GeoCodeDeps,
    weather_deps: WeatherDeps,
):
    geo_result = geo_code_agent.run_sync(user_prompt, deps=geo_deps)
    coordinates = geo_result.data
    weather_result = weather_agent.run_sync(
        f"Find the weather at following coordinates: {coordinates.model_dump_json()}",
        deps=weather_deps,
    )
    return weather_result.data


def main(user_prompt: str):
    load_dotenv(override=True)
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
    time.sleep(1)
    logfire.instrument_openai()
    logfire.instrument_httpx()

    client = Client()
    geo_deps = GeoCodeDeps(
        url="https://api.tomtom.com/search/2/geocode",
        client=client,
        geocode_api_key=os.getenv("GEOCODE_API_KEY"),
    )
    weather_deps = WeatherDeps(
        url="https://api.tomorrow.io/v4/weather/realtime",
        client=client,
        weather_api_key=os.getenv("WEATHER_API_KEY"),
    )

    result = workflow_chain(user_prompt, geo_deps, weather_deps)
    return result


if __name__ == "__main__":
    user_prompt = "Find Hyderbad's current weather"
    response = main(user_prompt)
    print(response)
