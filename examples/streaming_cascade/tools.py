# SPDX-License-Identifier: Apache-2.0
"""Example tool functions for the streaming cascade tool profile."""

from collections.abc import Mapping
from urllib.parse import quote

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WIKIPEDIA_SEARCH_URL = "https://{language}.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = (
    "https://{language}.wikipedia.org/api/rest_v1/page/summary/{title}"
)
# Wikimedia asks API clients to identify themselves and answers an anonymous one with 403.
WIKIPEDIA_HEADERS = {
    "User-Agent": "sglang-omni-example/1.0 (https://github.com/sgl-project/sglang-omni)"
}
REQUEST_TIMEOUT_S = 10.0
# The WMO weather interpretation codes Open-Meteo reports, as words the model can speak.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(arguments: Mapping[str, object]) -> dict[str, object]:
    """The current weather of the city in the arguments, from the Open-Meteo public API; an unknown city or a failed request raises, which the tool runner answers to the model as an error."""
    city = arguments.get("city")
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    else:
        pass
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        places = client.get(GEOCODING_URL, params={"name": city.strip(), "count": 1})
        places.raise_for_status()
        found = places.json().get("results") or []
        if not found:
            raise ValueError(f"no place called {city!r} was found")
        else:
            place = found[0]
        forecast = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
            },
        )
        forecast.raise_for_status()
        current = forecast.json()["current"]
    return {
        "city": place["name"],
        "country": place.get("country"),
        "temperature_c": current["temperature_2m"],
        "weather": WEATHER_CODES.get(current["weather_code"], "unknown"),
        "wind_speed_kmh": current["wind_speed_10m"],
    }


async def search_wikipedia(arguments: Mapping[str, object]) -> dict[str, object]:
    """The title and opening summary of the Wikipedia article matching the query in the arguments, from the language edition the arguments name (en unless given); no match or a failed request raises, which the tool runner answers to the model as an error."""
    query = arguments.get("query")
    language = arguments.get("language", "en")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    elif not isinstance(language, str) or not language.replace("-", "").isalpha():
        raise ValueError("language must be a Wikipedia language code such as en")
    else:
        pass
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_S, headers=WIKIPEDIA_HEADERS
    ) as client:
        search = await client.get(
            WIKIPEDIA_SEARCH_URL.format(language=language),
            params={
                "action": "query",
                "list": "search",
                "srsearch": query.strip(),
                "srlimit": 1,
                "format": "json",
            },
        )
        search.raise_for_status()
        matches = search.json()["query"]["search"]
        if not matches:
            raise ValueError(f"no article matches {query!r}")
        else:
            title = matches[0]["title"]
        summary = await client.get(
            WIKIPEDIA_SUMMARY_URL.format(language=language, title=quote(title))
        )
        summary.raise_for_status()
    return {"title": title, "summary": summary.json()["extract"]}
