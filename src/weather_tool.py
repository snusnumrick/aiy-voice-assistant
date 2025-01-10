import os
from typing import Dict
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.web_search import WebSearcher
import logging
import aiohttp
import json

logger = logging.getLogger(__name__)


def _format_weather_response(weather_data: Dict, timeframe: str) -> str:
    """Format the weather data response based on timeframe"""
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }

    def _format(data) -> list[str]:
        return [
            f"Temperature: {data['temperature']}°C",
            f"Temperature Apparent: {data['temperatureApparent']}°C",
            f"Visibility: {data['visibility']} km",
            f"Humidity: {data['humidity']}%",
            f"Wind Speed: {data['windSpeed']} m/s",
            f"Wind Gust: {data['windGust']} m/s",
            f"Wind Direction: {data['windDirection']} degrees clockwise from due north",
            f"Dew Point: {data['dewPoint']}C",
            f"Precipitation: {data['precipitationProbability']}%",
            f"Pressure: {data['pressureSurfaceLevel']} hPa",
            f"Description: {code_lookup.get(data['weatherCode'], 'Unknown')}"
        ]

    def _format_hourly(data) -> list[str]:
        return [
            f"{data['temperature']}C",
            f"{data['windSpeed']} m/s",
            f"{data['precipitationProbability']}%",
            f"{data['pressureSurfaceLevel']} hPa",
            f"{code_lookup.get(data['weatherCode'], 'Unknown')}"
        ]


    def _format_daily(data) -> list[str]:
        wc_min = code_lookup.get(data['weatherCodeMin'], 'Unknown')
        wc_max = code_lookup.get(data['weatherCodeMax'], 'Unknown')
        wc = wc_min if wc_min == wc_max else f"{wc_min} / {wc_max}"
        return [
            f"{data['temperatureMin']} - {data['temperatureMax']}°C",
            f"({wc})"
        ]

    if timeframe == "current":
        current = weather_data["data"]["values"]
        result = "Current weather:\n" + "\n".join(_format(current))
    elif timeframe == "hourly":
        hours = weather_data["timelines"]["hourly"][:24]
        result = "Hourly forecast (time: temperature, wind speed, precipitation, pressure, description)\n"
        for hour in hours:
            result += (
                f"{hour['time']}: {', '.join(_format_hourly(hour['values']))}\n"
            )
    else:  # daily
        days = weather_data["timelines"]["daily"][:7]
        result = "Daily forecast:\n"
        for day in days:
            result += (
                f"{day['time'][:10]}: "
                f"{', '.join(_format_daily(day['values']))}\n"
            )
    logger.info(f"Weather response: {result}")
    return result


class WeatherTool:
    """
    This class represents a weather tool that uses the Tomorrow.io API to fetch weather data.
    It provides both synchronous and asynchronous methods to get weather information.

    Class: WeatherTool

    Methods:
    - tool_definition(self) -> Tool
        Returns the definition of the tool including parameters for location and optional timeframe.

    - __init__(self, config: Config)
        Initializes the WeatherTool with API configuration.

    - get_weather(self, parameters: Dict[str, any]) -> str
        Performs a synchronous weather data fetch for the given location.

    - get_weather_async(self, parameters: Dict[str, any]) -> str
        Performs an asynchronous weather data fetch for the given location.
    """

    def tool_definition(self) -> Tool:
        return Tool(
            name="weather_info",
            description="Get weather information for a specific location",
            iterative=True,
            parameters=[
                ToolParameter(
                    name="location",
                    type="string",
                    description="Location as 'latitude,longitude' or 'city,country'",
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Optional: 'current', 'hourly', or 'daily' (default: current)",
                )
            ],
            required=["location"],
            processor=self.get_weather_async,
        )

    def __init__(self, config: Config):
        self.weather_api_key = os.environ.get("TOMORROW_API_KEY")
        self.geo_api_key = os.environ.get("GEOCODE_API_KEY")
        self.base_url = "https://api.tomorrow.io/v4/weather"
        self.led_processing_color = config.get(
            "processing_color", (0, 1, 0)
        )  # dark green
        self.led_processing_blink_period_ms = config.get(
            "processing_blink_period_ms", 300
        )
        self.web_searcher = WebSearcher(config)

    def _start_processing(self):
        pass

    def _stop_processing(self):
        pass

    def get_weather(self, parameters: Dict[str, any]) -> str:
        """Synchronous weather data fetch"""
        import requests

        if "location" not in parameters:
            logger.error(f"Missing required parameter location: {parameters}")
            return ""

        try:
            location = parameters["location"]
            timeframe = parameters.get("timeframe", "current")

            self._start_processing()

            url = f"{self.base_url}/{timeframe}"
            params = {
                "apikey": self.weather_api_key,
                "location": location,
                "units": "metric"
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                weather_data = response.json()
                return _format_weather_response(weather_data, timeframe)
            else:
                logger.error(f"API request failed: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return ""
        finally:
            self._stop_processing()

    async def get_lat_lng(self, location_description: str) -> tuple:
        """Get the latitude and longitude of a location.

        Args:
            location_description: A description of a location.
        """
        if ',' in location_description and not any(c.isalpha() for c in location_description):
            # Already in lat,lon format
            lat, lon = map(float, location_description.split(','))
            return float(lat), float(lon)

        if self.geo_api_key is None:
            logger.error('No geocoding API key provided')
            raise ValueError('No geocoding API key provided')

        async with aiohttp.ClientSession() as session:
            geocode_url = "https://geocode.maps.co/search"
            params = {
                'q': location_description,
                'api_key': self.geo_api_key,
            }
            async with session.get(geocode_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data[0]['lat']), float(data[0]['lon'])
                else:
                    logger.error(f"Could not geocode location: {location_description}")
                    raise ValueError(f"Could not geocode location: {location_description}")

    async def get_weather_async(self, parameters: Dict[str, any]) -> str:
        """Asynchronous weather data fetch"""
        if "location" not in parameters:
            raise ValueError(f"Missing required parameter location: {parameters}")

        timeframe = parameters.get("timeframe", "current")
        try:
            lat, lon = await self.get_lat_lng(parameters["location"])
            # lat, lon = await self._get_coordinates(parameters["location"])
            location = f"{lat},{lon}"
            # location = parameters["location"]

            self._start_processing()

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}"
                if timeframe == "current":
                    url += "/realtime"
                else:
                    url += "/forecast"
                params = {
                    "apikey": self.weather_api_key,
                    "location": location,
                    "units": "metric"
                }
                if timeframe == "daily":
                    params["timesteps"] = "1d"
                elif timeframe == "hourly":
                    params["timesteps"] = "1h"

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        weather_data = await response.json()
                        return _format_weather_response(weather_data, timeframe)
                    else:
                        raise ValueError(f"API request failed: {response.status} ({response.reason})")

        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            if timeframe == "current":
                query = f"Current weather for {parameters['location']}"
            else:
                query = f"{timeframe} weather forecast for {parameters['location']}"
            logger.warning(f"Fallback to web search: {query}")
            return await self.web_searcher.search_async(query)
        finally:
            self._stop_processing()


# Test code
if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass
    from dotenv import load_dotenv

    load_dotenv()

    # Mock Config class for testing
    @dataclass
    class MockConfig(Config):
        def get(self, key, default=None):
            config_values = {
                "processing_color": (0, 1, 0),
                "processing_blink_period_ms": 300
            }
            return config_values.get(key, default)

    async def test_weather_tool():
        # Initialize the tool
        weather_tool = WeatherTool(MockConfig())

        # Test cases
        test_locations = [
            {"location": "40.7128,-74.0060", "timeframe": "current"},  # NYC coordinates
            {"location": "London,UK", "timeframe": "hourly"},          # City name
            {"location": "Tokyo,Japan", "timeframe": "daily"}          # City name with daily forecast
        ]

        print("Testing WeatherTool...")
        print("-" * 50)

        for params in test_locations:
            try:
                print(f"\nFetching weather for {params['location']} ({params['timeframe']}):")
                result = await weather_tool.get_weather_async(params)
                print(result)
            except Exception as e:
                print(f"Error: {str(e)}")
            print("-" * 50)

    # Run the test
    print("Starting Weather Tool Test...")
    asyncio.run(test_weather_tool())