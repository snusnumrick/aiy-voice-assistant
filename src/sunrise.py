import requests
import aiohttp
import asyncio
from datetime import datetime
from typing import Optional, Dict, Union, Any, Awaitable
from urllib.parse import urljoin

def get_solar_data(
        latitude: float,
        longitude: float,
        date: Optional[str] = None,
        timezone: Optional[str] = None,
        time_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get solar data (sunrise, sunset, etc.) for a specific location and date.

    Args:
        latitude (float): Latitude of the location (-90 to 90)
        longitude (float): Longitude of the location (-180 to 180)
        date (str, optional): Date in YYYY-MM-DD format, or 'today'/'tomorrow'. Defaults to today.
        timezone (str, optional): Timezone for the returned times (e.g., 'UTC').
            Defaults to location's timezone.
        time_format (str, optional): Format for time display. Options: '24', 'military', 'unix'.
            Defaults to 12-hour format.

    Returns:
        dict: Dictionary containing solar data including:
            - date
            - sunrise
            - sunset
            - first_light
            - last_light
            - dawn
            - dusk
            - solar_noon
            - golden_hour
            - day_length
            - timezone
            - utc_offset

    Raises:
        ValueError: If latitude or longitude are out of valid ranges
        requests.RequestException: If the API request fails
    """
    # Validate latitude and longitude
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # Build API URL
    base_url = "https://api.sunrisesunset.io/json"
    params = {
        "lat": latitude,
        "lng": longitude
    }

    # Add optional parameters if provided
    if date:
        params["date"] = date
    if timezone:
        params["timezone"] = timezone
    if time_format:
        if time_format not in ["24", "military", "unix"]:
            raise ValueError("time_format must be one of: '24', 'military', 'unix'")
        params["time_format"] = time_format

    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes

        data = response.json()

        # Check if the API returned an error
        if data.get("status") != "OK":
            raise requests.RequestException(f"API returned status: {data.get('status')}")

        return data["results"]

    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch solar data: {str(e)}")

async def get_solar_data_async(
        latitude: float,
        longitude: float,
        date: Optional[str] = None,
        timezone: Optional[str] = None,
        time_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async version of get_solar_data using aiohttp.

    Args:
        latitude (float): Latitude of the location (-90 to 90)
        longitude (float): Longitude of the location (-180 to 180)
        date (str, optional): Date in YYYY-MM-DD format, or 'today'/'tomorrow'. Defaults to today.
        timezone (str, optional): Timezone for the returned times (e.g., 'UTC').
            Defaults to location's timezone.
        time_format (str, optional): Format for time display. Options: '24', 'military', 'unix'.
            Defaults to 12-hour format.

    Returns:
        dict: Dictionary containing solar data including:
            - date
            - sunrise
            - sunset
            - first_light
            - last_light
            - dawn
            - dusk
            - solar_noon
            - golden_hour
            - day_length
            - timezone
            - utc_offset

    Raises:
        ValueError: If latitude or longitude are out of valid ranges
        aiohttp.ClientError: If the API request fails
    """
    # Validate latitude and longitude
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # Build API URL
    base_url = "https://api.sunrisesunset.io/json"
    params = {
        "lat": latitude,
        "lng": longitude
    }

    # Add optional parameters if provided
    if date:
        params["date"] = date
    if timezone:
        params["timezone"] = timezone
    if time_format:
        if time_format not in ["24", "military", "unix"]:
            raise ValueError("time_format must be one of: '24', 'military', 'unix'")
        params["time_format"] = time_format

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                response.raise_for_status()  # Raise exception for bad status codes
                data = await response.json()

                # Check if the API returned an error
                if data.get("status") != "OK":
                    raise aiohttp.ClientError(f"API returned status: {data.get('status')}")

                return data["results"]

    except aiohttp.ClientError as e:
        raise aiohttp.ClientError(f"Failed to fetch solar data: {str(e)}")

# Example usage:
if __name__ == "__main__":
    async def main():
        try:
            # Example for New York City
            nyc_data = await get_solar_data_async(
                latitude=40.71427,
                longitude=-74.00597,
                time_format="24"  # Optional: get 24-hour format times
            )

            print("Solar data for New York City:")
            for key, value in nyc_data.items():
                print(f"{key}: {value}")

        except (ValueError, aiohttp.ClientError) as e:
            print(f"Error: {str(e)}")

    asyncio.run(main())
    try:
        # Example for New York City
        nyc_data = get_solar_data(
            latitude=40.71427,
            longitude=-74.00597,
            time_format="24"  # Optional: get 24-hour format times
        )

        print("Solar data for New York City:")
        for key, value in nyc_data.items():
            print(f"{key}: {value}")

    except (ValueError, requests.RequestException) as e:
        print(f"Error: {str(e)}")
