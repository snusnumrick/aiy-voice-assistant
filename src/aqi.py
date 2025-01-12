import requests
import aiohttp
import asyncio
from typing import Dict, Union, Optional, Awaitable

def get_air_quality(latitude: float, longitude: float, token: str) -> Dict[str, Union[str, dict]]:
    """
    Get air quality data for a specific latitude and longitude using the WAQI API.

    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        token (str): Your WAQI API token

    Returns:
        dict: Dictionary containing the air quality data with the following structure:
            - status (str): Status of the request ('ok' or 'error')
            - data (dict): Air quality data if status is 'ok', containing:
                - aqi (int): Air Quality Index
                - station (dict): Information about the nearest monitoring station
                - time (dict): Timestamp of the measurement

    Raises:
        requests.RequestException: If there's an error with the API request
        ValueError: If the coordinates are invalid
    """
    # Input validation
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise ValueError("Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180")

    # Construct the API URL
    base_url = "https://api.waqi.info"
    endpoint = f"/feed/geo:{latitude};{longitude}/"
    url = f"{base_url}{endpoint}"

    # Add token as parameter
    params = {"token": token}

    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse the response
        data = response.json()

        if data["status"] == "error":
            return {
                "status": "error",
                "message": data.get("message", "Unknown error occurred")
            }

        return data

    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"API request failed: {str(e)}"
        }

async def get_air_quality_async(latitude: float, longitude: float, token: str) -> Dict[str, Union[str, dict]]:
    """
    Async version of get_air_quality using aiohttp.

    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        token (str): Your WAQI API token

    Returns:
        dict: Dictionary containing the air quality data with the same structure as get_air_quality()

    Raises:
        aiohttp.ClientError: If there's an error with the API request
        ValueError: If the coordinates are invalid
    """
    # Input validation
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise ValueError("Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180")

    # Construct the API URL
    base_url = "https://api.waqi.info"
    endpoint = f"/feed/geo:{latitude};{longitude}/"
    url = f"{base_url}{endpoint}"

    # Add token as parameter
    params = {"token": token}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if data["status"] == "error":
                    return {
                        "status": "error",
                        "message": data.get("message", "Unknown error occurred")
                    }

                return data

    except aiohttp.ClientError as e:
        return {
            "status": "error",
            "message": f"API request failed: {str(e)}"
        }

# Example usage:
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Replace 'your_token' with your actual WAQI API token
    sample_token = os.getenv("WAQI_TOKEN")
    sample_lat = 40.7128
    sample_lng = -74.0060

    result = get_air_quality(sample_lat, sample_lng, sample_token)

    if result["status"] == "ok":
        print(f"Air Quality Index: {result['data']['aqi']}")
        print(f"Station: {result['data']['city']['name']}")
        print(f"Last Updated: {result['data']['time']['s']}")
    else:
        print(f"Error: {result['message']}")
