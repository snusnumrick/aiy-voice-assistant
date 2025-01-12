import requests
import aiohttp
import asyncio

class UVIndexError(Exception):
    """Custom exception for UV index API errors"""
    pass

def get_uv_index(lat: float, lng: float, api_key: str) -> dict:
    """
    Fetch UV index data for given latitude and longitude coordinates.

    Args:
        lat (float): Latitude (-90 to 90)
        lng (float): Longitude (-180 to 180)
        api_key (str): OpenUV API key

    Returns:
        dict: UV index data containing:
            - uv: Current UV index
            - uv_time: Timestamp of UV reading
            - uv_max: Maximum UV index for the day
            - uv_max_time: Time of maximum UV
            - ozone: Ozone level in DU
            - ozone_time: Timestamp of ozone reading

    Raises:
        UVIndexError: If API request fails or returns invalid data
        ValueError: If coordinates are out of valid range
    """
    # Validate coordinates
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= lng <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # API endpoint
    url = "https://api.openuv.io/api/v1/uv"

    # Request headers
    headers = {
        "x-access-token": api_key
    }

    # Query parameters
    params = {
        "lat": lat,
        "lng": lng
    }

    try:
        # Make API request
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        # Parse response
        data = response.json()['result']

        # Validate required fields
        required_fields = ["uv", "uv_time", "uv_max", "uv_max_time", "ozone", "ozone_time"]
        if not all(field in data for field in required_fields):
            raise UVIndexError("Missing required fields in API response")

        return data

    except requests.exceptions.RequestException as e:
        raise UVIndexError(f"API request failed: {str(e)}")
    except ValueError as e:
        raise UVIndexError(f"Invalid JSON response: {str(e)}")

async def get_uv_index_async(lat: float, lng: float, api_key: str) -> dict:
    """
    Async version of get_uv_index using aiohttp.

    Args:
        lat (float): Latitude (-90 to 90)
        lng (float): Longitude (-180 to 180)
        api_key (str): OpenUV API key

    Returns:
        dict: UV index data containing:
            - uv: Current UV index
            - uv_time: Timestamp of UV reading
            - uv_max: Maximum UV index for the day
            - uv_max_time: Time of maximum UV
            - ozone: Ozone level in DU
            - ozone_time: Timestamp of ozone reading

    Raises:
        UVIndexError: If API request fails or returns invalid data
        ValueError: If coordinates are out of valid range
    """
    # Validate coordinates
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= lng <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # API endpoint
    url = "https://api.openuv.io/api/v1/uv"

    # Request headers
    headers = {
        "x-access-token": api_key
    }

    # Query parameters
    params = {
        "lat": lat,
        "lng": lng
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Validate required fields
                required_fields = ["uv", "uv_time", "uv_max", "uv_max_time", "ozone", "ozone_time"]
                if not all(field in data['result'] for field in required_fields):
                    raise UVIndexError("Missing required fields in API response")

                return data['result']

    except aiohttp.ClientError as e:
        raise UVIndexError(f"API request failed: {str(e)}")
    except ValueError as e:
        raise UVIndexError(f"Invalid JSON response: {str(e)}")

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    async def main():
        try:
            # Replace with your API key
            API_KEY = os.getenv("OPENUV_API_KEY")

            # Example coordinates (Perth, Australia)
            result = await get_uv_index_async(lat=-31.9523, lng=115.8613, api_key=API_KEY)

            print(f"Current UV Index: {result['uv']}")
            print(f"Max UV Index: {result['uv_max']} at {result['uv_max_time']}")
            print(f"Ozone Level: {result['ozone']} DU")

        except (UVIndexError, ValueError) as e:
            print(f"Error: {str(e)}")

    asyncio.run(main())
