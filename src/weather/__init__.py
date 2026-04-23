"""Weather package exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "EnhancedWeatherTool",
    "Moon",
    "UVIndexError",
    "WeatherTool",
    "get_air_quality",
    "get_air_quality_async",
    "get_solar_data",
    "get_solar_data_async",
    "get_uv_index",
    "get_uv_index_async",
]

_EXPORTS = {
    "EnhancedWeatherTool": (".tool", "EnhancedWeatherTool"),
    "Moon": (".moon", "Moon"),
    "UVIndexError": (".openuv", "UVIndexError"),
    "WeatherTool": (".tool", "WeatherTool"),
    "get_air_quality": (".aqi", "get_air_quality"),
    "get_air_quality_async": (".aqi", "get_air_quality_async"),
    "get_solar_data": (".sunrise", "get_solar_data"),
    "get_solar_data_async": (".sunrise", "get_solar_data_async"),
    "get_uv_index": (".openuv", "get_uv_index"),
    "get_uv_index_async": (".openuv", "get_uv_index_async"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
