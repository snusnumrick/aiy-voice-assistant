import sys
import types
import unittest
from unittest.mock import patch

stub_ai_models_with_tools = types.ModuleType("src.ai_models_with_tools")


class Tool:
    pass


class ToolParameter:
    pass


stub_ai_models_with_tools.Tool = Tool
stub_ai_models_with_tools.ToolParameter = ToolParameter

stub_web_search = types.ModuleType("src.web_search")


class WebSearcher:
    def __init__(self, config):
        self.config = config


stub_web_search.WebSearcher = WebSearcher


def _load_weather_symbols():
    with patch.dict(
        sys.modules,
        {
            "src.ai_models_with_tools": stub_ai_models_with_tools,
            "src.web_search": stub_web_search,
        },
    ):
        from src.weather import (
            EnhancedWeatherTool as PackagedEnhancedWeatherTool,
        )
        from src.weather import (
            Moon as PackagedMoon,
        )
        from src.weather import (
            UVIndexError as PackagedUVIndexError,
        )
        from src.weather import (
            WeatherTool as PackagedWeatherTool,
        )
        from src.weather import (
            get_air_quality as packaged_get_air_quality,
        )
        from src.weather import (
            get_air_quality_async as packaged_get_air_quality_async,
        )
        from src.weather import (
            get_solar_data as packaged_get_solar_data,
        )
        from src.weather import (
            get_solar_data_async as packaged_get_solar_data_async,
        )
        from src.weather import (
            get_uv_index as packaged_get_uv_index,
        )
        from src.weather import (
            get_uv_index_async as packaged_get_uv_index_async,
        )
        from src.weather.aqi import get_air_quality, get_air_quality_async
        from src.weather.moon import Moon
        from src.weather.openuv import UVIndexError, get_uv_index, get_uv_index_async
        from src.weather.sunrise import get_solar_data, get_solar_data_async
        from src.weather.tool import EnhancedWeatherTool, WeatherTool

    return {
        "EnhancedWeatherTool": EnhancedWeatherTool,
        "Moon": Moon,
        "PackagedEnhancedWeatherTool": PackagedEnhancedWeatherTool,
        "PackagedMoon": PackagedMoon,
        "PackagedUVIndexError": PackagedUVIndexError,
        "PackagedWeatherTool": PackagedWeatherTool,
        "UVIndexError": UVIndexError,
        "WeatherTool": WeatherTool,
        "get_air_quality": get_air_quality,
        "get_air_quality_async": get_air_quality_async,
        "get_solar_data": get_solar_data,
        "get_solar_data_async": get_solar_data_async,
        "get_uv_index": get_uv_index,
        "get_uv_index_async": get_uv_index_async,
        "packaged_get_air_quality": packaged_get_air_quality,
        "packaged_get_air_quality_async": packaged_get_air_quality_async,
        "packaged_get_solar_data": packaged_get_solar_data,
        "packaged_get_solar_data_async": packaged_get_solar_data_async,
        "packaged_get_uv_index": packaged_get_uv_index,
        "packaged_get_uv_index_async": packaged_get_uv_index_async,
    }


class TestWeatherPackageImports(unittest.TestCase):
    def test_tool_exports_match_legacy_imports(self):
        symbols = _load_weather_symbols()

        self.assertIs(symbols["WeatherTool"], symbols["PackagedWeatherTool"])
        self.assertIs(symbols["EnhancedWeatherTool"], symbols["PackagedEnhancedWeatherTool"])

    def test_helper_exports_match_legacy_imports(self):
        symbols = _load_weather_symbols()

        self.assertIs(symbols["get_air_quality"], symbols["packaged_get_air_quality"])
        self.assertIs(symbols["get_air_quality_async"], symbols["packaged_get_air_quality_async"])
        self.assertIs(symbols["get_solar_data"], symbols["packaged_get_solar_data"])
        self.assertIs(symbols["get_solar_data_async"], symbols["packaged_get_solar_data_async"])
        self.assertIs(symbols["get_uv_index"], symbols["packaged_get_uv_index"])
        self.assertIs(symbols["get_uv_index_async"], symbols["packaged_get_uv_index_async"])
        self.assertIs(symbols["Moon"], symbols["PackagedMoon"])
        self.assertIs(symbols["UVIndexError"], symbols["PackagedUVIndexError"])
