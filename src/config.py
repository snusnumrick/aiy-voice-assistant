"""
Configuration management module.

This module provides a Config class for loading and managing application configuration
from both JSON files and environment variables, using Pydantic V2 for validation and typing.
"""

import os
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    A class to manage configuration settings for the application.

    This class loads configuration from a JSON file and environment variables,
    providing a unified interface to access these settings with type validation.

    Attributes are dynamically set based on the configuration file and environment variables.
    """

    config_file: str = Field(default="config.json")

    model_config = {
        "extra": "allow",
    }

    def __init__(self, **data):
        config_file = data.get('config_file', 'config.json')

        # Load from JSON file
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                data.update(file_config)

        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                data[key[4:].lower()] = value

        super().__init__(**data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            key (str): The configuration key to retrieve.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if not found.
        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration settings.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration values.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to associate with the key.
        """
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """
        Allow use of the 'in' operator to check if a key exists in the configuration.

        Args:
            key (str): The configuration key to check.

        Returns:
            bool: True if the key exists in the configuration, False otherwise.
        """
        return hasattr(self, key)
