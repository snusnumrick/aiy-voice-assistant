"""
Configuration management module.

This module provides a Config class for loading and managing application configuration
from both shared and user-specific JSON files and environment variables, using Pydantic for validation.
"""

import os
import json
from typing import Any, Dict
from pydantic import BaseModel


class Config(BaseModel):
    """
    A class to manage configuration settings for the application.

    This class loads configuration from multiple sources in the following order of precedence:
    1. Direct arguments passed to constructor
    2. Environment variables (prefixed with APP_)
    3. user.json (user-specific overrides)
    4. config.json (shared configuration)
    """

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(
        self,
        config_file: str = "config.json",
        user_config_file: str = "user.json",
        **data,
    ):
        # First collect all configuration data
        init_data = {}

        # Load from shared config file (lowest precedence)
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                    init_data.update(file_config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing config file {config_file}: {str(e)}")

        # Load from user config file (overrides shared config)
        if os.path.exists(user_config_file):
            try:
                with open(user_config_file, "r") as f:
                    user_config = json.load(f)
                    init_data.update(user_config)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Error parsing user config file {user_config_file}: {str(e)}"
                )

        # Load from environment variables (overrides both config files)
        for key, value in os.environ.items():
            if key.startswith("APP_"):
                try:
                    # Try to parse as JSON for complex types
                    parsed_value = json.loads(value)
                    init_data[key[4:].lower()] = parsed_value
                except json.JSONDecodeError:
                    # If not JSON, use the string value
                    init_data[key[4:].lower()] = value

        # Update with any direct arguments (highest precedence)
        init_data.update(data)

        # Initialize the Pydantic model with our collected data
        super().__init__(**init_data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            key (str): The configuration key to retrieve.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if not found.
        """
        return self.__dict__.get(key, default)

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
            return self.__dict__[key]
        except KeyError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration values.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to associate with the key.
        """
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Allow use of the 'in' operator to check if a key exists in the configuration.

        Args:
            key (str): The configuration key to check.

        Returns:
            bool: True if the key exists in the configuration, False otherwise.
        """
        return key in self.__dict__

    def dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the configuration.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration values.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __str__(self) -> str:
        """
        Provide a string representation of the configuration.

        Returns:
            str: A string representation of the configuration data.
        """
        return f"Config({self.dict()})"
