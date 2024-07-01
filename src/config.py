"""
Configuration management module.

This module provides a Config class for loading and managing application configuration
from both JSON files and environment variables.
"""

import os
import json
from typing import Any, Dict


class Config:
    """
    A class to manage configuration settings for the application.

    This class loads configuration from a JSON file and environment variables,
    providing a unified interface to access these settings.

    Attributes:
        config (Dict[str, Any]): A dictionary containing all configuration settings.
    """

    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the Config object.

        Args:
            config_file (str): Path to the JSON configuration file. Defaults to "config.json".
        """
        self.config: Dict[str, Any] = {}
        self.load_from_file(config_file)
        self.load_from_env()

    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration settings from a JSON file.

        Args:
            config_file (str): Path to the JSON configuration file.
        """
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))

    def load_from_env(self) -> None:
        """
        Load configuration settings from environment variables.

        This method looks for environment variables prefixed with 'APP_' and adds them
        to the configuration, removing the prefix and converting to lowercase.
        """
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                self.config[key[4:].lower()] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        Args:
            key (str): The configuration key to retrieve.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if not found.
        """
        return self.config.get(key, default)

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
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration values.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to associate with the key.
        """
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Allow use of the 'in' operator to check if a key exists in the configuration.

        Args:
            key (str): The configuration key to check.

        Returns:
            bool: True if the key exists in the configuration, False otherwise.
        """
        return key in self.config
