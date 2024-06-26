"""
This module provides a simple configuration class that can load configuration
"""
import os
import json
from typing import Any, Dict


class Config:
    def __init__(self, config_file: str = "config.json"):
        self.config: Dict[str, Any] = {}
        self.load_from_file(config_file)
        self.load_from_env()

    def load_from_file(self, config_file: str):
        """Load configuration from a JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))

    def load_from_env(self):
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                self.config[key[4:].lower()] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting of configuration"""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator to check if a key exists"""
        return key in self.config
