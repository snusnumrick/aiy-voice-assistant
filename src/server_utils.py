"""Utility functions for server communication"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_server_url(hostname: str = "localhost") -> str:
    """
    Get the server URL with port read from /tmp/music_server_port.txt

    Args:
        hostname: The hostname to use (default: "localhost")

    Returns:
        Server URL in format "http://{hostname}:{port}"
        Falls back to port 5001 if the port file doesn't exist or can't be read
    """
    port_file = Path("/tmp/music_server_port.txt")
    default_port = 5000

    try:
        if port_file.exists():
            port = port_file.read_text().strip()
            if port.isdigit():
                logger.debug(f"Read server port {port} from {port_file}")
                return f"http://{hostname}:{port}"
            else:
                logger.warning(f"Invalid port in {port_file}: {port}, using default {default_port}")
        else:
            logger.warning(f"Port file {port_file} not found, using default port {default_port}")
    except Exception as e:
        logger.warning(f"Failed to read port from {port_file}: {e}, using default port {default_port}")

    return f"http://{hostname}:{default_port}"
