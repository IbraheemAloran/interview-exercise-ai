"""
Centralized logging configuration for all services.
All logs are written to a single log.txt file and displayed on console.
"""
import logging
import os
from pathlib import Path

# Define log file path in the workspace root
LOG_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = LOG_DIR / "log.txt"

# Create a custom logger that writes to both console and file
def get_logger(name):
    """
    Get a logger instance that writes to both console and log.txt.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been set yet (prevent duplicates)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format for all handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler - writes to single log.txt
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
