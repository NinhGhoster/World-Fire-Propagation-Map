"""
World Fire Propagation Map - Logging Configuration

Replaces print statements with proper Python logging.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import Config


def setup_logging(name: str = None, level: str = None, log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, (level or Config.LOG_LEVEL).upper(), logging.INFO)
    
    logger = logging.getLogger(name or "fire_propagation")
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Format
    formatter = logging.Formatter(Config.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file or Config.LOG_LEVEL != "DEBUG":
        log_dir = Path(Config.LOGS_DIR)
        log_dir.mkdir(exist_ok=True)
        
        log_filename = log_file or f"fire_map_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the standard configuration."""
    return setup_logging(name)


# Convenience functions for common logging patterns
def log_info(logger, message: str):
    """Log info message."""
    logger.info(message)


def log_error(logger, message: str, exc_info: bool = True):
    """Log error message with optional exception info."""
    logger.error(message, exc_info=exc_info)


def log_warning(logger, message: str):
    """Log warning message."""
    logger.warning(message)


def log_debug(logger, message: str):
    """Log debug message."""
    logger.debug(message)


def log_exception(logger, message: str, exc_info: bool = True):
    """Log exception with message."""
    logger.exception(message, exc_info=exc_info)
