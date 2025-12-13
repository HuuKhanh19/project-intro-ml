"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.BRIGHT_BLUE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BOLD + Colors.BRIGHT_RED,
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str = "chest_xray",
    log_file: str = None,
    level: int = logging.INFO,
    use_colors: bool = True
) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        use_colors: Use colored output in console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors:
        console_format = ColoredFormatter(
            '%(levelname)s | %(message)s'
        )
    else:
        console_format = logging.Formatter(
            '%(levelname)s | %(message)s'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "chest_xray") -> logging.Logger:
    """
    Get existing logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, width: int = 70):
    """
    Log a section separator with title.
    
    Args:
        logger: Logger instance
        title: Section title
        width: Width of separator line
    """
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_dict(logger: logging.Logger, data: dict, indent: int = 2):
    """
    Log dictionary contents in a formatted way.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        indent: Indentation level
    """
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"{' ' * indent}{key}:")
            log_dict(logger, value, indent + 2)
        else:
            logger.info(f"{' ' * indent}{key}: {value}")


def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
    """
    Print colored text to console.
    
    Args:
        text: Text to print
        color: Color code from Colors class
        bold: Use bold text
    """
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.RESET}")


def print_separator(char: str = "=", width: int = 70, color: str = Colors.GREEN):
    """
    Print a separator line.
    
    Args:
        char: Character to use for separator
        width: Width of separator
        color: Color of separator
    """
    print_colored(char * width, color)


def print_header(title: str, width: int = 70, color: str = Colors.GREEN):
    """
    Print a formatted header.
    
    Args:
        title: Header title
        width: Width of header
        color: Color of header
    """
    print_separator("=", width, color)
    print_colored(title.center(width), color, bold=True)
    print_separator("=", width, color)


# Test functions
if __name__ == "__main__":
    # Test 1: Setup logger with console only
    print("\n1. Testing console logger...")
    logger = setup_logger("test_console")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test 2: Setup logger with file
    print("\n2. Testing file logger...")
    logger = setup_logger(
        "test_file",
        log_file="test_logs/test.log",
        use_colors=True
    )
    logger.info("This message goes to both console and file")
    
    # Test 3: Utility functions
    print("\n3. Testing utility functions...")
    log_section(logger, "SECTION HEADER")
    
    test_dict = {
        'model': 'densenet121',
        'training': {
            'epochs': 50,
            'batch_size': 32
        }
    }
    log_dict(logger, test_dict)
    
    # Test 4: Colored printing
    print("\n4. Testing colored printing...")
    print_header("EXPERIMENT RESULTS")
    print_colored("✓ Training completed", Colors.GREEN)
    print_colored("⚠ Warning: Low accuracy", Colors.YELLOW)
    print_colored("✗ Error occurred", Colors.RED)
    print_separator("-", 70, Colors.BLUE)
    
    print("\n✓ All logger tests passed!")