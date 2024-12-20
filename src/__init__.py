import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: str = "../output/logs", log_file: str = "debug.log") -> logging.Logger:
    """
    Sets up logging for the package
    """
    log_dir_path = Path(log_dir).resolve()                      # Resolve the absolute path of the log directory
    log_dir_path.mkdir(parents=True, exist_ok=True)             # Create the directory if it doesn't exist

    log_file_path = log_dir / log_file                          # Define the full path to the log file

    formatter = logging.Formatter(                              # Configure the logging format
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5 * 1024 * 1024, backupCount=2
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create the logger
    logger = logging.getLogger("src")
    logger.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log a message indicating logging setup
    logger.info("Logging setup complete")

    return logger

# Initialize logging for the src package
logger = setup_logging()

# Make the logger available for other modules
__all__ = ["setup_logging", "logger"]
