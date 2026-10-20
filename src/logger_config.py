"""
TenderIQ — Centralized Logging Configuration
==============================================

WHY LOGGING?
    Print statements are okay for tiny scripts, but they don't scale:
    - You can't turn them off
    - No timestamps
    - No severity levels
    - Can't send to file AND console at the same time
    - No easy way to track which module produced which message

WHAT LOGGING GIVES YOU:
    - Timestamps automatically
    - Severity levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Logs go to console AND to a file simultaneously
    - Every log message shows which module it came from
    - You can filter by module or level

USAGE IN OTHER FILES:
    from src.logger_config import get_logger

    logger = get_logger(__name__)

    logger.debug("Detailed info for debugging")
    logger.info("Normal operational message")
    logger.warning("Something unexpected but not fatal")
    logger.error("Something failed")
    logger.critical("Serious problem, app might crash")
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Log directory - all log files go here
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log file name includes date so you get one file per day
LOG_FILE = LOG_DIR / f"tenderiq_{datetime.now().strftime('%Y-%m-%d')}.log"

# Log format
# %(asctime)s     → timestamp
# %(name)s        → logger name (usually module name)
# %(levelname)s   → INFO, WARNING, etc.
# %(message)s     → the actual message
LOG_FORMAT_FILE = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
)

# Console format is shorter for readability
LOG_FORMAT_CONSOLE = "%(levelname)-8s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ═══════════════════════════════════════════════════════
# TRACKING (to avoid re-configuring loggers)
# ═══════════════════════════════════════════════════════

_configured_loggers = set()


# ═══════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger for a module.

    This function returns a logger that:
    - Writes DEBUG and above to the log file
    - Writes INFO and above to the console
    - Adds timestamps automatically
    - Doesn't duplicate handlers if called multiple times

    Args:
        name: Usually pass __name__ from the calling module.
              This shows in logs so you know which file logged what.
        level: Minimum level to log (default: INFO)

    Returns:
        Configured Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting extraction")
    """
    logger = logging.getLogger(name)

    # Avoid re-configuring same logger multiple times
    if name in _configured_loggers:
        return logger

    logger.setLevel(logging.DEBUG)  # Logger accepts all, handlers filter

    # Prevent duplicate messages from propagating to root logger
    logger.propagate = False

    # ─────────────────────────────────────────────
    # HANDLER 1: File handler (all logs to file)
    # ─────────────────────────────────────────────
    file_handler = logging.FileHandler(
        LOG_FILE,
        mode='a',           # append mode
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Everything to file
    file_formatter = logging.Formatter(LOG_FORMAT_FILE, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    # ─────────────────────────────────────────────
    # HANDLER 2: Console handler (INFO+ to terminal)
    # ─────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Only INFO+ to console
    console_formatter = logging.Formatter(LOG_FORMAT_CONSOLE)
    console_handler.setFormatter(console_formatter)

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Mark as configured
    _configured_loggers.add(name)

    return logger


# ═══════════════════════════════════════════════════════
# UTILITY: Log a section header for clarity
# ═══════════════════════════════════════════════════════

def log_section(logger: logging.Logger, title: str) -> None:
    """
    Print a visible section header in logs.

    Useful for separating different phases of the pipeline.

    Example:
        log_section(logger, "STEP 1: PDF EXTRACTION")
    """
    line = "=" * 60
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test")
    log_section(logger, "LOGGER TEST")
    logger.debug("This is a DEBUG message (only in file)")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print(f"\nLog file created: {LOG_FILE}")