"""
Utility functions for MR processing pipeline.
"""

import logging
import sys
import time
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue, current_process
from pathlib import Path
from typing import Optional, Tuple

from brainles_preprocessing.constants import Atlas
from brainles_preprocessing.utils.zenodo import fetch_atlases


def setup_logging(log_dir: Path, script_name: str, verbose: bool = True) -> logging.Logger:
    """
    Set up logging to file and console for single-process scripts.
    
    Args:
        log_dir: Directory to save log file
        script_name: Name of the script for log file naming
        verbose: If True, also log to console; if False, only log to file
        
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # Optionally add console handler
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


def setup_parallel_logging(
    log_dir: Path, 
    script_name: str,
    verbose: bool = True
) -> Tuple[logging.Logger, Queue, QueueListener]:
    """
    Set up main process logger with queue for multiprocessing support.
    
    Uses QueueHandler/QueueListener pattern for process-safe logging.
    Worker processes should use BufferedStudyLogger to buffer logs per study
    and flush them all at once when processing completes.
    
    Args:
        log_dir: Directory to save log file
        script_name: Name of the script for log file naming
        verbose: If True, also log to console; if False, only log to file
        
    Returns:
        Tuple of (logger, log_queue, queue_listener)
        
    Note:
        Remember to call queue_listener.stop() when done to properly
        flush and close the log handlers.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Create handlers for the main logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Build handler list (always file, optionally console)
    handlers = [file_handler]
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Create queue for multiprocessing
    log_queue = Queue()
    
    # Queue listener that writes to handlers
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    queue_listener.start()
    
    # Create main logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    
    logger.info(f"Logging to: {log_file}")
    
    return logger, log_queue, queue_listener


class BufferedStudyLogger:
    """
    A logger that buffers log messages for a study and flushes them all at once.
    
    This prevents interleaved logs when multiple processes are running in parallel.
    All log messages for a study are collected in memory and sent to the queue
    as a single block when flush() is called.
    
    Usage:
        logger = BufferedStudyLogger(log_queue, study_id)
        logger.info("Processing...")
        logger.info("Step 1 complete")
        # ... more processing ...
        logger.flush()  # Sends all messages at once
    """
    
    def __init__(self, log_queue: Queue, study_id: str):
        """
        Initialize buffered logger for a study.
        
        Args:
            log_queue: Queue for log messages (from setup_parallel_logging)
            study_id: Study identifier for log header
        """
        self.log_queue = log_queue
        self.study_id = study_id
        self.buffer = []
        self.process_name = current_process().name
    
    def _add_message(self, level: str, msg: str):
        """Add a message to the buffer with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        self.buffer.append(f"{timestamp} - {self.process_name} - {level} - {msg}")
    
    def info(self, msg: str):
        """Buffer an INFO level message."""
        self._add_message("INFO", msg)
    
    def debug(self, msg: str):
        """Buffer a DEBUG level message."""
        self._add_message("DEBUG", msg)
    
    def warning(self, msg: str):
        """Buffer a WARNING level message."""
        self._add_message("WARNING", msg)
    
    def error(self, msg: str):
        """Buffer an ERROR level message."""
        self._add_message("ERROR", msg)
    
    def flush(self):
        """
        Flush all buffered messages to the queue as a single block.
        
        Messages are sent with a study header/footer for clear separation.
        """
        if not self.buffer:
            return
        
        # Join all messages with newlines
        combined_message = f"{'-' * 60}\n" + "\n".join(self.buffer)
        
        # Create a log record and send to queue
        # Using a simple LogRecord that the QueueListener will process
        # With args=(), getMessage() will just return msg as-is
        record = logging.LogRecord(
            name="BufferedStudyLogger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=combined_message,
            args=(),
            exc_info=None
        )
        
        self.log_queue.put_nowait(record)
        self.buffer.clear()


def fetch_atlas(
    atlas: Atlas = Atlas.MNI152,
    max_retries: int = 5,
    retry_wait_seconds: int = 10,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Fetch atlas from Zenodo with retry logic.
    
    Args:
        atlas: Atlas enum value (default: MNI152)
        max_retries: Maximum retry attempts
        retry_wait_seconds: Wait time between retries
        logger: Optional logger instance
        
    Returns:
        Path to atlas image
        
    Raises:
        Exception if fetching fails after all retries
    """
    if logger:
        logger.info("Fetching atlas for registration...")
    
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            atlas_folder = fetch_atlases()
            atlas_path = atlas_folder / atlas.value
            
            if logger:
                logger.info(f"Atlas path: {atlas_path}")
            
            return atlas_path
            
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(f"Atlas fetch attempt {attempt}/{max_retries} failed: {e}")
            
            if attempt < max_retries:
                if logger:
                    logger.info(f"Retrying in {retry_wait_seconds} seconds...")
                time.sleep(retry_wait_seconds)
    
    raise Exception(f"Failed to fetch atlas after {max_retries} attempts") from last_exception


def accession_to_uid(accession: str) -> str:
    """Derive an anonymized UID from an accession number."""
    # Original function has been replaced with a dummy 
    # function that returns the accession number as is.
    return accession