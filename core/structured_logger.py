"""
core/structured_logger.py
-------------------------
Structured logging utilities for TransitLens.
Generates human-readable console output and machine-readable JSONL logs.
Tracks target-level statuses, stage execution times, and warning summaries.
"""

import logging
import json
from datetime import datetime, timezone
import time as _time
from pathlib import Path

# Global in-memory run telemetry
run_telemetry = {
    "run_id": "unknown",
    "status": "INCOMPLETE",
    "start_time": "",
    "end_time": "",
    "targets_completed": [],
    "targets_failed": [],
    "targets_skipped": [],
    "warnings_summary": {},
    "stage_runtimes_ms": {},
    "artifacts_generated": []
}

class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON structures."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        
        # Pull extra attributes if attached using the extra={} argument
        for attr in ["target_id", "run_id", "stage", "elapsed_ms", "warning_code"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
                
        return json.dumps(log_entry)

class TelemetryTrackingHandler(logging.Handler):
    """A logging handler that updates the global run telemetry in-memory."""
    def emit(self, record):
        # Track warning categories
        if record.levelno == logging.WARNING:
            msg = record.getMessage()
            warning_type = msg.split(":")[0] if ":" in msg else "General"
            run_telemetry["warnings_summary"][warning_type] = run_telemetry["warnings_summary"].get(warning_type, 0) + 1
            
        # Track elapsed times
        if hasattr(record, "stage") and hasattr(record, "elapsed_ms"):
            stage_name = getattr(record, "stage")
            duration = getattr(record, "elapsed_ms")
            run_telemetry["stage_runtimes_ms"][stage_name] = run_telemetry["stage_runtimes_ms"].get(stage_name, 0.0) + duration

def setup_structured_logging(
    console_level=logging.INFO,
    json_log_path: Path | None = None,
    run_id: str = "unknown"
):
    """
    Sets up structured logging handlers.
    Configures standard stderr output and optionally writes JSON logs to json_log_path.
    """
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to prevent duplicate prints
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        
    root_logger.setLevel(logging.DEBUG) # capture all, filter at handler level
    
    # 1. Console stderr handler (Human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. JSON log file handler (Machine-readable)
    if json_log_path:
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
        
    # 3. Telemetry tracking handler
    telemetry_handler = TelemetryTrackingHandler()
    telemetry_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(telemetry_handler)
    
    # Initialize telemetry fields
    run_telemetry["run_id"] = run_id
    run_telemetry["status"] = "RUNNING"
    run_telemetry["start_time"] = datetime.now(timezone.utc).isoformat()
    run_telemetry["targets_completed"] = []
    run_telemetry["targets_failed"] = []
    run_telemetry["targets_skipped"] = []
    run_telemetry["warnings_summary"] = {}
    run_telemetry["stage_runtimes_ms"] = {}
    run_telemetry["artifacts_generated"] = []

def finalize_telemetry(status: str = "COMPLETED"):
    """Marks run complete and records final timestamp."""
    run_telemetry["status"] = status
    run_telemetry["end_time"] = datetime.now(timezone.utc).isoformat()
    return run_telemetry
