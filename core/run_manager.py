"""
core/run_manager.py
-------------------
Manages execution directories (runs/<run_id>/), tracks artifacts,
computes checksums, and implements state-level pipeline resumability.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import yaml
import platform
import sys

class RunManager:
    """
    Manages run directories and records output artifacts, metadata, and hashes.
    Supports pipeline stage-level resumability.
    """
    def __init__(self, root_dir: Path, run_id: str, resume: bool = False):
        self.root_dir = root_dir
        self.run_dir = root_dir / "runs" / run_id
        self.resume = resume
        self.manifest_path = self.run_dir / "manifest.json"
        
        # Telemetry manifest structure
        self.manifest = {
            "run_id": run_id,
            "status": "INCOMPLETE",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": "",
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "artifacts": []
        }
        
    def setup_directories(self, resolved_config: dict) -> Path:
        """Creates the run directory structure and writes config metadata."""
        if self.run_dir.exists() and not self.resume:
            # Overwrite protection
            raise FileExistsError(
                f"Run directory {self.run_dir} already exists. "
                "Enable resume or use a unique run ID to avoid overwrite."
            )
            
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories structure
        subdirs = [
            "logs", "data", "detections", "features", "predictions",
            "diagnostics", "fits", "metrics", "plots", "reports"
        ]
        for sd in subdirs:
            (self.run_dir / sd).mkdir(parents=True, exist_ok=True)
            
        # If resuming, load existing manifest
        if self.resume and self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    self.manifest = json.load(f)
                self.manifest["status"] = "RESUMED"
            except Exception:
                pass
                
        # Write resolved config
        with open(self.run_dir / "resolved_config.yaml", "w") as f:
            yaml.dump(resolved_config, f, default_flow_style=False)
            
        # Write environment.json
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor(),
            "cmd_args": sys.argv
        }
        with open(self.run_dir / "environment.json", "w") as f:
            json.dump(env_info, f, indent=2)
            
        return self.run_dir
        
    def record_artifact(
        self,
        target_id: str,
        stage: str,
        relative_path: str,
        schema_name: str,
        schema_version: str,
    ) -> dict:
        """
        Records a newly generated artifact in the run manifest.
        Calculates SHA-256 hash and size.
        """
        abs_path = self.run_dir / relative_path
        if not abs_path.exists():
            return {}
            
        # Calculate SHA-256 hash
        sha256 = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()
        file_size = abs_path.stat().st_size
        
        artifact_entry = {
            "target_id": target_id,
            "stage": stage,
            "relative_path": relative_path,
            "schema_name": schema_name,
            "schema_version": schema_version,
            "hash": file_hash,
            "size_bytes": file_size,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Avoid duplicate entries for same relative path in manifest
        self.manifest["artifacts"] = [
            a for a in self.manifest["artifacts"]
            if a["relative_path"] != relative_path
        ]
        self.manifest["artifacts"].append(artifact_entry)
        self.save_manifest()
        
        return artifact_entry
        
    def is_stage_completed(self, target_id: str, stage: str, expected_rel_path: str) -> bool:
        """
        Checks if a stage is already completed for a given target.
        Verifies path existence and matches manifest records.
        """
        if not self.resume:
            return False
            
        abs_path = self.run_dir / expected_rel_path
        if not abs_path.exists():
            return False
            
        # Check if record exists in manifest
        for art in self.manifest.get("artifacts", []):
            if (
                art.get("target_id") == target_id 
                and art.get("stage") == stage 
                and art.get("relative_path") == expected_rel_path
            ):
                return True
                
        return False
        
    def save_manifest(self):
        """Saves current state of run manifest to manifest.json."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
            
    def finalize_run(self, status: str = "COMPLETED"):
        """Computes final hashes and writes checksums.sha256 file."""
        self.manifest["status"] = status
        self.manifest["end_time"] = datetime.now(timezone.utc).isoformat()
        self.save_manifest()
        
        # Compute checksums for all files in run directory
        checksum_entries = []
        for root, _, files in os.walk(self.run_dir):
            for file in files:
                if file == "checksums.sha256":
                    continue
                abs_fpath = Path(root) / file
                rel_fpath = abs_fpath.relative_to(self.run_dir)
                
                sha256 = hashlib.sha256()
                with open(abs_fpath, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
                fhash = sha256.hexdigest()
                checksum_entries.append(f"{fhash}  {rel_fpath.as_posix()}")
                
        # Sort for stable output
        checksum_entries.sort()
        
        with open(self.run_dir / "checksums.sha256", "w", encoding="utf-8") as f:
            f.write("\n".join(checksum_entries) + "\n")
