import os
import neptune
from neptune.types import File
from typing import Dict, Any, List, Optional

class NeptuneManager:
    """Manages Neptune AI integration for LLM evaluation."""
    
    def __init__(self, api_token: Optional[str] = None, project: Optional[str] = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_token = api_token or os.getenv("NEPTUNE_API_TOKEN")
        self.project = project or os.getenv("NEPTUNE_PROJECT")
        self.run = None

    def init_run(self, name: str, tags: List[str] = None, params: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> bool:
        """Initializes a Neptune run."""
        if not (self.api_token and self.project):
            return False
            
        try:
            self.run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                name=name,
                tags=tags,
                capture_hardware_metrics=True,
                capture_stdout=True,
                capture_stderr=True
            )
            
            if params:
                self.run["parameters"] = params
            
            if metadata:
                self.run["model_metadata"] = metadata
                if metadata.get("quantization"):
                    self.run["sys/tags"].add(f"quant:{metadata['quantization']}")
            
            print(f"🚀 Neptune run initialized: {self.run.get_url()}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to initialize Neptune: {e}")
            return False

    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any], latency: Dict[str, Any] = None):
        """Logs metrics and latency for a specific evaluation round in real-time."""
        if not self.run:
            return
            
        prefix = f"rounds/round_{round_num}"
        
        # Log metrics
        for m_name, m_data in metrics.items():
            if isinstance(m_data, dict) and 'score' in m_data:
                self.run[f"{prefix}/metrics/{m_name}"] = m_data['score']
            elif isinstance(m_data, (int, float)):
                self.run[f"{prefix}/metrics/{m_name}"] = m_data
        
        # Log latency
        if latency:
            self.run[f"{prefix}/latency/total_ms"] = latency.get('total_ms')
            self.run[f"{prefix}/latency/throughput"] = latency.get('tokens', {}).get('throughput_tokens_per_sec')

    def upload_artifact(self, local_path: str, neptune_path: str):
        """Uploads a file artifact to Neptune."""
        if self.run and os.path.exists(local_path):
            try:
                # Use File type for images to enable rich preview in Neptune
                if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.run[neptune_path].upload(File(local_path))
                else:
                    self.run[neptune_path].upload(local_path)
                print(f"DEBUG: Uploaded {local_path} -> {neptune_path}")
            except Exception as e:
                print(f"⚠️ Failed to upload {local_path} to Neptune: {e}")

    def upload_images_series(self, image_paths: List[str], neptune_path: str):
        """Uploads a list of images as a Neptune series (mosaic view)."""
        if not (self.run and image_paths):
            return
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    self.run[neptune_path].append(File(path))
                except Exception as e:
                    print(f"⚠️ Failed to append {path} to Neptune series {neptune_path}: {e}")

    def upload_directory_artifacts(self, local_dir: str, neptune_path_prefix: str, extensions=(".png", ".jpg", ".jpeg", ".json", ".csv", ".tex"), gallery_path: Optional[str] = None):
        """Uploads all matching files from a directory and its subdirectories to Neptune.
        
        Args:
            local_dir: Local directory to scan
            neptune_path_prefix: Prefix for Neptune artifact paths
            extensions: Tuple of allowed file extensions
            gallery_path: Optional Neptune path to append all images to (for mosaic view)
        """
        print(f"DEBUG: Starting artifact upload from {local_dir}")
        if not (self.run and os.path.exists(local_dir)):
            print(f"DEBUG: Skipping upload - Run: {self.run is not None}, Exists: {os.path.exists(local_dir)}")
            return
            
        files_found = 0
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                if f.lower().endswith(extensions):
                    files_found += 1
                    full_path = os.path.join(root, f)
                    # Create a relative path for Neptune based on the provided prefix and folder structure
                    rel_path = os.path.relpath(full_path, local_dir)
                    neptune_file_path = f"{neptune_path_prefix}/{rel_path}"
                    self.upload_artifact(full_path, neptune_file_path)
                    
                    # Also append images to gallery if requested
                    if gallery_path and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        print(f"DEBUG: Appending {f} to gallery {gallery_path}")
                        try:
                            self.run[gallery_path].append(File(full_path))
                        except Exception as e:
                            print(f"⚠️ Failed to append {full_path} to Neptune gallery {gallery_path}: {e}")
        print(f"DEBUG: Scanned {local_dir}, matching files found: {files_found}")

    def recover_log(self, run_id: str, target_path: str) -> bool:
        """Downloads a JSON log from a past Neptune run."""
        if not (self.api_token and self.project):
            print("❌ Neptune credentials missing. Cannot recover log.")
            return False
            
        try:
            run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                with_id=run_id,
                mode="read-only"
            )
            print(f"🔄 Recovering log from Neptune run: {run_id}")
            
            artifact_paths = ["final_logs/main_log", "artifacts/main_log.json", "model_artifacts/main_log.json"]
            success = False
            
            for path in artifact_paths:
                if path in run:
                    run[path].download(target_path)
                    print(f"✅ Log recovered and saved to: {target_path}")
                    success = True
                    break
            
            run.stop()
            return success
        except Exception as e:
            print(f"⚠️ Error recovering log from Neptune: {e}")
            return False

    def stop(self):
        """Stops the Neptune run."""
        if self.run:
            self.run.stop()
            self.run = None
