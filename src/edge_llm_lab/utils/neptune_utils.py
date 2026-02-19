import os
import neptune
from neptune.types import File
from typing import Dict, Any, List, Optional

class NeptuneManager:
    """Zarządza integracją z Neptune AI dla ewaluacji LLM."""
    
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
        """Inicjalizuje run w Neptune."""
        if not (self.api_token and self.project):
            print(f" Brak poświadczeń Neptune. Token={bool(self.api_token)}, Projekt={bool(self.project)}")
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
            
            print(f" Run zainicjalizowany: {self.run.get_url()}")
            return True
        except Exception as e:
            print(f" Błąd inicjalizacji Neptune: {e}")
            return False

    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any], latency: Dict[str, Any] = None):
        """Loguje metryki rundy w czasie rzeczywistym."""
        if not self.run:
            return
            
        prefix = f"rounds/round_{round_num}"
        
        # Logowanie metryk
        for m_name, m_data in metrics.items():
            if isinstance(m_data, dict) and 'score' in m_data:
                self.run[f"{prefix}/metrics/{m_name}"] = m_data['score']
            elif isinstance(m_data, (int, float)):
                self.run[f"{prefix}/metrics/{m_name}"] = m_data
        
        # Logowanie latencji
        if latency:
            self.run[f"{prefix}/latency/total_ms"] = latency.get('total_ms')
            self.run[f"{prefix}/latency/throughput"] = latency.get('tokens', {}).get('throughput_tokens_per_sec')

    def upload_artifact(self, local_path: str, neptune_path: str):
        """Wysyła artefakt plikowy do Neptune."""
        if self.run and os.path.exists(local_path):
            try:
                # Użyj typu File dla obrazów dla podglądu w Neptune
                if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.run[neptune_path].upload(File(local_path))
                else:
                    self.run[neptune_path].upload(local_path)
                print(f"DEBUG: Uploaded {local_path} -> {neptune_path}")
            except Exception as e:
                print(f" Błąd wysyłania {local_path} do Neptune: {e}")

    def upload_images_series(self, image_paths: List[str], neptune_path: str):
        """Wysyła listę obrazów jako serię (widok mozaiki)."""
        if not (self.run and image_paths):
            return
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    self.run[neptune_path].append(File(path))
                except Exception as e:
                    print(f" Błąd dodawania {path} do serii {neptune_path}: {e}")

    def upload_directory_artifacts(self, local_dir: str, neptune_path_prefix: str, extensions=(".png", ".jpg", ".jpeg", ".json", ".csv", ".tex"), gallery_path: Optional[str] = None):
        """Przesyła pasujące pliki z katalogu do Neptune.
        
        Args:
            local_dir: Katalog lokalny
            neptune_path_prefix: Prefiks ścieżki w Neptune
            extensions: Dozwolone rozszerzenia
            gallery_path: Opcjonalna ścieżka galerii obrazów
        """
        print(f"DEBUG: Starting artifact upload from {local_dir}")
        if not (self.run and os.path.exists(local_dir)):
            print(f"DEBUG: Skipping upload - Run: {self.run is not None}, Exists: {os.path.exists(local_dir)}")
            return
            
        files_found = 0
        for root, dirs, files in os.walk(local_dir):
            print(f"DEBUG: Scanned {root} - found {len(files)} files: {files}")
            for f in files:
                if f.lower().endswith(extensions):
                    files_found += 1
                    full_path = os.path.join(root, f)
                    # Ścieżka relatywna dla Neptune na podstawie folderów
                    rel_path = os.path.relpath(full_path, local_dir)
                    neptune_file_path = f"{neptune_path_prefix}/{rel_path}"
                    self.upload_artifact(full_path, neptune_file_path)
                    
                    # Dodaj obrazy do galerii jeśli podano ścieżkę
                    if gallery_path and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        print(f"DEBUG: Appending {f} to gallery {gallery_path}")
                        try:
                            self.run[gallery_path].append(File(full_path))
                        except Exception as e:
                            print(f" Błąd dodawania {full_path} do galerii {gallery_path}: {e}")
        print(f"DEBUG: Scanned {local_dir}, matching files found: {files_found}")

    def recover_log(self, run_id: str, target_path: str) -> bool:
        """Pobiera log JSON z poprzedniego runu."""
        if not (self.api_token and self.project):
            print(" Brak poświadczeń Neptune.")
            return False
            
        try:
            run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                with_id=run_id,
                mode="read-only"
            )
            print(f" Przywracanie logu z runu: {run_id}")
            
            artifact_paths = ["final_logs/main_log", "artifacts/main_log.json", "model_artifacts/main_log.json"]
            success = False
            
            for path in artifact_paths:
                if path in run:
                    run[path].download(target_path)
                    print(f" Log zapisany: {target_path}")
                    success = True
                    break
            
            run.stop()
            return success
        except Exception as e:
            print(f" Błąd pobierania logu: {e}")
            return False

    def stop(self):
        """Zatrzymuje run Neptune."""
        if self.run:
            self.run.stop()
            self.run = None
