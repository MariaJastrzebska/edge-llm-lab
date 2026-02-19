import doctest
import glob
import os
import sys
import io
import importlib.util
from contextlib import redirect_stdout
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def clear_terminal():
    """Wyczyść terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

class MyHandler(FileSystemEventHandler):
    def __init__(self, directory):
        self.directory = directory

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"\nZmiana w: {event.src_path}")
            run_doctests()

def run_doctests():
    clear_terminal()
    python_files = glob.glob(os.path.join(directory, "*.py"))
    if not python_files:
        print(f"Nie znaleziono plików .py w katalogu: {directory}")
        return
    total_failures = 0
    total_tests = 0
    for file in python_files:
        # Importuj moduł z absolutnej ścieżki i uruchom doctesty
        try:
            module_name = os.path.splitext(os.path.basename(file))[0]
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                buf = io.StringIO()
                with redirect_stdout(buf):
                    failures, tests = doctest.testmod(module, verbose=True)

                total_failures += failures
                total_tests += tests

                if failures:
                    print(f"\n❌ {file}: {failures}/{tests}")
                    print(buf.getvalue())
            else:
                print(f"Błąd importu modułu z pliku: {file}")
        except Exception as e:
            print(f"Błąd w pliku {file}: {e}")
    # Always show a one-line summary after each run
    status = "" if total_failures == 0 else "❌"
    print(f"{status} Summary: {total_tests} tests, {total_failures} failures")

if __name__ == "__main__":
    # Watch ONLY the folder where this script lives
    directory = os.path.dirname(os.path.abspath(__file__))
    # Ensure parent directory is on sys.path so `import thesis_generators.*` works in doctests
    parent = os.path.dirname(directory)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Uruchom doctesty na starcie
    print("Uruchamianie początkowych doctestów...")
    run_doctests()
    # Ustaw monitorowanie
    event_handler = MyHandler(directory)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    print(f"Monitorowanie: {directory}. Naciśnij Ctrl+C, aby zatrzymać.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Zatrzymano monitorowanie.")
    observer.join()