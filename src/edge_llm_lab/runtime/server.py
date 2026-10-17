"""Owned llama-server lifecycle. Reuse only identical runtime configurations."""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests


class LlamaServerRuntime:
    def __init__(self, executable, host, port, timeout):
        self.executable, self.host, self.port = executable, host, port
        self.timeout = timeout
        self.session = requests.Session()
        self.process = None
        self.log = None
        self.identity = None

    def ensure(self, model_path, context_size, max_tokens, optimisations):
        identity = json.dumps([str(Path(model_path).resolve()), context_size, max_tokens, optimisations], sort_keys=True)
        if self.process is not None and self.process.poll() is None and self.identity == identity:
            return self.process
        self.stop()
        # Never terminate a process started outside this runtime.
        try:
            self.session.get(f'http://{self.host}:{self.port}/health', timeout=2)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError('Configured server port is occupied; choose another port')
        command = [self.executable, '--model', model_path, '--host', self.host,
                   '--port', str(self.port), '--ctx-size', str(context_size),
                   '--n-predict', str(max_tokens), '--jinja']
        for key, value in sorted(optimisations.items()):
            if not key.startswith('--') or key in {'--model', '--host', '--port', '--ctx-size', '--n-predict'}:
                raise ValueError(f'Unsupported or conflicting runtime option: {key}')
            command.append(key)
            if value is not None:
                command.append(str(value))
        self.log = tempfile.TemporaryFile(mode='w+t')
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'llama-server exited with code {self.process.returncode}')
                try:
                    response = self.session.get(f'http://{self.host}:{self.port}/health', timeout=1)
                    if response.status_code == 200:
                        self.identity = identity
                        return self.process
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            raise TimeoutError('llama-server readiness deadline exceeded')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.identity = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def close(self):
        self.stop()
        self.session.close()
