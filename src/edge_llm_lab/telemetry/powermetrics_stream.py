"""Continuous powermetrics sampling with per-window energy integration.

This lifts the ``KNOWN LIMITATION`` documented in :mod:`.resources`: a
:class:`MacPowermetricsBackend` sample is an *instantaneous* power reading (mW)
taken before/after a run, which cannot answer "how many Joules did this
generation cost". G1/G2 need Joules, so this module keeps ONE long-lived
``powermetrics`` process running for the whole experiment and integrates its
power samples over arbitrary time windows.

Design notes that matter for the paper:

* **No clock drift.** ``powermetrics`` text output timestamps samples only to
  1-second resolution, but reports each sample's true duration
  (``(306.11ms elapsed)``). Instead of accumulating those durations from a
  start time (which drifts), a reader thread stamps each sample with
  ``time.monotonic()`` at the moment the sample's last line is parsed. A sample
  therefore covers ``[stamp - elapsed, stamp]``, self-correcting every sample.
* **Whole-SoC measurement.** powermetrics reports package power, not
  per-process power. Energy for a generation window includes everything else
  running on the machine, including powermetrics itself. Record an idle
  baseline (:meth:`PowermetricsRecorder.measure_baseline`) and report both
  gross and baseline-subtracted energy; the sampler's own overhead is constant
  across conditions, so it cancels in a *comparison* but not in an absolute
  number.
* **Coverage is reported, never assumed.** :meth:`energy_for_window` returns
  ``coverage`` (fraction of the window actually covered by samples). A window
  shorter than the sampling interval can land between samples; the caller must
  be able to see that rather than silently receive an interpolated number.

Requires passwordless sudo for powermetrics (``/etc/sudoers.d/powermetrics``).
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

_ELAPSED_RE = re.compile(r"\(([\d.]+)ms elapsed\)")
_CPU_RE = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW")
_GPU_RE = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW")
_ANE_RE = re.compile(r"^ANE Power:\s*([\d.]+)\s*mW")
_COMBINED_RE = re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")


@dataclass
class PowerSample:
    """One powermetrics sample, covering ``[t_end - duration_s, t_end]``."""

    t_end: float  # time.monotonic() when this sample finished being parsed
    duration_s: float
    cpu_mw: Optional[float] = None
    gpu_mw: Optional[float] = None
    ane_mw: Optional[float] = None
    combined_mw: Optional[float] = None

    @property
    def t_start(self) -> float:
        return self.t_end - self.duration_s


@dataclass
class EnergyWindow:
    """Integrated energy over one [start, end] monotonic-clock window."""

    duration_s: float
    n_samples: int
    coverage: float
    cpu_j: Optional[float] = None
    gpu_j: Optional[float] = None
    ane_j: Optional[float] = None
    combined_j: Optional[float] = None
    mean_cpu_mw: Optional[float] = None
    mean_gpu_mw: Optional[float] = None
    mean_combined_mw: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "energy_window_s": round(self.duration_s, 4),
            "energy_n_samples": self.n_samples,
            "energy_coverage": round(self.coverage, 4),
            "cpu_j": _r(self.cpu_j),
            "gpu_j": _r(self.gpu_j),
            "ane_j": _r(self.ane_j),
            "combined_j": _r(self.combined_j),
            "mean_cpu_mw": _r(self.mean_cpu_mw, 1),
            "mean_gpu_mw": _r(self.mean_gpu_mw, 1),
            "mean_combined_mw": _r(self.mean_combined_mw, 1),
        }


def _r(v: Optional[float], nd: int = 5) -> Optional[float]:
    return None if v is None else round(v, nd)


class PowermetricsRecorder:
    """Long-lived ``powermetrics`` process + integration over time windows.

    Usage::

        rec = PowermetricsRecorder(interval_ms=100)
        rec.start()
        t0 = time.monotonic(); ...generate...; t1 = time.monotonic()
        window = rec.energy_for_window(t0, t1)
        rec.stop()
    """

    def __init__(self, interval_ms: int = 100, settle_s: float = 2.0):
        self.interval_ms = interval_ms
        self.settle_s = settle_s
        self.samples: List[PowerSample] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.start_error: Optional[str] = None
        self.baseline_mw: Optional[float] = None

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> bool:
        """Launch powermetrics. Returns False (and sets ``start_error``) on failure."""
        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power",
            "-i", str(self.interval_ms),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            self.start_error = f"failed to launch powermetrics: {e}"
            return False

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Wait for the first real sample so callers never integrate over an
        # empty buffer.
        deadline = time.monotonic() + max(self.settle_s, 5.0)
        while time.monotonic() < deadline:
            with self._lock:
                if self.samples:
                    return True
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() or "").strip() if self._proc.stderr else ""
                self.start_error = (
                    f"powermetrics exited early (rc={self._proc.returncode}): {err[:400]}"
                )
                return False
            time.sleep(0.05)
        self.start_error = "powermetrics produced no samples before timeout"
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=3)

    def __enter__(self) -> "PowermetricsRecorder":
        self.started = self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- reader --------------------------------------------------------

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        duration_s: Optional[float] = None
        cur = {}
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _ELAPSED_RE.search(line)
            if "Sampled system activity" in line and m:
                duration_s = float(m.group(1)) / 1000.0
                cur = {}
                continue
            for key, rx in (
                ("cpu_mw", _CPU_RE),
                ("gpu_mw", _GPU_RE),
                ("ane_mw", _ANE_RE),
            ):
                mm = rx.match(line)
                if mm:
                    cur[key] = float(mm.group(1))
            mc = _COMBINED_RE.match(line)
            if mc and duration_s is not None:
                # "Combined Power" is the last power line of a sample block:
                # stamp NOW so the sample interval is anchored to real time.
                sample = PowerSample(
                    t_end=time.monotonic(),
                    duration_s=duration_s,
                    cpu_mw=cur.get("cpu_mw"),
                    gpu_mw=cur.get("gpu_mw"),
                    ane_mw=cur.get("ane_mw"),
                    combined_mw=float(mc.group(1)),
                )
                with self._lock:
                    self.samples.append(sample)
                duration_s = None
                cur = {}

    # ---- integration ---------------------------------------------------

    def energy_for_window(self, t_start: float, t_end: float) -> EnergyWindow:
        """Integrate power over ``[t_start, t_end]`` (time.monotonic seconds).

        Each sample contributes ``power * overlap(sample_interval, window)``.
        Partial overlaps at the window edges are prorated, which is why a
        100 ms sampler can still bound a 700 ms generation sensibly.
        """
        window_s = max(t_end - t_start, 0.0)
        with self._lock:
            samples = list(self.samples)

        acc = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "combined": 0.0}
        have = {"cpu": False, "gpu": False, "ane": False, "combined": False}
        covered = 0.0
        n = 0
        for s in samples:
            overlap = min(s.t_end, t_end) - max(s.t_start, t_start)
            if overlap <= 0:
                continue
            n += 1
            covered += overlap
            for key, mw in (
                ("cpu", s.cpu_mw),
                ("gpu", s.gpu_mw),
                ("ane", s.ane_mw),
                ("combined", s.combined_mw),
            ):
                if mw is not None:
                    acc[key] += (mw / 1000.0) * overlap  # mW→W, W·s = J
                    have[key] = True

        coverage = (covered / window_s) if window_s > 0 else 0.0

        def mean_mw(key: str) -> Optional[float]:
            if not have[key] or covered <= 0:
                return None
            return (acc[key] / covered) * 1000.0

        return EnergyWindow(
            duration_s=window_s,
            n_samples=n,
            coverage=coverage,
            cpu_j=acc["cpu"] if have["cpu"] else None,
            gpu_j=acc["gpu"] if have["gpu"] else None,
            ane_j=acc["ane"] if have["ane"] else None,
            combined_j=acc["combined"] if have["combined"] else None,
            mean_cpu_mw=mean_mw("cpu"),
            mean_gpu_mw=mean_mw("gpu"),
            mean_combined_mw=mean_mw("combined"),
        )

    def measure_baseline(self, seconds: float = 5.0) -> Optional[float]:
        """Mean combined power (mW) over an idle stretch, for baseline subtraction."""
        t0 = time.monotonic()
        time.sleep(seconds)
        w = self.energy_for_window(t0, time.monotonic())
        self.baseline_mw = w.mean_combined_mw
        return self.baseline_mw
