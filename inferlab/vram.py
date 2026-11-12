"""Peak VRAM sampling via nvidia-smi.

Polling nvidia-smi during a run is what catches the failure mode that end-to-end timing
misses entirely: a config that finishes the benchmark because nothing ever OOMs, but leaves
only a few hundred MB of headroom and would collapse under real concurrency.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Self


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


class VramSampler:
    """Sample `memory.used` on one GPU at a fixed interval; report the peak.

    Runs on a daemon thread. Sampling is best-effort: if nvidia-smi is missing or fails, the
    sampler degrades to "no data" rather than breaking the benchmark.
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.peak_mb: float | None = None
        self.samples: int = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not available():
            self.error = "nvidia-smi not found on PATH"
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if out.returncode == 0:
                    used = float(out.stdout.strip().splitlines()[0])
                    self.samples += 1
                    if self.peak_mb is None or used > self.peak_mb:
                        self.peak_mb = used
                elif self.error is None:
                    self.error = out.stderr.strip()[:200]
            except Exception as exc:  # noqa: BLE001 - sampling must never kill the run
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"[:200]
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, object]:
        return {
            "peak_mb": self.peak_mb,
            "samples": self.samples,
            "interval_s": self.interval,
            "gpu_index": self.gpu_index,
            "error": self.error,
        }
