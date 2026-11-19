"""
Memory Monitoring Module.
Tracks process Resident Set Size (RSS), peak memory, system RAM utilization,
and GPU memory usage during model inference and benchmarking.
"""

import os
import threading
import time
from typing import Any, Dict, Optional
import psutil


class MemoryMonitor:
    """
    Monitors process and system memory consumption over time.
    Can run as a background thread to sample peak RAM during active inference.
    """

    def __init__(self, target_pid: Optional[int] = None, interval_sec: float = 0.05):
        self.target_pid = target_pid or os.getpid()
        self.interval_sec = interval_sec
        self.process = psutil.Process(self.target_pid)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.baseline_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.current_rss_bytes = 0
        self.baseline_system_ram_gb = 0.0
        self.peak_system_ram_gb = 0.0

    def start(self):
        """Starts background memory sampling."""
        try:
            mem_info = self.process.memory_info()
            self.baseline_rss_bytes = mem_info.rss
            self.peak_rss_bytes = mem_info.rss
            self.current_rss_bytes = mem_info.rss
            sys_mem = psutil.virtual_memory()
            self.baseline_system_ram_gb = sys_mem.used / (1024**3)
            self.peak_system_ram_gb = self.baseline_system_ram_gb
        except Exception:
            pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                mem_info = self.process.memory_info()
                rss = mem_info.rss
                self.current_rss_bytes = rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss

                sys_used = psutil.virtual_memory().used / (1024**3)
                if sys_used > self.peak_system_ram_gb:
                    self.peak_system_ram_gb = sys_used
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval_sec)

    def stop(self) -> Dict[str, float]:
        """Stops sampling and returns metrics in MB and GB."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Final check
        try:
            mem_info = self.process.memory_info()
            if mem_info.rss > self.peak_rss_bytes:
                self.peak_rss_bytes = mem_info.rss
        except Exception:
            pass

        return {
            "baseline_ram_mb": round(self.baseline_rss_bytes / (1024**2), 2),
            "peak_ram_mb": round(self.peak_rss_bytes / (1024**2), 2),
            "delta_ram_mb": round((self.peak_rss_bytes - self.baseline_rss_bytes) / (1024**2), 2),
            "peak_ram_gb": round(self.peak_rss_bytes / (1024**3), 3),
            "baseline_system_ram_gb": round(self.baseline_system_ram_gb, 2),
            "peak_system_ram_gb": round(self.peak_system_ram_gb, 2),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def monitor_subprocess_memory(proc_obj, interval_sec: float = 0.05) -> Dict[str, float]:
    """
    Tracks peak memory for a subprocess.Popen object until termination.
    """
    peak_rss = 0
    try:
        p = psutil.Process(proc_obj.pid)
        while proc_obj.poll() is None:
            try:
                rss = p.memory_info().rss
                # include child processes if spawned
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                if rss > peak_rss:
                    peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(interval_sec)
    except Exception:
        pass

    return {
        "peak_rss_mb": round(peak_rss / (1024**2), 2),
        "peak_rss_gb": round(peak_rss / (1024**3), 3),
    }


if __name__ == "__main__":
    print("Testing MemoryMonitor...")
    with MemoryMonitor() as m:
        # Allocate a temporary 50 MB buffer
        arr = bytearray(50 * 1024 * 1024)
        time.sleep(0.2)
        del arr
    stats = m.stop()
    print("MemoryMonitor stats:", stats)
