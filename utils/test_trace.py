"""Per-sample trace log for test.py (survives native crashes via flush after each stage)."""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
from datetime import datetime, timezone


def print_repo_state() -> None:
    """Print branch/commit/dirty state so server runs are attributable."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        flag = " (dirty working tree)" if dirty else ""
        print(f"Git: branch={branch} commit={rev}{flag}")
        if dirty:
            print("  Uncommitted changes — server may not match origin:")
            for line in dirty.splitlines()[:12]:
                print(f"    {line}")
            if len(dirty.splitlines()) > 12:
                print(f"    ... +{len(dirty.splitlines()) - 12} more")
    except (OSError, subprocess.CalledProcessError):
        print("Git: (not a git checkout or git unavailable)")


def _rss_mb() -> float:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: ru_maxrss in KiB; macOS: bytes
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024 * 1024)
        return ru.ru_maxrss / 1024
    except Exception:
        return float("nan")


class TestTraceLogger:
    """Append-only TSV; each line flushed immediately."""

    HEADER = "ts_utc\tidx\tstage\tunique_id\tframe_id\tfile_name\trss_mb\tnote\n"

    def __init__(self, path: str | None):
        self.path = path
        self._fp = None
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        new_file = not os.path.isfile(path) or os.path.getsize(path) == 0
        self._fp = open(path, "a", encoding="utf-8", buffering=1)
        if new_file:
            self._fp.write(self.HEADER)
            self._fp.flush()
        faulthandler.enable(file=self._fp, all_threads=True)
        print(f"Test trace log: {path} (faulthandler enabled)")

    @property
    def enabled(self) -> bool:
        return self._fp is not None

    def event(
        self,
        idx: int,
        stage: str,
        *,
        unique_id: str = "",
        frame_id: str = "",
        file_name: str = "",
        note: str = "",
    ) -> None:
        if not self._fp:
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        line = (
            f"{ts}\t{idx}\t{stage}\t{unique_id}\t{frame_id}\t"
            f"{file_name}\t{_rss_mb():.1f}\t{note}\n"
        )
        self._fp.write(line)
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def close(self) -> None:
        if self._fp:
            self.event(-1, "trace_closed", note="normal exit")
            self._fp.close()
            self._fp = None
