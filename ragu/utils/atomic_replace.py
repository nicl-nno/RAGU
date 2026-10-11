"""Bounded retries for transient Windows locks during atomic persistence."""
import os
import time


def _replace_with_retry(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Keep atomic replacement semantics; fail normally for permanent errors."""
    deadline = time.monotonic() + 3.0
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            remaining = deadline - time.monotonic()
            if getattr(error, 'winerror', None) not in (5, 32, 33) or remaining <= 0:
                raise
            time.sleep(min(0.05, remaining))
