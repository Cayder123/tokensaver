"""
SessionLogger — Record AI usage to JSONL log files.

Logs are stored at: ~/.tokensaver/logs/YYYY-MM-DD.jsonl
Each line is a JSON record of one API call.

Example:
    from tokensaver import SessionLogger

    logger = SessionLogger(tag="my-project")
    logger.log(
        provider="anthropic",
        model="claude-sonnet-4-5",
        input_tokens=1200,
        output_tokens=350,
        cost_usd=0.00889,
    )
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# SessionLogger
# ---------------------------------------------------------------------------

class SessionLogger:
    """
    Append-only JSONL logger for AI API usage.

    Args:
        tag: Optional project/feature tag for filtering in reports.
        log_dir: Directory for log files. Default: ~/.tokensaver/logs/
        session_id: Unique session identifier. Auto-generated if not provided.
    """

    DEFAULT_LOG_DIR = Path.home() / ".tokensaver" / "logs"

    def __init__(
        self,
        tag: str = "",
        log_dir: Optional[Path | str] = None,
        session_id: Optional[str] = None,
    ):
        self.tag = tag
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())[:8]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Append one usage record to today's log file.

        Returns the record dict that was written.
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "tag": self.tag,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost_usd, 8),
        }
        if extra:
            record.update(extra)

        log_file = self._today_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_today(self) -> list[dict]:
        """Read all records from today's log."""
        return self._read_file(self._today_file())

    def read_date(self, date_str: str) -> list[dict]:
        """Read records for a specific date (YYYY-MM-DD)."""
        return self._read_file(self.log_dir / f"{date_str}.jsonl")

    def read_last_n_days(self, n: int) -> list[dict]:
        """Read records from the last N days."""
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        records = []
        for i in range(n):
            date = today - timedelta(days=i)
            records.extend(self.read_date(str(date)))
        return records

    def read_all(self) -> list[dict]:
        """Read all records across all log files."""
        records = []
        for path in sorted(self.log_dir.glob("*.jsonl")):
            records.extend(self._read_file(path))
        return records

    def filter_by_tag(self, records: list[dict], tag: str) -> list[dict]:
        return [r for r in records if r.get("tag") == tag]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _today_file(self) -> Path:
        today = datetime.now(timezone.utc).date()
        return self.log_dir / f"{today}.jsonl"

    @staticmethod
    def _read_file(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
