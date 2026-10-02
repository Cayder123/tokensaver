"""
Reporter — Generate cost/usage reports from session logs.

Example:
    from tokensaver import Reporter, SessionLogger

    logger = SessionLogger()
    reporter = Reporter(logger)
    reporter.print_today()
    reporter.print_last_n_days(7)
    reporter.export_csv("report.csv")
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tokensaver.logger import SessionLogger


class Reporter:
    """
    Generate and display usage reports from session logs.

    Args:
        logger: SessionLogger instance to read data from.
    """

    def __init__(self, logger: Optional[SessionLogger] = None):
        self._logger = logger or SessionLogger()

    # ------------------------------------------------------------------
    # Print reports to terminal
    # ------------------------------------------------------------------

    def print_today(self, tag: Optional[str] = None) -> None:
        records = self._logger.read_today()
        if tag:
            records = self._logger.filter_by_tag(records, tag)
        self._print_report(records, title="Today")

    def print_last_n_days(self, n: int = 7, tag: Optional[str] = None) -> None:
        records = self._logger.read_last_n_days(n)
        if tag:
            records = self._logger.filter_by_tag(records, tag)
        self._print_report(records, title=f"Last {n} days")

    def print_all(self, tag: Optional[str] = None) -> None:
        records = self._logger.read_all()
        if tag:
            records = self._logger.filter_by_tag(records, tag)
        self._print_report(records, title="All time")

    def _print_report(self, records: list[dict], title: str) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
            self._print_rich(records, title)
        except ImportError:
            self._print_plain(records, title)

    def _print_rich(self, records: list[dict], title: str) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text

        console = Console()
        now = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

        if not records:
            console.print(f"[dim]📊 TokenSaver — {title} — No records found[/dim]")
            return

        # Aggregate by provider + model
        agg: dict[tuple, dict] = defaultdict(lambda: {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0
        })
        for r in records:
            key = (r.get("provider", "?"), r.get("model", "?"))
            agg[key]["input_tokens"] += r.get("input_tokens", 0)
            agg[key]["output_tokens"] += r.get("output_tokens", 0)
            agg[key]["total_tokens"] += r.get("total_tokens", 0)
            agg[key]["cost_usd"] += r.get("cost_usd", 0.0)
            agg[key]["calls"] += 1

        table = Table(
            title=f"📊 TokenSaver Report — {title} — {now}",
            box=box.ROUNDED,
            show_footer=True,
        )
        table.add_column("Provider", style="cyan", footer="TOTAL")
        table.add_column("Model", style="white")
        table.add_column("Calls", style="dim", justify="right")
        table.add_column("Input Tokens", justify="right", style="green")
        table.add_column("Output Tokens", justify="right", style="yellow")
        table.add_column("Total Tokens", justify="right", style="bold white")
        table.add_column("Cost USD", justify="right", style="bold magenta",
                         footer=f"${sum(v['cost_usd'] for v in agg.values()):.4f}")

        total_input = total_output = total_tokens = 0
        for (provider, model), data in sorted(agg.items()):
            total_input += data["input_tokens"]
            total_output += data["output_tokens"]
            total_tokens += data["total_tokens"]
            table.add_row(
                provider,
                model,
                str(data["calls"]),
                f"{data['input_tokens']:,}",
                f"{data['output_tokens']:,}",
                f"{data['total_tokens']:,}",
                f"${data['cost_usd']:.4f}",
            )

        console.print(table)

        # Top session
        if records:
            top = max(records, key=lambda r: r.get("total_tokens", 0))
            ts = top.get("ts", "")[:16].replace("T", " ")
            console.print(
                f"[dim]💡 Top call: session [bold]{top.get('session')}[/bold] "
                f"at {ts} — {top.get('total_tokens', 0):,} tokens "
                f"(${top.get('cost_usd', 0):.4f})[/dim]"
            )

    def _print_plain(self, records: list[dict], title: str) -> None:
        """Fallback plain text report."""
        print(f"\n{'='*60}")
        print(f"TokenSaver Report — {title}")
        print(f"{'='*60}")
        if not records:
            print("No records found.")
            return
        total_tokens = sum(r.get("total_tokens", 0) for r in records)
        total_cost = sum(r.get("cost_usd", 0) for r in records)
        print(f"Records: {len(records)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(
        self, output_path: str | Path, records: Optional[list[dict]] = None
    ) -> Path:
        """Export all records (or supplied records) to a CSV file."""
        if records is None:
            records = self._logger.read_all()
        path = Path(output_path)
        if not records:
            path.write_text("No records.\n", encoding="utf-8")
            return path

        fieldnames = [
            "ts", "session", "tag", "provider", "model",
            "input_tokens", "output_tokens", "total_tokens", "cost_usd",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        return path

    # ------------------------------------------------------------------
    # Summary dict (for programmatic access)
    # ------------------------------------------------------------------

    def summary(self, records: Optional[list[dict]] = None) -> dict:
        """Return aggregated summary as a dict."""
        if records is None:
            records = self._logger.read_all()
        return {
            "total_calls": len(records),
            "total_input_tokens": sum(r.get("input_tokens", 0) for r in records),
            "total_output_tokens": sum(r.get("output_tokens", 0) for r in records),
            "total_tokens": sum(r.get("total_tokens", 0) for r in records),
            "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in records), 6),
        }
