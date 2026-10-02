"""
TokenSaver CLI -- Main entrypoint.

Commands:
  tokensaver count    -- Count tokens in a file or stdin
  tokensaver compress -- Compress a prompt to save tokens
  tokensaver clean    -- Convert HTML to clean Markdown
  tokensaver report   -- Show usage/cost reports
  tokensaver status   -- Show current session budget status
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0", prog_name="tokensaver")
def cli():
    """TokenSaver -- Token optimization toolkit for AI projects."""
    pass


# ---------------------------------------------------------------------------
# count command
# ---------------------------------------------------------------------------

@cli.command("count")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--stdin", "use_stdin", is_flag=True, help="Read text from stdin (pipe)")
@click.option("--provider", "-p", default="anthropic", show_default=True,
              help="AI provider: anthropic | openai | gemini")
@click.option("--model", "-m", default="claude-sonnet-4-5", show_default=True,
              help="Model name for accurate token counting")
@click.option("--output-tokens", "-o", default=0, type=int,
              help="Expected output tokens for cost estimation")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def count_cmd(
    file: Optional[str],
    use_stdin: bool,
    provider: str,
    model: str,
    output_tokens: int,
    as_json: bool,
):
    """Count tokens in a FILE or stdin.

    \b
    Examples:
      tokensaver count prompt.txt
      tokensaver count prompt.txt --provider openai --model gpt-4o-mini
      cat prompt.txt | tokensaver count --stdin
    """
    from tokensaver import TokenCounter

    if use_stdin or (not file):
        if sys.stdin.isatty() and not use_stdin:
            raise click.UsageError("Provide a FILE or use --stdin to pipe text.")
        text = sys.stdin.read()
    else:
        text = Path(file).read_text(encoding="utf-8")

    counter = TokenCounter(provider=provider, model=model)
    result = counter.count(text)
    cost = counter._cost(result.tokens, output_tokens)

    if as_json:
        import json
        click.echo(json.dumps({
            "input_tokens": result.tokens,
            "output_tokens": output_tokens,
            "total_tokens": result.tokens + output_tokens,
            "cost_usd": cost,
            "provider": provider,
            "model": model,
        }))
        return

    console.print("\n[bold cyan]>> Token Count[/bold cyan]")
    console.print(f"  Provider  : {provider}")
    console.print(f"  Model     : {model}")
    console.print(f"  Input     : [green]{result.tokens:,}[/green] tokens")
    if output_tokens:
        console.print(f"  Output    : [yellow]{output_tokens:,}[/yellow] tokens (estimated)")
    console.print(f"  Est. cost : [magenta]${cost:.6f}[/magenta]")

    if result.tokens > 32_000:
        console.print(f"\n[yellow]WARNING: Large prompt ({result.tokens:,} tokens). Consider compression.[/yellow]")
    elif result.tokens > 8_000:
        console.print(f"\n[dim]TIP: Run 'tokensaver compress' to reduce this by 10-60%.[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# compress command
# ---------------------------------------------------------------------------

@cli.command("compress")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--stdin", "use_stdin", is_flag=True)
@click.option("--mode", type=click.Choice(["simple", "llmlingua"]), default="simple",
              show_default=True, help="Compression mode")
@click.option("--ratio", "-r", default=0.5, type=float,
              help="Target ratio for llmlingua (0.5 = keep 50%)")
@click.option("--provider", "-p", default="anthropic", show_default=True)
@click.option("--model", "-m", default="claude-sonnet-4-5", show_default=True)
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save compressed output to file (default: print to stdout)")
@click.option("--dry-run", is_flag=True, help="Show stats only, do not print compressed text")
def compress_cmd(file, use_stdin, mode, ratio, provider, model, output, dry_run):
    """Compress a prompt to reduce token usage.

    \b
    Examples:
      tokensaver compress prompt.txt
      tokensaver compress prompt.txt --mode llmlingua --ratio 0.4
      tokensaver compress prompt.txt --output compressed.txt
    """
    from tokensaver import PromptCompressor

    if use_stdin or not file:
        if sys.stdin.isatty() and not use_stdin:
            raise click.UsageError("Provide a FILE or use --stdin.")
        text = sys.stdin.read()
    else:
        text = Path(file).read_text(encoding="utf-8")

    console.print(f"[dim]Compressing with mode={mode}...[/dim]")

    compressor = PromptCompressor(mode=mode, target_ratio=ratio, provider=provider, model=model)
    result = compressor.compress(text)

    console.print("\n[bold cyan]>> Compression Result[/bold cyan]")
    console.print(f"  Original  : {result.original_tokens:,} tokens")
    console.print(f"  Compressed: [green]{result.compressed_tokens:,}[/green] tokens")
    console.print(f"  Saved     : [green]{result.saved_tokens:,} tokens ({result.saved_percent:.1f}%)[/green]")

    if not dry_run:
        if output:
            Path(output).write_text(result.compressed_text, encoding="utf-8")
            console.print(f"  Saved to  : {output}")
        else:
            console.print("\n[dim]--- Compressed Output ---[/dim]")
            click.echo(result.compressed_text)
    console.print()


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------

@cli.command("clean")
@click.argument("file", type=click.Path(exists=True), required=False)
@click.option("--url", "-u", default=None, help="Fetch and clean a URL")
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--keep-links", is_flag=True, default=True)
@click.option("--keep-images", is_flag=True, default=False)
@click.option("--stats", is_flag=True, help="Show token reduction stats")
def clean_cmd(file, url, output, keep_links, keep_images, stats):
    """Convert HTML to clean Markdown (removes nav, scripts, ads).

    \b
    Examples:
      tokensaver clean page.html
      tokensaver clean --url https://docs.example.com/api
      tokensaver clean page.html --output clean.md --stats
    """
    from tokensaver import HTMLCleaner

    cleaner = HTMLCleaner(keep_images=keep_images, keep_links=keep_links)

    if url:
        console.print(f"[dim]Fetching {url}...[/dim]")
        try:
            markdown = cleaner.clean_url(url)
        except Exception as e:
            console.print(f"[red]Error fetching URL: {e}[/red]")
            sys.exit(1)
    elif file:
        html = Path(file).read_text(encoding="utf-8")
        if stats:
            info = cleaner.token_reduction_estimate(html)
            console.print("\n[bold cyan]>> HTML Cleaning Stats[/bold cyan]")
            console.print(f"  Original  : {info['original_chars']:,} chars")
            console.print(f"  Cleaned   : {info['cleaned_chars']:,} chars")
            console.print(f"  Reduction : [green]{info['estimated_reduction_percent']}%[/green]\n")
        markdown = cleaner.clean(html)
    else:
        raise click.UsageError("Provide a FILE or --url.")

    if output:
        Path(output).write_text(markdown, encoding="utf-8")
        console.print(f"[green]OK[/green] Saved to {output}")
    else:
        click.echo(markdown)


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option("--today", "period", flag_value="today", default=True)
@click.option("--last", "period", flag_value="last")
@click.option("--all", "period", flag_value="all")
@click.option("--days", "-d", default=7, show_default=True,
              help="Number of days for --last")
@click.option("--tag", "-t", default=None, help="Filter by project tag")
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "json"]),
              default="table", show_default=True)
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save CSV/JSON to file")
def report_cmd(period, days, tag, fmt, output):
    """Show usage and cost reports.

    \b
    Examples:
      tokensaver report
      tokensaver report --last --days 7
      tokensaver report --all --tag my-project
      tokensaver report --last --format csv --output report.csv
    """
    from tokensaver import SessionLogger, Reporter

    logger = SessionLogger()
    reporter = Reporter(logger)

    if period == "today":
        records = logger.read_today()
        title = "Today"
    elif period == "last":
        records = logger.read_last_n_days(days)
        title = f"Last {days} days"
    else:
        records = logger.read_all()
        title = "All time"

    if tag:
        records = logger.filter_by_tag(records, tag)
        title += f" (tag: {tag})"

    if fmt == "csv":
        path = output or "tokensaver_report.csv"
        reporter.export_csv(path, records)
        console.print(f"[green]OK[/green] CSV saved to {path}")
    elif fmt == "json":
        import json
        data = reporter.summary(records)
        if output:
            Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print(f"[green]OK[/green] JSON saved to {output}")
        else:
            click.echo(json.dumps(data, indent=2))
    else:
        reporter._print_report(records, title)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@cli.command("status")
@click.option("--provider", "-p", default="anthropic", show_default=True)
@click.option("--model", "-m", default="claude-sonnet-4-5", show_default=True)
def status_cmd(provider, model):
    """Show today's token budget and remaining allowance.

    \b
    Examples:
      tokensaver status
      tokensaver status --provider openai --model gpt-4o-mini
    """
    from tokensaver import SessionLogger, BudgetGuard
    from tokensaver.pricing import PricingTable

    logger = SessionLogger()
    records = logger.read_today()

    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_tokens = total_input + total_output
    total_cost = sum(r.get("cost_usd", 0) for r in records)

    pt = PricingTable()
    price = pt.get(provider, model)

    console.print("\n[bold cyan]>> TokenSaver -- Today's Status[/bold cyan]")
    console.print(f"  API calls    : {len(records)}")
    console.print(f"  Input tokens : [green]{total_input:,}[/green]")
    console.print(f"  Output tokens: [yellow]{total_output:,}[/yellow]")
    console.print(f"  Total tokens : {total_tokens:,}")
    console.print(f"  Total cost   : [magenta]${total_cost:.4f}[/magenta]")

    if price:
        console.print(f"\n[dim]Pricing ({provider}/{model}):[/dim]")
        console.print(f"  [dim]Input:  ${price.input_per_1m:.2f} / 1M tokens[/dim]")
        console.print(f"  [dim]Output: ${price.output_per_1m:.2f} / 1M tokens[/dim]")

    guard = BudgetGuard(on_warning=lambda u: None, on_exceed=lambda u: None)
    guard._used_tokens = total_tokens
    guard._used_usd = total_cost
    usage = guard.usage
    if usage.percent is not None:
        pct = usage.percent * 100
        bar_len = 30
        filled = int(bar_len * usage.percent)
        color = "green" if pct < 60 else ("yellow" if pct < 80 else "red")
        bar = f"[{color}]{'#' * filled}[/{color}]{'.' * (bar_len - filled)}"
        console.print(f"\n  Budget: [{color}]{pct:.1f}%[/{color}] {bar}")
        console.print(
            f"  [dim]Default session limits: "
            f"{BudgetGuard.DEFAULT_SESSION_TOKENS:,} tokens / "
            f"${BudgetGuard.DEFAULT_SESSION_USD:.2f}[/dim]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
