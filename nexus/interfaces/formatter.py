"""
nexus/interfaces/formatter.py
────────────────────────────
Visual response formatter layer using Rich for premium terminal aesthetics.
Provides graceful fallback if Rich is not installed.
"""
import sys
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ResponseFormatter:
    """
    Handles all terminal output formatting. 
    Zero business logic — purely presentation.
    """
    def __init__(self):
        if RICH_AVAILABLE:
            self.theme = Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "bold green",
                "jarvis": "bold blue",
                "task": "bold magenta",
                "code": "bold cyan",
                "dim": "grey50"
            })
            self.console = Console(theme=self.theme)
        else:
            self.console = None

    def print_welcome(self):
        """Header displayed at session start."""
        msg = "NEXUS OS v2.0 | Jarvis Protocol | Local-First AI"
        if RICH_AVAILABLE:
            self.console.print("\n")
            self.console.print(Panel(
                Text(msg, justify="center", style="jarvis"),
                border_style="jarvis",
                expand=True
            ))
            self.console.print("[dim]Type your command below. Type 'exit' to disconnect.[/dim]\n")
        else:
            print(f"\n--- {msg} ---")

    def print_intent_identified(self, intent: str, confidence: float):
        """Show what the system understood before processing."""
        if RICH_AVAILABLE:
            self.console.print(f"[dim]Intent identified: [bold]{intent}[/bold] ({confidence:.2f} confidence)[/dim]")
        else:
            print(f"> Intent: {intent} ({confidence:.2f})")

    def format_chat(self, content: str):
        """Render Jarvis conversational responses."""
        if RICH_AVAILABLE:
            self.console.print("\n")
            self.console.print(Panel(
                Markdown(content),
                title="NEXUS",
                title_align="left",
                border_style="jarvis",
                padding=(1, 2)
            ))
            self.console.print("\n")
        else:
            print(f"\nNEXUS: {content}\n")

    def format_task_result(self, task: Any):
        """Render completion summary for a TASK intent."""
        summary = task.result.summary if task.result else "Task completed with no summary."
        if RICH_AVAILABLE:
            style = "success" if task.status.name == "DONE" else "error"
            title = "Task Success" if task.status.name == "DONE" else "Task Failure"
            self.console.print(Panel(
                summary,
                title=title,
                border_style=style,
                padding=(1, 2)
            ))
            if hasattr(task.result, "output_path") and task.result.output_path:
                self.console.print(f"[dim]Output saved to: {task.result.output_path}[/dim]")
        else:
            print(f"\n[{task.status.name}] {summary}\n")

    def format_code_intel(self, summary: str):
        """Render CODE intent results (explain, refactor, etc)."""
        if RICH_AVAILABLE:
            self.console.print("\n")
            self.console.print(Panel(
                Markdown(summary),
                title="Code Intelligence",
                border_style="code",
                padding=(1, 2)
            ))
            self.console.print("\n")
        else:
            print(f"\n--- Code Intelligence ---\n{summary}\n")

    def print_error(self, message: str, suggestion: str = ""):
        """Render system or execution errors."""
        if RICH_AVAILABLE:
            err_text = Text(f"❌ {message}\n", style="error")
            if suggestion:
                err_text.append(f"\n💡 Suggestion: {suggestion}", style="info")
            self.console.print(Panel(err_text, border_style="error", title="Error"))
        else:
            print(f"\nERROR: {message}")
            if suggestion:
                print(f"SUGGESTION: {suggestion}\n")

    def step(self, message: str, status: str = "pending"):
        """Show progress steps in the pipeline."""
        if RICH_AVAILABLE:
            if status == "pending":
                icon = "[yellow]⏳[/yellow]"
                style = "info"
            elif status == "done":
                icon = "[green]✅[/green]"
                style = "success"
            else:
                icon = "[red]❌[/red]"
                style = "error"
            self.console.print(f"{icon} [dim]{message}[/dim]", style=style)
        else:
            prefix = "[...]" if status == "pending" else "[OK ]" if status == "done" else "[ERR]"
            print(f"{prefix} {message}")
