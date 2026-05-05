"""
nexus/interfaces/telegram.py
────────────────────────────
Telegram bot interface — Stubbed for Phase 3 integration.
"""
from typing import Any
from nexus.interfaces.formatter import ResponseFormatter


class TelegramInterface:
    """
    Stub for Telegram integration.
    Will use ResponseFormatter to ensure visual consistency with CLI.
    """
    def __init__(self):
        self.formatter = ResponseFormatter()

    def send_response(self, task: Any):
        """
        In Phase 3, this will map Formatter blocks to Telegram UI elements.
        """
        pass


def run_telegram_bot() -> None:
    raise NotImplementedError(
        "Telegram interface is scheduled for Phase 3. "
        "Core CLI and Jarvis personality must be verified first."
    )
