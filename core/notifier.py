"""Signal delivery: console + optional Telegram.

No third-party dependency. Telegram is reached via its HTTP Bot API using the
standard library, and any failure degrades to console-only so a network blip
never stops the signal service.

Setup (Telegram):
  1. Message @BotFather, send /newbot, copy the token -> TELEGRAM_BOT_TOKEN.
  2. Message your new bot once (say "hi"), then open
     https://api.telegram.org/bot<token>/getUpdates and copy the chat id ->
     TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from config.settings import Config

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.telegram_enabled = bool(config.telegram_bot_token and config.telegram_chat_id)
        if not self.telegram_enabled:
            logger.info("Telegram not configured — signals will print to console only.")

    def send(self, text: str) -> None:
        """Print to console always, and push to Telegram if configured."""
        # Console (the log) is the always-on channel.
        for line in text.splitlines():
            logger.info(line)
        if self.telegram_enabled:
            self._send_telegram(text)

    def _send_telegram(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": self.config.telegram_chat_id, "text": text, "disable_web_page_preview": "true"}
        ).encode()
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "crypto-agent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
            if not body.get("ok"):
                logger.warning("Telegram API returned not-ok: %s", body)
        except Exception as exc:
            logger.warning("Telegram send failed (%s); signal still logged to console", exc)
