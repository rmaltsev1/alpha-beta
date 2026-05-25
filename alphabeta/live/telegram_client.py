"""Telegram Bot API client with retries, rate limiting, and idempotent dispatch.

Used by the live signal pipeline to deliver paper-trade alerts to the user's chat.
Designed to be safe to import without network access (no calls in module load).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from alphabeta.config import settings


TELEGRAM_API = "https://api.telegram.org"
DEFAULT_TIMEOUT = 15.0
RETRY_BACKOFF = (1.0, 3.0, 7.0)


class TelegramError(RuntimeError):
    """Raised when Telegram API rejects our request."""


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        if not settings.telegram_bot or not settings.chat_id:
            raise RuntimeError(
                "TELEGRAM_BOT and CHAT_ID must be set in .env for live signal delivery"
            )
        return cls(bot_token=settings.telegram_bot, chat_id=settings.chat_id)


class TelegramClient:
    """Thin synchronous wrapper around the Bot API send-message endpoint.

    Adds: retries, idempotency dedup (via local SQLite), dry-run mode, log of sent messages.
    """

    def __init__(
        self,
        cfg: Optional[TelegramConfig] = None,
        *,
        dry_run: bool = False,
        log_db: Optional[Path] = None,
    ):
        self.cfg = cfg or TelegramConfig.from_env()
        self.dry_run = dry_run
        self.log_db = log_db or (settings.data_dir / "live" / "telegram_log.sqlite")
        self.log_db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.log_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_messages (
                    idempotency_key TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    parse_mode TEXT,
                    message_id INTEGER,
                    sent_ts REAL NOT NULL,
                    dry_run INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _already_sent(self, key: str) -> bool:
        """Dedup only against real (non-dry-run) sends.

        Dry-run records are useful for inspecting what would be sent, but they
        must not block a subsequent real send with the same key.
        """
        with sqlite3.connect(self.log_db) as conn:
            cur = conn.execute(
                "SELECT 1 FROM sent_messages WHERE idempotency_key = ? AND dry_run = 0",
                (key,),
            )
            return cur.fetchone() is not None

    def _record(
        self,
        key: str,
        text: str,
        parse_mode: Optional[str],
        message_id: Optional[int],
        dry_run: bool,
    ) -> None:
        with sqlite3.connect(self.log_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sent_messages "
                "(idempotency_key, chat_id, text, parse_mode, message_id, sent_ts, dry_run) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, self.cfg.chat_id, text, parse_mode, message_id, time.time(), int(dry_run)),
            )

    @staticmethod
    def make_key(*parts: str) -> str:
        """Build a deterministic idempotency key from parts (e.g., timeframe + bar timestamp + sleeve)."""
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    def send(
        self,
        text: str,
        *,
        parse_mode: Optional[str] = "MarkdownV2",
        idempotency_key: Optional[str] = None,
        disable_notification: bool = False,
    ) -> Optional[int]:
        """Send a message. Returns Telegram message_id, or None if dry-run / deduped.

        Raises TelegramError after exhausting retries on transient API failures.
        """
        key = idempotency_key or self.make_key(text)
        if self._already_sent(key):
            return None  # idempotent no-op

        if self.dry_run:
            print(f"[DRY-RUN telegram] {text}")
            self._record(key, text, parse_mode, None, dry_run=True)
            return None

        params = {
            "chat_id": self.cfg.chat_id,
            "text": text,
            "disable_notification": "true" if disable_notification else "false",
        }
        if parse_mode:
            params["parse_mode"] = parse_mode

        url = f"{TELEGRAM_API}/bot{self.cfg.bot_token}/sendMessage"
        data = urllib.parse.urlencode(params).encode()

        last_err: Optional[Exception] = None
        for backoff in RETRY_BACKOFF:
            try:
                req = urllib.request.Request(url, data=data, method="POST")
                with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as r:
                    resp = json.loads(r.read())
                if not resp.get("ok"):
                    raise TelegramError(f"Telegram refused: {resp}")
                msg_id = resp.get("result", {}).get("message_id")
                self._record(key, text, parse_mode, msg_id, dry_run=False)
                return msg_id
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
                # 429 = rate limited; 5xx = transient; 4xx other = give up
                if e.code == 429 or e.code >= 500:
                    last_err = TelegramError(f"HTTP {e.code}: {body}")
                    time.sleep(backoff)
                    continue
                raise TelegramError(f"HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(backoff)
                continue

        raise TelegramError(f"exhausted retries: {last_err!r}")

    def get_me(self) -> dict:
        url = f"{TELEGRAM_API}/bot{self.cfg.bot_token}/getMe"
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT) as r:
            return json.loads(r.read())


def escape_markdown_v2(text: str) -> str:
    """Escape characters reserved by Telegram MarkdownV2."""
    reserved = r"_*[]()~`>#+-=|{}.!"
    return "".join("\\" + c if c in reserved else c for c in text)
