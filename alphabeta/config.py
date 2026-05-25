"""Centralized config loader. Reads .env once."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    # postgres (via tunnel)
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "15432"))
    db_user: str = os.getenv("DB_USER", "rektfree")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "rektfree")

    # ssh tunnel target (only the script needs these, but we expose for diag)
    vm_host: str = os.getenv("VM_HOST", "62.238.28.3")
    vm_user: str = os.getenv("VM_USER", "deploy")
    ssh_key: str = os.path.expanduser(os.getenv("SSH_KEY", "~/.ssh/id_ed25519"))
    local_pg_port: int = int(os.getenv("LOCAL_PG_PORT", "15432"))

    # oanda (api fallback)
    oanda_api_key: str = os.getenv("OANDA_API_KEY", "")
    oanda_account_id: str = os.getenv("OANDA_ACCOUNT_ID", "")
    oanda_base_url: str = os.getenv("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")

    # local storage root (resolved relative to repo if not absolute)
    data_dir: Path = (REPO_ROOT / os.getenv("DATA_DIR", "data")).resolve()

    # telegram signal delivery (paper trading)
    telegram_bot: str = os.getenv("TELEGRAM_BOT", "")
    chat_id: str = os.getenv("CHAT_ID", "")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
