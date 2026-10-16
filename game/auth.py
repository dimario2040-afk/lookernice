"""Проверка Telegram.WebApp.initData (HMAC-SHA256 с токеном бота).

Алгоритм Telegram:
    secret_key = HMAC_SHA256(key="WebAppData", msg=bot_token)
    hash       = HMAC_SHA256(key=secret_key,  msg=data_check_string)
где data_check_string — все поля кроме `hash`, отсортированные по имени,
склеенные через `\\n` как `key=value`.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl


def verify_init_data(init_data: str, bot_token: str, max_age: int = 86400) -> dict | None:
    """Возвращает {"user": {...}, "auth_date": int} или None, если подпись/свежесть плохие."""
    if not init_data or not bot_token:
        return None

    try:
        pairs = dict(parse_qsl(init_data, strict_parsing=True))
    except ValueError:
        return None

    received_hash = pairs.pop("hash", None)
    if not received_hash:
        return None

    data_check_string = "\n".join(f"{k}={pairs[k]}" for k in sorted(pairs))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calc_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calc_hash, received_hash):
        return None

    try:
        auth_date = int(pairs.get("auth_date", "0"))
    except ValueError:
        return None

    # Свежесть (защита от переигрывания старого initData).
    if max_age and auth_date and (time.time() - auth_date) > max_age:
        return None

    user = None
    if "user" in pairs:
        try:
            user = json.loads(pairs["user"])
        except json.JSONDecodeError:
            user = None

    return {"user": user, "auth_date": auth_date}
