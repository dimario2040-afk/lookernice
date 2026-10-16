"""Telegram Stars платежи MOG (Bot API: XTR).

Поток:
  1. UI/бот → create_invoice_link / send_invoice (payload = mog:<product>:<uid>:<nonce>)
  2. Юзер платит → приходит pre_checkout_query → отвечаем ok=True
  3. Приходит message.successful_payment → начисляем товар по payload
"""
from __future__ import annotations

import logging
import uuid

from aiogram import Bot
from aiogram.types import LabeledPrice

from . import config, db

log = logging.getLogger("mog-pay")

CURRENCY = "XTR"  # Telegram Stars


def _prices(product: dict) -> list[LabeledPrice]:
    return [LabeledPrice(label=product["title"], amount=int(product["stars"]))]


def _payload(product_key: str, user_id: int) -> str:
    return f"mog:{product_key}:{user_id}:{uuid.uuid4().hex[:10]}"


def parse_payload(payload: str) -> tuple[str | None, int | None]:
    """'mog:<product>:<uid>:<nonce>' → (product_key, user_id)."""
    parts = (payload or "").split(":")
    if len(parts) >= 3 and parts[0] == "mog":
        try:
            return parts[1], int(parts[2])
        except ValueError:
            return None, None
    return None, None


def validate_payload(payload: str) -> tuple[bool, str | None]:
    """Проверка payload для pre_checkout (ok, error_message)."""
    product_key, user_id = parse_payload(payload)
    if not product_key or product_key not in config.STARS_CATALOG:
        return False, "Товар не найден — попробуй ещё раз."
    if not user_id:
        return False, "Некорректный платёж."
    return True, None


async def create_link(bot: Bot, product_key: str, user_id: int) -> str:
    """Ссылка на оплату для Mini App (Telegram.WebApp.openInvoice)."""
    product = config.STARS_CATALOG[product_key]
    return await bot.create_invoice_link(
        title=product["title"],
        description="MOG — развлекательная гача 18+",
        payload=_payload(product_key, user_id),
        currency=CURRENCY,
        prices=_prices(product),
    )


async def send_invoice(bot: Bot, chat_id: int, product_key: str, user_id: int) -> None:
    """Счёт прямо в чат (для кнопки в боте)."""
    product = config.STARS_CATALOG[product_key]
    await bot.send_invoice(
        chat_id=chat_id,
        title=product["title"],
        description="MOG — развлекательная гача 18+",
        payload=_payload(product_key, user_id),
        currency=CURRENCY,
        prices=_prices(product),
    )


async def grant(payload: str) -> dict | None:
    """Начисляет товар после успешной оплаты. Возвращает товар или None."""
    product_key, user_id = parse_payload(payload)
    if not product_key or not user_id:
        return None
    product = config.STARS_CATALOG.get(product_key)
    if not product:
        return None
    await db.get_or_create_user(user_id)
    await db.add_purchase(user_id, rolls=int(product.get("rolls", 0)), crit=int(product.get("crit", 0)))
    log.info("grant %s -> user %s (+%s rolls, +%s crit)", product_key, user_id, product.get("rolls", 0), product.get("crit", 0))
    return product


def grant_text(product: dict) -> str:
    bits = []
    if product.get("rolls"):
        bits.append(f"+{product['rolls']} роллов")
    if product.get("crit"):
        bits.append(f"+{product['crit']} крит-токен(ов)")
    tail = ", ".join(bits) if bits else "спасибо!"
    return f"Начислено: {tail}. Открывай MOG 🎰"
