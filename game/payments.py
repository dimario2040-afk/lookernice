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


def _payload(product_key: str, user_id: int, extra: tuple = ()) -> str:
    parts = ["mog", product_key, str(user_id), *[str(x) for x in extra]]
    parts.append(uuid.uuid4().hex[:10])
    return ":".join(parts)


def parse_payload(payload: str) -> tuple[str | None, int | None, list[str]]:
    """'mog:<product>:<uid>[:<extra>...]:<nonce>' → (product_key, user_id, extras)."""
    parts = (payload or "").split(":")
    if len(parts) >= 4 and parts[0] == "mog":
        try:
            return parts[1], int(parts[2]), parts[3:-1]
        except ValueError:
            return None, None, []
    return None, None, []


def validate_payload(payload: str) -> tuple[bool, str | None]:
    """Проверка payload для pre_checkout (ok, error_message)."""
    product_key, user_id, extras = parse_payload(payload)
    if not product_key or not user_id:
        return False, "Некорректный платёж."
    if product_key == "artifact":
        if len(extras) < 2:
            return False, "Артефакт не распознан — попробуй ещё раз."
        return True, None
    if product_key not in config.STARS_CATALOG:
        return False, "Товар не найден — попробуй ещё раз."
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


async def send_artifact_invoice(
    bot: Bot, chat_id: int, user_id: int, battle_id: int, item: str
) -> None:
    """Счёт на артефакт для баттла (payload несёт battle_id и item)."""
    from . import artifacts

    a = artifacts.ARTIFACTS[item]
    await bot.send_invoice(
        chat_id=chat_id,
        title=f"{a['emoji']} {a['name']}",
        description=f"Артефакт для МОГ-баттла #{battle_id}: {a['desc']}",
        payload=_payload("artifact", user_id, (battle_id, item)),
        currency=CURRENCY,
        prices=[LabeledPrice(label=a["name"], amount=int(a["stars"]))],
    )


async def grant(payload: str) -> dict | None:
    """Начисляет товар после успешной оплаты. Возвращает товар или None."""
    product_key, user_id, extras = parse_payload(payload)
    if not product_key or not user_id:
        return None
    await db.get_or_create_user(user_id)

    # 🎒 Артефакт для баттла: цепляем к бою сразу (без инвентаря).
    if product_key == "artifact":
        from . import artifacts

        try:
            battle_id, item = int(extras[0]), extras[1]
        except (IndexError, ValueError):
            return None
        if item not in artifacts.ARTIFACTS:
            return None
        await db.add_battle_artifact(battle_id, user_id, item, paid=True)
        a = artifacts.ARTIFACTS[item]
        log.info("grant artifact %s -> battle %s, user %s", item, battle_id, user_id)
        return {
            "title": f"{a['emoji']} {a['name']}",
            "artifact_key": item,
            "battle_id": battle_id,
        }

    product = config.STARS_CATALOG.get(product_key)
    if not product:
        return None

    # 🎁 Подарок: одноразовый токен, который юзер перешлёт другу.
    gift_kind = product.get("gift")
    if gift_kind:
        token = await db.create_gift_token(gift_kind, user_id)
        log.info("grant gift %s -> token (by %s)", gift_kind, user_id)
        return {**product, "gift_kind": gift_kind, "gift_token": token}

    await db.add_purchase(user_id, rolls=int(product.get("rolls", 0)), crit=int(product.get("crit", 0)))
    log.info("grant %s -> user %s (+%s rolls, +%s crit)", product_key, user_id, product.get("rolls", 0), product.get("crit", 0))
    return product


def grant_text(product: dict) -> str:
    if product.get("gift_token"):
        return "Подарок готов — перешли сообщение другу 👇"
    if product.get("artifact_key"):
        return (
            f"Артефакт прикреплён к бою #{product['battle_id']}. "
            "Буст учтётся в табло, когда соперник пришлёт фото."
        )
    bits = []
    if product.get("rolls"):
        bits.append(f"+{product['rolls']} роллов")
    if product.get("crit"):
        bits.append(f"+{product['crit']} крит-токен(ов)")
    tail = ", ".join(bits) if bits else "спасибо!"
    return f"Начислено: {tail}. Открывай MOG 🎰"
