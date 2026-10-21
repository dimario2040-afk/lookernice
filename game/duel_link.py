"""Дуэли по ссылке: вызов без группы, бой в личке, итог обоим в личку.

Схема: вызывающий жмёт «⚔️ Сразиться» под карточкой → бот выдаёт ссылку
t.me/бот?start=duel_<код> → соперник жмёт, попадает в бота, принимает вызов,
кидает фото → бот считает баллы и пишет табло обоим.

Отличие от battles: там бой живёт в чате (duel решается по фото в том же чате),
здесь чата нет — всё происходит в личке с ботом, поэтому своя таблица.
"""
from __future__ import annotations

import secrets

from . import config
from . import db


def new_code() -> str:
    """Короткий URL-безопасный код вызова (8 симв. — 48 бит энтропии)."""
    return secrets.token_urlsafe(6)[:8]


async def create(user: dict, score: float) -> dict:
    """Создаёт вызов от user на его текущий балл. Возвращает строку duel_links."""
    code = new_code()
    await db.create_duel_link(
        code=code,
        challenger_id=user["id"],
        challenger_name=user["name"],
        challenger_score=score,
        ttl_hours=config.DUEL_LINK_TTL_HOURS,
    )
    assert await db.get_duel_link(code) is not None
    return await db.get_duel_link(code)  # type: ignore[return-value]


async def accept(code: str, user: dict) -> tuple[bool, str]:
    """Соперник принимает вызов. Возвращает (ok, reason)."""
    duel = await db.get_duel_link(code)
    if not duel:
        return False, "not_found"
    if duel["status"] == "expired":
        return False, "expired"
    if duel["status"] != "open":
        return False, "closed"
    if duel["challenger_id"] == user["id"]:
        return False, "self"
    return await db.accept_duel_link(code, user["id"], user["name"])


async def resolve_for(user_id: int, opponent_score: float) -> dict | None:
    """Если у юзера есть принятый вызов — считает бой и возвращает табло.

    Возвращает None, если боя нет (обычное фото в личке).
    """
    duel = await db.pending_duel_link_for(user_id)
    if not duel:
        return None

    cid = int(duel["challenger_id"])
    cs = float(duel["challenger_score"] or 0.0)
    os_ = float(opponent_score or 0.0)
    if cs > os_:
        winner = cid
    elif os_ > cs:
        winner = user_id
    else:
        winner = None
    await db.resolve_duel_link(duel["code"], os_, winner)
    return {
        "code": duel["code"],
        "challenger_id": cid,
        "challenger_name": duel["challenger_name"],
        "challenger_score": cs,
        "opponent_id": user_id,
        "opponent_name": duel["opponent_name"],
        "opponent_score": os_,
        "winner_id": winner,
    }


def _mention(user_id: int, name: str) -> str:
    import html as _html

    # Telegram allows '<' and '&' in names: unescaped they killed the whole
    # invite/result message ("can't parse entities") and nobody saw the duel.
    safe = _html.escape(name or "", quote=False)
    return f'<a href="tg://user?id={user_id}">{safe}</a>'


def invite_text(duel: dict, bot_username: str) -> tuple[str, str]:
    """Текст приглашения для пересылки. Возвращает (текст, ссылка)."""
    link = f"https://t.me/{bot_username}?start=duel_{duel['code']}"
    hours = config.DUEL_LINK_TTL_HOURS
    text = (
        f'⚔️ <b>{_mention(duel["challenger_id"], duel["challenger_name"])}</b> вызывает тебя на дуэль!\n\n'
        f"Его МОГ: <b>{duel['challenger_score']:.4f}</b>\n\n"
        "Жми ссылку — попадёшь в бота, примешь вызов и кинешь своё фото. "
        "Кто выше по МОГ — тот и победил.\n"
        "В конце оба получат карточку с табло: два балла, победитель, разрыв.\n\n"
        f"{link}\n\n"
        f'<i>Вызов живёт {hours} ч. Не явишься — он сгорит.</i>'
    )
    return text, link


def accepted_text(duel: dict) -> str:
    return (
        f'⚔️ Ты принял вызов от {_mention(duel["challenger_id"], duel["challenger_name"])}.\n'
        f'Его МОГ: <b>{duel["challenger_score"]:.4f}</b>\n\n'
        "📸 <b>Что кидать:</b> ответь <b>на это сообщение</b> своим фото — "
        "или просто пришли фото, я жду именно от тебя.\n"
        "🎁 <b>Что получишь:</b> карточку с твоим МОГ и табло боя на картинке — "
        "оба балла, кто победил и с каким разрывом."
    )


def result_text(r: dict) -> str:
    """Табло боя + поздравление победителя. Одинаково для обоих."""
    c = _mention(r["challenger_id"], r["challenger_name"])
    o = _mention(r["opponent_id"], r["opponent_name"])
    cs, os_ = r["challenger_score"], r["opponent_score"]
    if r["winner_id"] is None:
        verdict = "🤝 <b>НИЧЬЯ</b> — оба на месте. Так себе мог."
    elif r["winner_id"] == r["challenger_id"]:
        verdict = f"👑 <b>ПОБЕДА: {c}</b>"
    else:
        verdict = f"👑 <b>ПОБЕДА: {o}</b>"
    return (
        f"⚔️ <b>ДУЭЛЬ ЗАВЕРШЕНА</b>\n"
        f"┌───────── ⚔️ ─────────┐\n"
        f"│  {cs:.4f}  │  {os_:.4f}  │\n"
        f"└────────────────────────┘\n"
        f"{c} — вызывающий · {o} — принимавший\n\n"
        f"{verdict}\n\n"
        '<i>MOG — игра. Баллы случайны, не оценка внешности.</i>'
    )


def opponent_joined_text(duel: dict) -> str:
    """Уведомление ВЫЗЫВАЮЩЕМУ, что соперник принял и ждёт фото."""
    return (
        f'⚔️ <b>{_mention(duel["opponent_id"], duel["opponent_name"])}</b> '
        "принял твой вызов и кидает фото 👀\n\n"
        "Тебе присылать ничего не нужно — твой балл "
        f'<b>{duel["challenger_score"]:.4f}</b> уже в игре.\n'
        "Как придёт его фото — выдам табло обоим."
    )


def expired_text(duel: dict) -> str:
    """Уведомление ВЫЗЫВАЮЩЕМУ, что никто не пришёл."""
    return (
        f"⌛ <b>Соперник слился.</b>\n\n"
        f'Вызов на {duel["challenger_score"]:.4f} провисел '
        f'{config.DUEL_LINK_TTL_HOURS} ч без ответа.\n'
        "Ссылка больше не работает. Хочешь ещё — создай новый вызов ⚔️"
    )