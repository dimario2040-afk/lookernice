"""MOG-АРЕНА: чатовая битва до 8 игроков (Royale) — без мини-аппа и без HTTPS.

Поток: /arena → лобби с кнопкой «Я в деле» → старт (любой участник при ≥2) →
один ранкинг по ЛУЧШЕМУ роллу каждого. Скоринг детерминирован по фото, поэтому
честно и стабильно (без «повезло на рандоме»).
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from . import db

MIN_PLAYERS = 2
MAX_PLAYERS = 8
LOBBY_TTL = 10 * 60  # секунд: протухшее лобби не блокирует новые

_MEDALS = ["🥇", "🥈", "🥉"]


def _fighters(n: int) -> str:
    """Русское склонение: 1 боец / 2 бойца / 5 бойцов."""
    if n % 10 == 1 and n % 100 != 11:
        return "боец"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "бойца"
    return "бойцов"


def _esc(s: str) -> str:
    return escape(s or "", quote=False)


def _age_seconds(created_at: str) -> float:
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return 0.0


async def create(chat_id: int, host: dict) -> tuple[dict | None, bool]:
    """Открывает лобби. Живое уже есть → возвращает его (created=False)."""
    active = await db.get_active_arena(chat_id)
    if active:
        if _age_seconds(active["created_at"]) <= LOBBY_TTL:
            return active, False
        await db.set_arena_status(active["id"], "expired")  # протухшее закрываем
    arena_id = await db.create_arena(chat_id, host["id"], host["name"], MAX_PLAYERS)
    return await db.get_arena(arena_id), True


async def players(arena_id: int) -> list[dict]:
    return await db.arena_players(arena_id)


async def join(arena_id: int, user: dict) -> tuple[bool, str]:
    """Вступление по ЛУЧШЕМУ роллу юзера. Без ролла — просим сначала кинуть фото."""
    arena = await db.get_arena(arena_id)
    if not arena or arena["status"] != "lobby":
        return False, "Это лобби уже закрыто."
    best = await db.get_best_roll(user["id"])
    if not best:
        return False, "Сначала кинь фото в чат — получишь свой МОГ, потом жми 👇"
    if await db.arena_count(arena_id) >= arena["max_players"]:
        return False, "Мест нет — стартуем без тебя 😈"
    added = await db.add_arena_player(
        arena_id, user["id"], user["name"], best["score"], best["tier"]
    )
    return (True, "Ты в деле ⚔️") if added else (False, "Ты уже в деле.")


async def leave(arena_id: int, user_id: int) -> bool:
    return await db.remove_arena_player(arena_id, user_id)


async def resolve(arena_id: int) -> list[dict]:
    """Закрывает арену и возвращает финальный ранкинг (уже по score DESC)."""
    await db.set_arena_status(arena_id, "done")
    return await db.arena_players(arena_id)


def lobby_text(arena: dict, players: list[dict]) -> str:
    n, cap = len(players), arena["max_players"]
    names = ", ".join(_esc(p["name"]) for p in players) or "пока никого"
    return (
        f"⚔️ <b>МОГ-АРЕНА #{arena['id']}</b> · {n}/{cap}\n"
        f"Хост: <b>{_esc(arena['host_name'])}</b>\n\n"
        f"Бойцы: {names}\n\n"
        "Жми «Я в деле» 👇 Заходишь со своим лучшим МОГом.\n"
        f"<i>Старт — как соберётся ≥{MIN_PLAYERS} и кто-то нажмёт «Старт».</i>"
    )


def lobby_keyboard(arena: dict, players: list[dict]):
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    n, cap = len(players), arena["max_players"]
    rows = [
        [
            InlineKeyboardButton(
                text=f"⚔️ Я в деле ({n}/{cap})",
                callback_data=f"arena_join:{arena['id']}",
            )
        ]
    ]
    if n >= MIN_PLAYERS:
        rows.append(
            [
                InlineKeyboardButton(
                    text="▶️ Старт", callback_data=f"arena_start:{arena['id']}"
                ),
                InlineKeyboardButton(
                    text="🚪 Выйти", callback_data=f"arena_leave:{arena['id']}"
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🚪 Выйти", callback_data=f"arena_leave:{arena['id']}"
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def result_text(arena: dict, players: list[dict]) -> str:
    ranked = sorted(players, key=lambda p: p["score"], reverse=True)
    if not ranked:
        return f"🏆 <b>МОГ-АРЕНА #{arena['id']}</b> — никто не зашёл 😔"
    lines = []
    for i, p in enumerate(ranked):
        medal = _MEDALS[i] if i < len(_MEDALS) else "▫️"
        tag = f'<a href="tg://user?id={p["user_id"]}">{_esc(p["name"])}</a>'
        lines.append(f'{medal} <b>{p["score"]:.4f}</b> · {p["tier"]} — {tag}')
    win = ranked[0]
    return (
        f"🏆 <b>МОГ-АРЕНА #{arena['id']}</b> · {len(ranked)} {_fighters(len(ranked))}\n\n"
        + "\n".join(lines)
        + f'\n\n👑 <b>Корона: {_esc(win["name"])}</b> · {win["score"]:.4f}'
        + "\n\n<i>MOG — игра. Баллы случайны, не оценка внешности.</i>"
    )


def rematch_keyboard(arena_id: int):
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔁 Новая арена", callback_data=f"arena_new:{arena_id}"
                )
            ]
        ]
    )
