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


# ------------------------------------------------------------------ плей-офф
def _seed_positions(n: int) -> list[int]:
    """Стандартная расстановка сеяных в сетке размера n (степень 2)."""
    pos = [1]
    while len(pos) < n:
        m = len(pos) * 2
        nxt: list[int] = []
        for p in pos:
            nxt.append(p)
            nxt.append(m + 1 - p)
        pos = nxt
    return pos


def bracket(players: list[dict]) -> dict:
    """Single-elim сетка от лучшего сеяного. Победа — по score (детерминированно).

    Возвращает {"rounds": [[(a, b, winner), ...], ...], "champion": player|None}.
    Игроки с одинаковым score — побеждает тот, кто выше в сеянии (стабильность).
    """
    seeds = sorted(players, key=lambda p: p["score"], reverse=True)
    if not seeds:
        return {"rounds": [], "champion": None}
    if len(seeds) == 1:
        return {"rounds": [], "champion": seeds[0]}

    n = 1
    while n < len(seeds):
        n *= 2
    field: list[dict | None] = [
        seeds[pos - 1] if pos <= len(seeds) else None for pos in _seed_positions(n)
    ]

    rounds: list[list[tuple]] = []
    while len(field) > 1:
        matches: list[tuple] = []
        nxt: list[dict | None] = []
        for i in range(0, len(field), 2):
            a, b = field[i], field[i + 1]
            if a is None and b is None:
                nxt.append(None)
                matches.append((None, None, None))
            elif a is None:
                nxt.append(b)
                matches.append((None, b, b))
            elif b is None:
                nxt.append(a)
                matches.append((a, None, a))
            else:
                win = a if a["score"] >= b["score"] else b
                nxt.append(win)
                matches.append((a, b, win))
        rounds.append(matches)
        field = nxt

    return {"rounds": rounds, "champion": field[0]}


def bracket_text(arena: dict, players: list[dict]) -> str:
    """Текстовая сетка плей-офф (для сообщения под табло)."""
    br = bracket(players)
    rounds = br.get("rounds") or []
    if not rounds:
        return ""
    out = [f"🏆 <b>МОГ-АРЕНА #{arena['id']} — СЕТКА</b>", ""]
    total = len(rounds)
    for ri, matches in enumerate(rounds):
        title = "ФИНАЛ" if ri == total - 1 else f"Раунд {ri + 1}"
        out.append(f"<b>{title}</b>")
        for a, b, w in matches:
            if a is None and b is None:
                continue
            if a is None:
                out.append(f"  ▫️ бай · <b>{_esc(b['name'])}</b>")
            elif b is None:
                out.append(f"  ▫️ бай · <b>{_esc(a['name'])}</b>")
            else:
                out.append(
                    f"  {_esc(a['name'])} {a['score']:.2f} — "
                    f"{_esc(b['name'])} {b['score']:.2f}"
                    f" → <b>{_esc(w['name'])}</b> ★"
                )
        out.append("")
    ch = br.get("champion")
    if ch:
        out.append(
            f"👑 <b>Чемпион: {_esc(ch['name'])}</b> · {ch['score']:.4f} ({ch['tier']})"
        )
    return "\n".join(out)
