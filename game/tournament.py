"""ТЕСТОВЫЙ ТУРНИР: один человек загружает до 8 фото и сам разводит сетку.

Зачем: когда в чате никого нет и баттл запустить не с кем — можно прогнать
сетку на своих фото (друзья, мемы, селфи из галереи) и посмотреть, кто дойдёт
до финала. Счёт детерминирован по фото, поэтому сетка честная и повторяемая.

Поток: /tournament → до 8 фото → «Готово» → свап слотов 1..8 → «FIGHT» →
раунды текстом (без анимации) → чемпион.

Здесь только чистая логика и тексты: ни БД, ни Telegram — так проще тестировать.
"""
from __future__ import annotations

from html import escape

MIN_PHOTOS = 2
MAX_PHOTOS = 8

# Разница в МОГ, после которой бой считается разгромным / впритык.
FATALITY_GAP = 2.0
FLAWLESS_GAP = 0.35


def _esc(s: str) -> str:
    return escape(str(s or ""), quote=False)


def _round_title(index: int, total: int) -> str:
    """Название раунда по порядку: ROUND 1, ROUND 2, ... SEMIFINAL, FINAL."""
    if index >= total - 1:
        return "FINAL"
    if index == total - 2:
        return "SEMIFINAL"
    return f"ROUND {index + 1}"


def build_rounds(items: list[dict]) -> tuple[list[list[tuple]], dict | None]:
    """Сетка строго по порядку items (как расставил человек), без пересева.

    items: [{"slot": 1, "score": 6.1, "tier": "Epic"}, ...] в порядке пар:
    (1,2), (3,4), ... Нечётный хвост получает бай и проходит дальше сам.

    Возвращает (rounds, champion); rounds — список раундов, каждый список матчей
    (a, b, winner), где b=None означает бай.
    """
    field = list(items)
    rounds: list[list[tuple]] = []
    if not field:
        return rounds, None
    while len(field) > 1:
        nxt: list[dict] = []
        matches: list[tuple] = []
        for i in range(0, len(field), 2):
            a = field[i]
            b = field[i + 1] if i + 1 < len(field) else None
            if b is None:
                nxt.append(a)
                matches.append((a, None, a))
                continue
            # Ничья по score — побеждает тот, кто выше в сетке (стабильно).
            winner = a if a["score"] >= b["score"] else b
            nxt.append(winner)
            matches.append((a, b, winner))
        rounds.append(matches)
        field = nxt
    return rounds, field[0]


def _verdict(gap: float) -> str:
    """Короткая добивка в стиле аркадного файтинга (без анимации, только текст)."""
    if gap >= FATALITY_GAP:
        return "💀 <b>FATALITY</b>"
    if gap >= FLAWLESS_GAP:
        return "🔥 <b>FLAWLESS VICTORY</b>"
    return "😮‍💨 <b>PHOTO FINISH</b>"


def _fighter(p: dict | None) -> str:
    if p is None:
        return "—"
    return f'<b>{p["slot"]}</b>. {_esc(p.get("label") or "Фото")}'


def match_line(a: dict, b: dict | None, w: dict) -> str:
    """Одна строка матча: кто с кем и кто выиграл."""
    if b is None:
        return f"▫️ бай — {_fighter(a)} проходит дальше"
    gap = abs(a["score"] - b["score"])
    return (
        f'{_fighter(a)} <code>{a["score"]:.4f}</code> '
        f'⚔ {_fighter(b)} <code>{b["score"]:.4f}</code>\n'
        f'   → <b>победа: {_fighter(w)}</b> · {_verdict(gap)}'
    )


def rounds_text(rounds: list[list[tuple]], *, total: int) -> str:
    """Весь турнир одним сообщением: раунды, матчи, добивки."""
    if not rounds:
        return "⚔️ <b>ТЕСТОВЫЙ ТУРНИР</b>\n\nПусто — нужно хотя бы 2 фото."
    out = [f"⚔️ <b>ТЕСТОВЫЙ ТУРНИР</b> · {total} фото", ""]
    for i, matches in enumerate(rounds):
        title = _round_title(i, len(rounds))
        out.append(f"<b>{title} — FIGHT!</b>")
        for a, b, w in matches:
            out.append(match_line(a, b, w))
        out.append("")
    return "\n".join(out).rstrip()


def champion_text(champion: dict, *, total: int) -> str:
    """Финальный блок: кто взял корону."""
    return (
        f"🏆 <b>CHAMPION</b> · {_esc(champion.get('label') or 'Фото')} "
        f"(слот {champion['slot']})\n"
        f"<b>{champion['score']:.4f}</b> · {_esc(champion['tier'])}\n\n"
        f"<i>Сетка из {total} фото, победитель — по МОГ. MOG — игра, "
        "баллы случайны, не оценка внешности.</i>"
    )


def result_text(rounds: list[list[tuple]], champion: dict, *, total: int) -> str:
    """Раунды + чемпион одним текстом (для одного сообщения без анимации)."""
    return (
        rounds_text(rounds, total=total)
        + "\n\n"
        + champion_text(champion, total=total)
    )


# ---------------------------------------------------------------- клавиатуры
def slots_keyboard(n: int, picked: int | None = None):
    """Кнопки слотов 1..n. Нажал один — подсветился, нажал второй — свап."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    rows: list[list] = []
    row: list = []
    for i in range(1, n + 1):
        mark = f"[{i}]" if picked == i else str(i)
        row.append(
            InlineKeyboardButton(text=mark, callback_data=f"tt:pick:{i}")
        )
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(text="⚔️ FIGHT", callback_data="tt:fight"),
            InlineKeyboardButton(text="✖️ Отмена", callback_data="tt:cancel"),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def collect_keyboard(n: int):
    """Панель сбора: «Готово» активна, когда есть минимум 2 фото."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    rows = []
    if n >= MIN_PHOTOS:
        rows.append(
            [InlineKeyboardButton(text=f"✅ Готово ({n})", callback_data="tt:done")]
        )
    rows.append(
        [InlineKeyboardButton(text="✖️ Отмена", callback_data="tt:cancel")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def collect_text(n: int) -> str:
    """Текст панели сбора. Мелкий, без простыни."""
    left = MAX_PHOTOS - n
    if n == 0:
        return (
            "🏟 <b>ТЕСТОВЫЙ ТУРНИР</b>\n\n"
            f"Кидай до {MAX_PHOTOS} фото — альбомом или по одному.\n"
            "Потом расставишь пары и запустишь сетку."
        )
    tail = f"Можно ещё {left}." if left else "Хватит — жми «Готово»."
    return f"🏟 <b>ТЕСТОВЫЙ ТУРНИР</b>\n\n📥 Собрано: <b>{n}/{MAX_PHOTOS}</b>. {tail}"


def panel_text(n: int, picked: int | None) -> str:
    """Панель распределения: что за порядок и что делать дальше."""
    hint = (
        f"Выбран слот <b>{picked}</b> — жми второй, чтобы поменять их местами."
        if picked
        else "Жми два номера, чтобы поменять их местами. Пары: (1,2), (3,4)…"
    )
    return (
        f"🎛 <b>Расстановка</b> · {n} фото\n\n{hint}\n\n"
        "Фото пронумерованы в том же порядке, что в альбоме ниже. "
        "Нечётный хвост получает бай."
    )


def fight_keyboard():
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔁 Ещё раз", callback_data="tt:reset")],
            [InlineKeyboardButton(text="✖️ Закрыть", callback_data="tt:cancel")],
        ]
    )
