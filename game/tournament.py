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
from .textfmt import fmt

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


_STAT_RU = {
    "jaw": "челюсть",
    "eyes": "взгляд",
    "cheeks": "скулы",
    "harmony": "гармония",
    "groom": "груминг",
    "aura": "аура",
}

# Хайп перед боем — по разнице баллов. Тон: угарный, самоироничный, без травли.
_HYPE_HUGE = (
    "букмекеры посмотрели пару и ушли курить",
    "похоже на спарринг с манекеном, который уже сдался",
    "второй участник заранее извиняется за неудобства",
    "разрыв такой, что можно было не выставлять второе фото",
    "тут не бой, а документальная хроника унижения",
)
_HYPE_MID = (
    "крепкая заруба: фаворит есть, но он об этом не знает",
    "второй не мальчик для битья, но и не любимец публики",
    "как борщ без сметаны — съедобно, но никто не в восторге",
    "нормальный матч, решит не талант, а отсутствие провалов",
    "судьи уже достали блокноты, зрители — телефоны",
)
_HYPE_CLOSE = (
    "качели: решит один пиксель и немного удачи",
    "оба на грани, и оба это чувствуют",
    "ничья по вайбу — разводим по цифрам, как взрослые",
    "такое обычно заканчивается спором в комментариях",
    "разница в погрешности, так что победил тот, кто не моргнул",
)

# Добивка короне.
_CROWN = (
    "Корона жмёт. Он терпит.",
    "Корона подошла с первого раза — такое бывает раз в жизни.",
    "Корона держится на честном слове и хорошем ракурсе.",
)


def _rng(*parts) -> "random.Random":
    """Стабильный рандом: одни и те же фото → одни и те же шутки."""
    import hashlib
    import random

    seed = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16))


def _top_stats(p: dict, k: int = 2) -> list[tuple[str, int]]:
    st = p.get("stats") or {}
    if not st:
        return []
    return sorted(st.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def _grade(v: int) -> str:
    """Как звучит цифра стата по-человечески."""
    if v >= 85:
        return "неприлично"
    if v >= 70:
        return "крепко"
    if v >= 55:
        return "рабоче"
    return "на морально-волевых"


def hype_line(a: dict, b: dict, rng: "random.Random") -> str:
    """Подводка к матчу: почему на это стоит смотреть."""
    gap = abs(a["score"] - b["score"])
    if gap >= FATALITY_GAP:
        pool = _HYPE_HUGE
    elif gap <= FLAWLESS_GAP:
        pool = _HYPE_CLOSE
    else:
        pool = _HYPE_MID
    return rng.choice(pool)


_WHY_TAIL_BIG = (
    "и это не бой, а объяснение",
    "второй стат даже не приехал",
    "тут можно было не выходить на ринг",
    "чистая арифметика без интриги",
)
_WHY_TAIL_MID = (
    "с запасом",
    "уверенно, без нервов",
    "как по расписанию",
    "ровно столько, сколько нужно",
)
_WHY_TAIL_TINY = (
    "на тоненького",
    "на честном слове и одном пикселе",
    "судьи дышали через раз",
    "решил один кадр",
)


_WHY_TIGHT = (
    "и этого хватило, чтобы решить исход на волоске",
    "но общий балл всё равно качнулся одним пикселем",
    "а сетка упала в его сторону на честном слове",
)


def why_line(
    winner: dict, loser: dict, rng: "random.Random | None" = None, tight: bool = False
) -> str:
    """Почему победил именно он — через разницу статов, а не «повезло»."""
    sw = winner.get("stats") or {}
    sl = loser.get("stats") or {}
    keys = [k for k in sw if k in sl]
    if not keys:
        return ""
    key = max(keys, key=lambda k: (sw[k] - sl[k], sw[k]))
    diff = sw[key] - sl[key]
    if diff <= 0:
        return (
            "🧠 Почему: ни один стат не решающий — просто меньше провалов "
            "по сумме, а это тоже искусство"
        )
    rng = rng or _rng(key, sw[key], sl[key])
    if tight:
        # Общий балл решился на волоске — «с запасом» тут звучало бы как враньё.
        pool = _WHY_TIGHT
    elif diff >= 25:
        pool = _WHY_TAIL_BIG
    elif diff >= 10:
        pool = _WHY_TAIL_MID
    else:
        pool = _WHY_TAIL_TINY
    return (
        f"🧠 Почему: {_STAT_RU.get(key, key)} {sw[key]} против {sl[key]} — "
        f"{rng.choice(pool)}"
    )


def _fighter(p: dict | None) -> str:
    if p is None:
        return "—"
    return f"<b>#{p['slot']}</b>"


def match_line(a: dict, b: dict | None, w: dict, *, rng=None) -> str:
    """Блок матча: пара, подводка, победа, причина победы."""
    if b is None:
        return f"▫️ бай: {_fighter(a)} проходит без боя"
    rng = rng or _rng(a.get("hash"), b.get("hash"), a["score"], b["score"])
    gap = abs(a["score"] - b["score"])
    loser = b if w is a else a
    lines = [
        f'🥊 {_fighter(a)} <code>{fmt(a["score"])}</code> '
        f'⚔ {_fighter(b)} <code>{fmt(b["score"])}</code>',
        f"   💬 {hype_line(a, b, rng)}",
        f"   ✅ победа: {_fighter(w)} · {_verdict(gap)}",
    ]
    why = why_line(w, loser, rng, tight=gap <= FLAWLESS_GAP)
    if why:
        lines.append(f"   {why}")
    return "\n".join(lines)


_NO_LUCK = (
    "кадр не дотянул",
    "не хватило одного ракурса",
    "сетка была против",
    "фото уходит в архив",
    "свет упал не туда",
)


def verdict_label(gap: float) -> str:
    """Вердикт словами без разметки — для штампа на постере и для подписи."""
    if gap >= FATALITY_GAP:
        return "FATALITY"
    if gap >= FLAWLESS_GAP:
        return "FLAWLESS VICTORY"
    return "PHOTO FINISH"


def top_note(p: dict) -> str:
    """Козырь фото одной строкой: «челюсть 84». Для постера, без emoji."""
    tops = _top_stats(p, 1)
    if not tops:
        return ""
    k, v = tops[0]
    return f"{_STAT_RU.get(k, k)} {v}"


def _loser_note(loser: dict, rng) -> str:
    """Что стало с проигравшим — вторая половина второго абзаца."""
    tops = _top_stats(loser, 1)
    if tops:
        k, v = tops[0]
        return (
            f"Проигравшему <b>#{loser['slot']}</b> даже "
            f"{_STAT_RU.get(k, k)} {v} не помогли: {rng.choice(_NO_LUCK)}."
        )
    return f"<b>#{loser['slot']}</b> уходит без трофея: {rng.choice(_NO_LUCK)}."


def match_post_text(a: dict, b: dict, w: dict, *, rng=None) -> str:
    """Два абзаца про бой — подпись к VS-постеру.

    Абзац 1: кто с кем, баллы, разрыв, козыри обоих.
    Абзац 2: почему победил именно он и что стало с проигравшим.
    Без HTML-заголовков: их добавляет вызывающий код.
    """
    rng = rng or _rng(a.get("hash"), b.get("hash"), a["score"], b["score"])
    gap = abs(a["score"] - b["score"])
    loser = b if w is a else a

    para1 = (
        f"<b>#{a['slot']}</b> {fmt(a['score'])} против "
        f"<b>#{b['slot']}</b> {fmt(b['score'])} — разрыв {fmt(gap)}. "
        f"{hype_line(a, b, rng)}"
    )
    ta, tb = _top_stats(a, 1), _top_stats(b, 1)
    if ta and tb:
        para1 += (
            f" Козырь первого — {_STAT_RU.get(ta[0][0], ta[0][0])} {ta[0][1]}, "
            f"второго — {_STAT_RU.get(tb[0][0], tb[0][0])} {tb[0][1]}."
        )

    core = (
        f"<b>#{w['slot']}</b> забирает бой — {verdict_label(gap)} "
        f"с запасом {fmt(gap)}."
    )
    why = why_line(w, loser, rng, tight=gap <= FLAWLESS_GAP)
    if why:
        why_txt = why.replace("🧠 Почему: ", "").strip()
        if why_txt:
            core += f" {why_txt[0].upper()}{why_txt[1:]}."
    return para1 + "\n\n" + core + " " + _loser_note(loser, rng)


def bye_post(a: dict, title: str) -> str:
    """Бай — не бой, но и о нём надо сказать, иначе лента рвётся."""
    return (
        f"<b>{title}</b>\n\n"
        f"▫️ <b>#{a['slot']}</b> проходит дальше без боя: пары не нашлось. "
        f"Балл у него всё равно {fmt(a['score'])} — сетка просто подождала."
    )


def result_header(rounds: list[list[tuple]], champion: dict, *, total: int) -> str:
    """Короткая шапка после FIGHT: дальше идут отдельные посты по боям."""
    return (
        f"⚔️ <b>СЕТКА СЫГРАНА</b> · {total} фото\n\n"
        f"🏆 Чемпион: <b>#{champion['slot']}</b> · "
        f"{fmt(champion['score'])} · {_esc(champion['tier'])}\n\n"
        "Ниже — по постеру на каждый бой и два абзаца: как прошло и почему так."
    )


def path_plain(rounds: list[list[tuple]] | None, champion: dict) -> str:
    """Путь чемпиона без emoji — для постера (DejaVu не рисует эмодзи)."""
    if not rounds:
        return ""
    wins = [m for r in rounds for m in r if m[2] is champion and m[1] is not None]
    if not wins:
        return ""
    avg = sum(abs(m[0]["score"] - m[1]["score"]) for m in wins) / len(wins)
    cnt = f"{len(wins)} " + _plural(len(wins), "победа", "победы", "побед")
    return f"{cnt} · средний разрыв {avg:.2f}"


def crown_line(champion: dict) -> str:
    """Та же коронная фраза, что в подписи: один сид → один и тот же текст."""
    rng = _rng("crown", champion.get("hash"), champion["score"])
    return rng.choice(_CROWN)


def _seed_of(rounds: list[list[tuple]]) -> str:
    """Один сид на весь турнир — чтобы повторный FIGHT давал те же шутки."""
    parts = []
    for matches in rounds:
        for a, b, _w in matches:
            parts.append(str(a.get("hash") or f"{a['slot']}:{a['score']}"))
            if b is not None:
                parts.append(str(b.get("hash") or f"{b['slot']}:{b['score']}"))
    return "|".join(sorted(parts))


def rounds_text(rounds: list[list[tuple]], *, total: int) -> str:
    """Весь турнир одним сообщением: раунды, матчи, подводки, добивки."""
    if not rounds:
        return "⚔️ <b>ТЕСТОВЫЙ ТУРНИР</b>\n\nПусто — нужно хотя бы 2 фото."
    rng = _rng(_seed_of(rounds))
    out = [f"⚔️ <b>ТЕСТОВЫЙ ТУРНИР</b> · {total} фото", ""]
    for i, matches in enumerate(rounds):
        title = _round_title(i, len(rounds))
        out.append(f"<b>{title} — FIGHT!</b>")
        for a, b, w in matches:
            out.append(match_line(a, b, w, rng=rng))
        out.append("")
    return "\n".join(out).rstrip()


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение: 1 победа / 2 победы / 5 побед."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def _path_line(rounds: list[list[tuple]] | None, champion: dict) -> str:
    """Путь чемпиона: сколько боёв, сколько баёв, средний разрыв."""
    if not rounds:
        return ""
    wins = [m for r in rounds for m in r if m[2] is champion and m[1] is not None]
    byes = [m for r in rounds for m in r if m[2] is champion and m[1] is None]
    if not wins:
        return ""
    avg = sum(abs(m[0]["score"] - m[1]["score"]) for m in wins) / len(wins)
    tail = ""
    if byes:
        tail = f" · {len(byes)} " + _plural(len(byes), "бай", "бая", "баёв")
    cnt = f"{len(wins)} " + _plural(len(wins), "победа", "победы", "побед")
    return f"🛣 Путь: {cnt}{tail} · средний разрыв {avg:.2f}"


def champion_text(
    champion: dict, *, total: int, rounds: list[list[tuple]] | None = None
) -> str:
    """Финальный блок: кто взял корону и ЧЕМ он хорош."""
    tops = _top_stats(champion, 2)
    if tops:
        pairs = " и ".join(f"{_STAT_RU.get(k, k)} {v}" for k, v in tops)
        grades = {_grade(v) for _, v in tops}
        # Одинаковая оценка у обоих статов — не повторяем слово дважды.
        strong = (
            f"{pairs} — {_grade(tops[0][1])}"
            if len(grades) == 1 and len(tops) > 1
            else ", ".join(f"{_STAT_RU.get(k, k)} {v} ({_grade(v)})" for k, v in tops)
        )
    else:
        strong = "данных нет, но харизма на месте"
    rng = _rng("crown", champion.get("hash"), champion["score"])
    lines = [
        f"🏆 <b>CHAMPION</b> · {_fighter(champion)}",
        f"<b>{fmt(champion['score'])}</b> · {_esc(champion['tier'])}",
        "",
        f"💪 Чем берёт: {strong}",
    ]
    path = _path_line(rounds, champion)
    if path:
        lines.append(path)
    lines += [
        "",
        f"👑 {rng.choice(_CROWN)}",
        "",
        f"<i>Сетка из {total} фото, победитель — по МОГ. MOG — игра, "
        "баллы случайны, не оценка внешности.</i>",
    ]
    return "\n".join(lines)


def result_text(rounds: list[list[tuple]], champion: dict, *, total: int) -> str:
    """Раунды + чемпион одним текстом (для одного сообщения без анимации)."""
    return (
        rounds_text(rounds, total=total)
        + "\n\n"
        + champion_text(champion, total=total, rounds=rounds)
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
