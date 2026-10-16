"""MOG-артефакты: покупаемые бусты для БАТТЛОВ (модель B — capped + контры).

Принципы (осознанный компромисс «рофл ↔ честность»):
  • Реальный, но ОГРАНИЧЕННЫЙ буст: суммарно не больше BOOST_CAP за бой.
  • Всё видно сопернику (никакого скрытого pay-to-win).
  • «Анти-сглаз» даёт контригру: снимает САМЫЙ СИЛЬНЫЙ артефакт соперника
    → гонка вооружений → повторные покупки.
  • 1 бесплатный артефакт в день — чтобы F2P попробовали механику.
  • Работает ТОЛЬКО в баттлах, не в соло-ролле (баттл = момент продажи).
"""
from __future__ import annotations

from html import escape

BOOST_CAP = 0.5  # максимум суммарного буста за один бой
FREE_PER_DAY = 1  # сколько бесплатных артефактов в день

# Каталог. stars — цена в Telegram Stars.
ARTIFACTS: dict[str, dict] = {
    "lift": {
        "name": "Лифтинг-скотч",
        "emoji": "💧",
        "stars": 5,
        "boost": 0.15,
        "desc": "+0.15 к МОГУ",
    },
    "light": {
        "name": "Правильный свет",
        "emoji": "📐",
        "stars": 8,
        "boost": 0.20,
        "desc": "+0.20 к МОГУ",
    },
    "osteo": {
        "name": "Остеопат-стажёр",
        "emoji": "🦴",
        "stars": 10,
        "boost": 0.25,
        "desc": "+0.25 к МОГУ",
    },
    "gaze": {
        "name": "Сильный взгляд",
        "emoji": "😤",
        "stars": 12,
        "boost": 0.30,
        "desc": "+0.30 к МОГУ",
    },
    "anti": {
        "name": "Анти-сглаз",
        "emoji": "🧿",
        "stars": 20,
        "boost": 0.00,
        "desc": "снимает сильнейший артефакт соперника",
        "counter": True,
    },
}


def _esc(s: str) -> str:
    return escape(s or "", quote=False)


def _boost_of(item: str) -> float:
    return float(ARTIFACTS.get(item, {}).get("boost", 0.0))


def compute(entries: list[dict]) -> dict[int, dict]:
    """Считает эффективные счёта с учётом бустов и контр.

    entries: [{"id": int, "score": float, "items": [item_key, ...]}, ...]
    Возвращает {id: {"raw", "eff", "boost", "items", "antis", "counters_used"}}.
    """
    items = {e["id"]: list(e.get("items") or []) for e in entries}

    # Контры: каждый «анти-сглаз» игрока снимает САМЫЙ СИЛЬНЫЙ не-anti артефакт
    # среди его соперников (в дуэли — очевидно у кого; в арене — у лидера).
    counters_used = {e["id"]: 0 for e in entries}
    for e in entries:
        me = e["id"]
        for _ in range(items[me].count("anti")):
            best = None  # (boost, owner_id, idx)
            for o in entries:
                if o["id"] == me:
                    continue
                for i, it in enumerate(items[o["id"]]):
                    if it == "anti":
                        continue
                    b = _boost_of(it)
                    if best is None or b > best[0]:
                        best = (b, o["id"], i)
            if best is None:
                break
            del items[best[1]][best[2]]
            counters_used[me] += 1

    out: dict[int, dict] = {}
    for e in entries:
        boosts = [_boost_of(it) for it in items[e["id"]] if it != "anti"]
        boost = round(min(sum(boosts), BOOST_CAP), 4)
        raw = round(float(e.get("score") or 0.0), 4)
        out[e["id"]] = {
            "raw": raw,
            "eff": round(raw + boost, 4),
            "boost": boost,
            "items": items[e["id"]],
            "antis": items[e["id"]].count("anti"),
            "counters_used": counters_used[e["id"]],
        }
    return out


def _items_line(items: list[str]) -> str:
    parts = [
        f"{ARTIFACTS[k]['emoji']} {ARTIFACTS[k]['name']}"
        for k in items
        if k in ARTIFACTS
    ]
    return ", ".join(parts)


def result_line(name: str, comp: dict) -> str:
    """Одна строка про буст игрока для итогового табло (или '')."""
    bits = []
    if comp.get("boost", 0) > 0:
        bits.append(f"⚡ +{comp['boost']:.3f} буст ({_items_line(comp['items'])})")
    if comp.get("counters_used"):
        bits.append(f"🧿 Анти-сглаз снял {comp['counters_used']} артефакт(ов)")
    if not bits:
        return ""
    return f"   <i>{_esc(name)}: " + " · ".join(bits) + "</i>"


def shop_text(battle_id: int, my_items: list[str], opp_items: list[str]) -> str:
    """Текст магазина артефактов для дуэли."""
    mine = _items_line(my_items) or "пусто"
    theirs = _items_line(opp_items) or "пусто"
    return (
        f"🎒 <b>АРТЕФАКТЫ · МОГ-БАТТЛ #{battle_id}</b>\n\n"
        f"Твоё: {mine}\n"
        f"У соперника: {theirs}\n\n"
        f"Бусты суммируются до <b>+{BOOST_CAP:.2f}</b> за бой. "
        "🧿 снимает сильнейший артефакт соперника.\n"
        "<i>Трать Stars — на реванш и ржу. Балл честный: кап и всё видно.</i>"
    )


def shop_keyboard(battle_id: int, my_items: list[str]):
    """Клавиатура магазина: купить артефакт / взять бесплатный."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    rows = []
    for key, a in ARTIFACTS.items():
        owned = my_items.count(key)
        suffix = f" · ×{owned}" if owned else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{a['emoji']} {a['name']} — {a['stars']}⭐{suffix}",
                    callback_data=f"art_buy:{battle_id}:{key}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="🎁 Бесплатный артефакт (1/день)",
                callback_data=f"art_free:{battle_id}",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
