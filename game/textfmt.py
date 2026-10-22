"""Форматирование для людей: два знака у балла и одна фраза вердикта.

Зачем отдельный модуль: `5.8636` встречалось в 49 местах и везде выглядело как
машинный лог, а не как цифра, которой хвастаются. Теперь одна функция на всех.
"""
from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def fmt(score: float) -> str:
    """Балл для людей: 5.86, а не 5.8636."""
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def verdict_line(roast: str, limit: int = 120) -> str:
    """Первая фраза роаста — то, что реально влезает в строку карточки.

    Если первая фраза совсем огрызок («Ну что ж.»), берём весь текст: короткая
    строка на постере выглядит как баг вёрстки, а не как шутка.
    """
    text = re.sub(r"<[^>]+>", "", roast or "")
    text = " ".join(text.split())
    if not text:
        return ""
    first = _SENTENCE.split(text, maxsplit=1)[0].strip()
    if len(first) < 18 and len(text) > len(first):
        first = text
    if len(first) > limit:
        cut = first[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—–-")
        first = (cut or first[:limit]) + "…"
    return first
