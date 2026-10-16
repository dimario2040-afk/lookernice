"""Математика ролла MOG.

base детерминированно из sha256(фото) → одна фотка всегда даёт одну базу.
Поверх — шум ±0.3 (нормальное), редкий крит (+0.5..0.8, Mythic) и провал (−0.5).
"""
from __future__ import annotations

import hashlib
import random

from . import config

# ---------------------------------------------------------------- база/тиры


def photo_hash(photo_bytes: bytes) -> str:
    """Стабильный хэш фото (в БД пишем только его, байты не храним)."""
    return hashlib.sha256(photo_bytes).hexdigest()


def _hash_int(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def base_score(photo_bytes: bytes) -> float:
    """Детерминированная база 2.0–4.4 из содержимого фото."""
    h = _hash_int(photo_bytes)
    frac = (h % 1_000_000) / 1_000_000
    return config.BASE_MIN + frac * (config.BASE_MAX - config.BASE_MIN)


def tier_for(score: float, crit: bool = False) -> str:
    """Тир/редкость по score (Mythic — только крит)."""
    if crit:
        return "Mythic"
    if score >= config.TIER_LEGENDARY:
        return "Legendary"
    if score >= config.TIER_EPIC:
        return "Epic"
    if score >= config.TIER_RARE:
        return "Rare"
    return "Common"


# ---------------------------------------------------------------- статы карты

_STAT_KEYS = ("jaw", "eyes", "cheeks", "harmony", "groom", "aura")

# Профили: насколько стат «тянется» за общим баллом + личный сдвиг.
_STAT_PROFILE = {
    "jaw": (0.55, 0.12),
    "eyes": (0.60, 0.05),
    "cheeks": (0.50, 0.10),
    "harmony": (0.70, 0.00),
    "groom": (0.35, 0.25),
    "aura": (0.45, 0.20),
}


def _stats(photo_bytes: bytes, base: float) -> dict[str, int]:
    """6 статов 0–100: base × профиль + детерминированный джиттер от хэша."""
    h = _hash_int(photo_bytes)
    norm = (base - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN)  # 0..1
    out: dict[str, int] = {}
    for i, key in enumerate(_STAT_KEYS):
        slope, offset = _STAT_PROFILE[key]
        jitter = ((h >> (i * 7)) % 21 - 10) / 100.0  # −0.10..+0.10
        val = (slope + offset) * norm + jitter + (norm - 0.5) * 0.2
        out[key] = max(1, min(99, round(val * 100)))
    return out


# ---------------------------------------------------------------- roast-банк
# Жёстко, 18+, «свой пацан с форума». БЕЗ призывов к самоповреждению/суициду
# и без оскорблений по защищённым признакам.
_ROASTS = {
    "Common": [
        "Брат, тут не мог, тут горе. Срочно в зал и к барберу.",
        "Камера тебя не любит, но, честно, тут дело не в камере.",
        "Вердикт: artistic sub-5. Мьюинг тебе в помощь, дружище.",
        "Ты не могнул никого. Даже себя. Иди сушись.",
        "Филлеров у природы не завезли — держи что есть и не ной.",
    ],
    "Rare": [
        "Середнячок с амбициями. Кости на месте, но упаковка подводит.",
        "MTN-классика: не пугаешь, но и не распахивают двери.",
        "Есть база — не хватает полировки. Работай над вайбом.",
        "Нормальный парень, которого легко смогать чуть подкачанной опцией.",
    ],
    "Epic": [
        "Опа, HTN. Кто-то явно знает, что такое canthal tilt.",
        "Хорошая генетика, не спали её питанием и стрижкой.",
        "Ты мог. Не Chad, но близко, и зал это чувствует.",
        "Кости тащат. Осталось не запороть режимом сна.",
    ],
    "Legendary": [
        "Chad-lite. Заходишь в чат — и все резко захотели в зал.",
        "Тут природа расщедрилась. Не растеряй, пластырь не нужен.",
        "Ты практически дизайнерская порода. Осторожно с аудиторией.",
    ],
    "Mythic": [
        "💎 КРИТ! Это не лицо, это легендарка из пака. Все в комнате смоганы.",
        "💎 MYTHIC. Суперприз. Свет лампочки сам подстраивается под тебя.",
        "💎 Крит на миллион. Аукцион открыт, ставки в звёздах.",
    ],
}


def _roast(tier: str) -> str:
    return random.choice(_ROASTS[tier])


def stats_for(photo_bytes: bytes) -> dict[str, int]:
    """Детерминированные статы зон (для неон-аннотации «слабые места»)."""
    return _stats(photo_bytes, base_score(photo_bytes))


# ---------------------------------------------------------------- сам ролл


def roll(
    photo_bytes: bytes,
    *,
    force_crit: bool = False,
    force_fail: bool = False,
) -> dict:
    """Один ролл. Возвращает dict для БД и API."""
    base = base_score(photo_bytes)

    crit = force_crit or (not force_fail and random.random() < config.CRIT_CHANCE)
    failed = (not crit) and (force_fail or random.random() < config.FAIL_CHANCE)

    bonus = 0.0
    if crit:
        bonus = random.uniform(config.CRIT_BONUS_MIN, config.CRIT_BONUS_MAX)
    elif failed:
        bonus = -config.FAIL_PENALTY

    noise = random.gauss(0.0, config.ROLL_SIGMA)
    score = base + noise + bonus
    score = max(config.SCORE_MIN, min(config.SCORE_MAX, score))
    score = round(score, 2)

    tier = tier_for(score, crit=crit)
    return {
        "score": score,
        "base": round(base, 2),
        "tier": tier,
        "crit": crit,
        "failed": failed,
        "stats": _stats(photo_bytes, base),
        "roast": _roast(tier),
        "disclaimer": config.DISCLAIMER,
        "photo_hash": photo_hash(photo_bytes),
    }
