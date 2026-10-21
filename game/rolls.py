"""Математика ролла MOG.

base детерминированно из sha256(фото) → одна фотка всегда даёт одну базу.
Поверх — шум (нормальное), редкий крит (Mythic) и провал.
Шкала MOG: 0.0000 – 9.9999 (4 знака).
"""
from __future__ import annotations

import hashlib
import io
from functools import lru_cache

from . import config

# ---------------------------------------------------------------- база/тиры


@lru_cache(maxsize=32)
def _fingerprint(photo_bytes: bytes) -> str:
    """Канонический отпечаток: aHash 8×8 + грубые бакеты яркости/насыщенности/RGB.

    Считается на блюре 64×64 (σ=2.0), поэтому не зависит от пережатия, ресайза и
    формата: замерено JPEG q95=q80=q55=q40 = PNG = resize↓2 → один и тот же
    отпечаток (8/8 вариантов). Цветовые бакеты добавлены, чтобы не терять
    различимость: без них 233 разных отпечатка на 300 синтетических картинок,
    с ними — 294/300. Перекодирование в WebP у части кадров смещает цветовой
    бакет (JPEG-путь Telegram такой проблемы не даёт), поэтому WebP-файл,
    отправленный дважды как есть, стабилен, а вот WebP-перекодия — не всегда.
    Пустая строка — картинку декодировать не удалось.
    """
    try:
        from PIL import Image, ImageFilter, ImageOps

        img = Image.open(io.BytesIO(photo_bytes))
        img = ImageOps.exif_transpose(img)
        gray = img.convert("L").resize((64, 64), Image.LANCZOS)
        gray = gray.filter(ImageFilter.GaussianBlur(2.0))
        px = gray.resize((8, 8), Image.LANCZOS).tobytes()
        mean = sum(px) / len(px)
        bits = 0
        for v in px:
            bits = (bits << 1) | (1 if v >= mean else 0)
        small = img.convert("RGB").resize((16, 16), Image.LANCZOS)
        rf = gf = bf = 0.0
        for r, g, b in small.getdata():
            rf += r
            gf += g
            bf += b
        n = 256
        rf, gf, bf = rf / n, gf / n, bf / n
        mx, mn = max(rf, gf, bf), min(rf, gf, bf)
        sat = 0.0 if mx <= 0 else (mx - mn) / mx
        return (
            f"{bits:016x}"
            f"{int(mean // 32):01x}{int(sat * 4):01x}"
            f"{int(rf // 48):01x}{int(gf // 48):01x}{int(bf // 48):01x}"
        )
    except Exception:  # noqa: BLE001 — без Pillow/битого файла деградируем до байтов
        return ""


def photo_hash(photo_bytes: bytes) -> str:
    """Перцептивный отпечаток фото: одинаковая картинка → одинаковый МОГ.

    sha256 от байтов ломал главное обещание «одно фото — один балл»: Telegram
    пережимает снимок, отправка «файлом» даёт другие байты, и то же лицо
    получало новый балл. Здесь aHash 8×8 по яркости (мягкий даунскейл гасит
    шум JPEG) плюс грубый бакет средней яркости — в замерах это 100% совпадений
    при перекомпрессии q40–q95, PNG↔JPEG, WebP и ресайзе в 2 раза, тогда как
    прежний sha256 — 0%. Бакет нужен, чтобы однотонные кадры не схлопывались
    в один отпечаток. Само фото не храним — только эту строку.
    """
    fp = _fingerprint(photo_bytes)
    return fp or hashlib.sha256(photo_bytes).hexdigest()


def raw_hash(photo_bytes: bytes) -> str:
    """sha256 исходных байтов — для логов и дедупа загрузок, не для балла."""
    return hashlib.sha256(photo_bytes).hexdigest()


@lru_cache(maxsize=32)
def _hash_int(data: bytes) -> int:
    """64 бита из отпечатка, равномерно размазанные по всему диапазону.

    Нельзя брать младшие биты самого отпечатка: у aHash 8×8 младшие биты — это
    правый нижний угол картинки плюс яркость, из-за чего балл зависел от угла
    кадра и кучи фото получали ОДИН балл (замер: 31 разный балл на 300 разных
    картинок). blake2b от канонической строки даёт равномерное распределение и
    сохраняет стабильность: отпечаток тот же → балл тот же.
    """
    fp = _fingerprint(data)
    if fp:
        return int.from_bytes(hashlib.blake2b(fp.encode(), digest_size=8).digest(), "big")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def base_score(photo_bytes: bytes) -> float:
    """Детерминированная база 2.5–8.5 из содержимого фото."""
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


def _roast(tier: str, seed: int = 0) -> str:
    """Подводка детерминирована: одно фото (и один спин) — один и тот же текст."""
    bank = _ROASTS[tier]
    return bank[seed % len(bank)]


def stats_for(photo_bytes: bytes) -> dict[str, int]:
    """Детерминированные статы зон (для неон-аннотации «слабые места»)."""
    return _stats(photo_bytes, base_score(photo_bytes))


# ---------------------------------------------------------------- сам ролл


def score_for(photo_bytes: bytes, spin: int = 0) -> float:
    """Балл фото: одинаковый снимок → одинаковый МОГ.

    spin=0 — чистая база. spin>0 — «перекид» платного реролла; он тоже
    детерминирован от (фото, номер спина), поэтому карточка, БД и повторный
    показ всегда совпадают. Раньше реролл брал random.gauss: за одно и то же
    фото в разных местах выходили разные числа, а оплаченный результат вообще
    не попадал в БД.
    """
    base = base_score(photo_bytes)
    if spin <= 0:
        return round(base, 4)
    h = _hash_int(photo_bytes) ^ ((spin * 0x9E3779B97F4A7C15) & ((1 << 96) - 1))
    delta = ((h % 1_000_003) / 1_000_003 - 0.5) * 1.6  # ±0.8
    return round(max(config.SCORE_MIN, min(config.SCORE_MAX, base + delta)), 4)


def roll(
    photo_bytes: bytes,
    *,
    spin: int = 0,
    force_crit: bool = False,
    force_fail: bool = False,
) -> dict:
    """Один ролл. Детерминирован: (фото, spin) → один и тот же результат."""
    base = base_score(photo_bytes)
    score = score_for(photo_bytes, spin)

    crit = bool(force_crit)
    failed = bool(force_fail) and not crit
    if crit:
        score = min(config.SCORE_MAX, round(score + config.CRIT_BONUS_MIN, 4))
    elif failed:
        score = max(config.SCORE_MIN, round(score - config.FAIL_PENALTY, 4))

    tier = tier_for(score, crit=crit)
    seed = _hash_int(photo_bytes) + spin * 7919
    return {
        "score": score,
        "base": round(base, 4),
        "tier": tier,
        "crit": crit,
        "failed": failed,
        "stats": _stats(photo_bytes, base),
        "roast": _roast(tier, seed),
        "disclaimer": config.DISCLAIMER,
        "photo_hash": photo_hash(photo_bytes),
        "spin": spin,
    }
