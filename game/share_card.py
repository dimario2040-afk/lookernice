"""Карточка результата 1080×1920 — то, что человек реально перешлёт в сторис.

Смысл: обычная МОГ-карточка это фото с маленькой печатью в углу — её не видно
в ленте и стыдно показывать. Здесь наоборот: вертикаль под сторис, огромный
балл, тир, одна фраза вердикта, подпись автора и ссылка на бота.

Внутри только Pillow. Палитра — общая (`game.palette`), текст без emoji:
DejaVu рисует их квадратами.
"""
from __future__ import annotations

import io
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from game.palette import ACCENT, FAINT, INK, MINT, MUTED, TEAL_0, TEAL_1
from game.textfmt import fmt

W, H = 1080, 1920
PHOTO_H = 1200
MARGIN = 56

_FONT_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
)

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int):
    size = max(8, int(size))
    cached = _font_cache.get(size)
    if cached is not None:
        return cached
    font = None
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:  # noqa: BLE001 — шрифт не критичен, есть дефолт
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _cover(raw: bytes, w: int, h: int) -> Image.Image:
    """Фото «в край»: центр-кроп под нужное соотношение."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    src_ratio = img.width / max(1, img.height)
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_w = int(img.height * dst_ratio)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / dst_ratio)
        top = max(0, int((img.height - new_h) * 0.32))  # лицо обычно выше центра
        img = img.crop((0, top, img.width, top + new_h))
    return img.resize((w, h), Image.LANCZOS)


def _wrap(draw, text: str, font, max_w: int, max_lines: int) -> list[str]:
    words = (text or "").split()
    lines: list[str] = []
    cur = ""
    for word in words:
        probe = f"{cur} {word}".strip()
        if draw.textlength(probe, font=font) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and words:
        joined = " ".join(lines)
        if len(joined) < len(text) and not joined.endswith("…"):
            lines[-1] = lines[-1].rstrip(" ,;:—–-") + "…"
    return lines


def share_card(
    raw: bytes,
    *,
    score: float,
    tier: str,
    verdict: str = "",
    handle: str = "",
    link: str = "",
) -> bytes:
    """Вертикальная карточка результата. Возвращает JPEG bytes."""
    layer = Image.new("RGB", (W, H), TEAL_0)
    layer.paste(_cover(raw, W, PHOTO_H), (0, 0))

    # Скрим снизу: под ним живут балл и вердикт. Без него цифры тонут в фото.
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    start = int(PHOTO_H * 0.42)
    for y in range(start, PHOTO_H):
        k = (y - start) / max(1, PHOTO_H - start)
        alpha = int(255 * min(1.0, k ** 1.25))
        sd.line([(0, y), (W, y)], fill=TEAL_0 + (alpha,))
    layer = Image.alpha_composite(layer.convert("RGBA"), scrim).convert("RGB")
    d = ImageDraw.Draw(layer)

    # Тир — чипом в левом верхнем углу, чтобы читался даже в превью сторис.
    if tier:
        f_tier = _font(40)
        tw = d.textlength(tier.upper(), font=f_tier)
        d.rounded_rectangle(
            [MARGIN, MARGIN, MARGIN + tw + 44, MARGIN + 74],
            radius=37,
            fill=TEAL_1,
            outline=ACCENT,
            width=2,
        )
        d.text((MARGIN + 22, MARGIN + 15), tier.upper(), font=f_tier, fill=MINT)

    # Балл: два знака, огромно, с мятным свечением.
    score_text = fmt(score)
    size = 300
    f_score = _font(size)
    while d.textlength(score_text, font=f_score) > W - 2 * MARGIN and size > 90:
        size = int(size * 0.9)
        f_score = _font(size)
    label_f = _font(46)
    score_w = d.textlength(score_text, font=f_score)
    score_y = PHOTO_H - int(size * 0.92) - 96

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text(((W - score_w) / 2, score_y), score_text, font=f_score, fill=MINT + (255,))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    layer = Image.alpha_composite(layer.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(layer)

    d.text(((W - d.textlength("MOG", font=label_f)) / 2, score_y - 74), "MOG", font=label_f, fill=MUTED)
    d.text(((W - score_w) / 2, score_y), score_text, font=f_score, fill=MINT)

    # Вердикт — одна-две строки под фото.
    y = PHOTO_H + 64
    if verdict:
        f_v = _font(52)
        for line in _wrap(d, verdict, f_v, W - 2 * MARGIN, 2):
            w = d.textlength(line, font=f_v)
            d.text(((W - w) / 2, y), line, font=f_v, fill=INK)
            y += 66

    # Подпись автора и ссылка.
    f_handle, f_link = _font(44), _font(34)
    y_handle = H - 200
    if handle:
        d.text((MARGIN, y_handle), handle[:28], font=f_handle, fill=ACCENT)
    if link:
        lw = d.textlength(link, font=f_link)
        d.text((W - MARGIN - lw, y_handle + 8), link, font=f_link, fill=MUTED)

    d.rectangle([MARGIN, y_handle - 34, W - MARGIN, y_handle - 30], fill=TEAL_1)
    f_foot = _font(30)
    foot = "18+ · балл выдуманный · это не оценка внешности"
    d.text(((W - d.textlength(foot, font=f_foot)) / 2, H - 118), foot, font=f_foot, fill=FAINT)

    out = io.BytesIO()
    layer.save(out, format="JPEG", quality=90, optimize=True)
    return out.getvalue()
