"""Постеры турнира: VS-карточка на бой и карточка чемпиона.

Визуальный референс — витрина стриминга (КИОН): тёмный фон, фото «в край»
без рамок, мягкий градиентный скрим, минимум хрома и один акцент. Палитра
перекрашена в бирюзу: от почти чёрного #05131A до мятного #7CF5E4.

Внутри только Pillow: free tier на Render, лишние зависимости не нужны.
Текст на картинке — без emoji (DejaVu их не рисует, вышли бы квадраты).
"""

from __future__ import annotations

import io
import os

W, H = 1080, 1350  # портретный постер: в ленте Telegram читается во всю ширину

# ── палитра: один источник на все картинки бота ─────────────────────────────
from game.palette import (  # noqa: E402 — константы, порядок не важен
    ACCENT,
    FAINT,
    INK,
    MINT,
    MUTED,
    TEAL_0,
    TEAL_1,
    TEAL_2,
)


def _font(size: int):
    """Кириллический жирный шрифт: сперва наш в репо, потом системные."""
    from PIL import ImageFont

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (
        os.path.join(here, "assets", "fonts", "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 — нет файла/не TrueType
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:  # noqa: BLE001 — старый Pillow
        return ImageFont.load_default()


def prepare_thumb(raw: bytes, side: int = 720) -> bytes:
    """Миниатюра для состояния турнира: EXIF-поворот + ≤ side px.

    Байты оригинала не держим (память free tier), но и не перекачиваем фото с
    Telegram на каждый бой — постеры собираются из этих миниатюр.
    """
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=82, optimize=True)
    return out.getvalue()


def _cover(raw: bytes, w: int, h: int):
    """Фото в кадр без искажений: масштаб «по большей стороне» + центральный кроп."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize(
        (max(w, int(img.width * scale + 0.5)), max(h, int(img.height * scale + 0.5))),
        Image.LANCZOS,
    )
    left = (img.width - w) // 2
    top = (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _base():
    """Фон-градиент по вертикали + еле заметная сетка (глубина без шума)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), TEAL_0)
    d = ImageDraw.Draw(img)
    for y in range(H):
        k = y / (H - 1)
        d.line(
            [(0, y), (W, y)],
            fill=(
                int(TEAL_0[0] + (TEAL_1[0] - TEAL_0[0]) * k),
                int(TEAL_0[1] + (TEAL_1[1] - TEAL_0[1]) * k),
                int(TEAL_0[2] + (TEAL_1[2] - TEAL_0[2]) * k),
            ),
        )
    for y in range(0, H, 72):
        d.line([(0, y), (W, y)], fill=TEAL_1, width=1)
    return img.convert("RGBA")


def _scrim(layer, y0: int, y1: int, color=TEAL_0, peak: float = 1.0):
    """Градиентная шторка под текст: снизу непрозрачно, сверху прозрачно.

    Нужна, чтобы подписи читались на любом фото — светлом, пёстром, каком угодно.
    """
    from PIL import Image, ImageDraw

    grad = Image.new("RGBA", (W, y1 - y0), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    span = max(1, y1 - y0)
    for i in range(span):
        a = int(255 * peak * (i / span) ** 1.6)
        gd.line([(0, i), (W, i)], fill=color + (a,))
    layer.alpha_composite(grad, (0, y0))


def _text_center(d, cx: int, y: int, text: str, font, fill, anchor: str = "ma"):
    d.text((cx, y), text, font=font, fill=fill, anchor=anchor)


def _glow(layer, cx: int, y: int, text: str, font, color, radius: int = 18):
    """Неон: размытая копия под чётким текстом."""
    from PIL import Image, ImageDraw, ImageFilter

    halo = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(halo).text((cx, y), text, font=font, fill=color + (255,), anchor="ma")
    halo = halo.filter(ImageFilter.GaussianBlur(radius))
    layer.alpha_composite(halo)
    layer.alpha_composite(halo)
    ImageDraw.Draw(layer).text((cx, y), text, font=font, fill=INK + (255,), anchor="ma")


def _chip(layer, x: int, y: int, text: str, *, size: int = 26, right: bool = False):
    """Компактная плашка-ярлык: «MOG», «FINAL», «#3». right=True — прижать к правому краю."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(layer)
    f = _font(size)
    b = d.textbbox((0, 0), text, font=f)
    tw, th = b[2] - b[0], b[3] - b[1]
    w, h = tw + 36, th + 22
    x0 = W - 40 - w if right else x
    d.rounded_rectangle(
        [x0, y, x0 + w, y + h], radius=h // 2, fill=TEAL_0 + (232,), outline=ACCENT, width=2
    )
    d.text((x0 + 18 - b[0], y + 11 - b[1]), text, font=f, fill=INK)
    return w, h


def _column(layer, thumb: bytes, x0: int, x1: int, y0: int, y1: int, *, winner: bool):
    """Фото во всю колонку. Победитель — сочный, проигравший — притушен."""
    from PIL import ImageEnhance

    img = _cover(thumb, x1 - x0, y1 - y0)
    if not winner:
        img = ImageEnhance.Color(img).enhance(0.45)
        img = ImageEnhance.Brightness(img).enhance(0.72)
    else:
        img = ImageEnhance.Contrast(img).enhance(1.06)
    layer.alpha_composite(img.convert("RGBA"), (x0, y0))


def vs_card(
    left_thumb: bytes,
    right_thumb: bytes,
    *,
    round_title: str,
    slot_a: int,
    score_a: float,
    slot_b: int,
    score_b: float,
    winner: str,
    verdict: str,
    gap: float,
    note_a: str = "",
    note_b: str = "",
) -> bytes:
    """Постер боя: два фото в край, VS, баллы, штамп вердикта. JPEG bytes."""
    from PIL import Image, ImageDraw

    layer = _base()
    hero_bottom = 1030
    mid = W // 2
    seam = 5

    _column(layer, left_thumb, 0, mid - seam // 2, 0, hero_bottom, winner=winner == "a")
    _column(layer, right_thumb, mid + seam // 2, W, 0, hero_bottom, winner=winner == "b")

    d = ImageDraw.Draw(layer)
    d.rectangle([mid - seam // 2, 0, mid + seam // 2, hero_bottom], fill=TEAL_0 + (255,))
    # скрим снизу: под ним живут баллы и вердикт; сверху — под шапку
    _scrim(layer, 620, hero_bottom, TEAL_0, peak=1.0)
    _scrim(layer, 0, 190, TEAL_0, peak=0.85)

    d = ImageDraw.Draw(layer)
    _chip(layer, 40, 34, "MOG", size=28)
    d.text((W - 40, 46), round_title.upper(), font=_font(30), fill=MINT, anchor="ra")

    # бейджи слотов — ближе к баллам, а не поверх лиц
    _chip(layer, 40, hero_bottom - 168, f"#{slot_a}", size=30)
    _chip(layer, 0, hero_bottom - 168, f"#{slot_b}", size=30, right=True)

    # VS — круглый бейдж на стыке, а не текст поверх лиц
    cx, cy, r = mid, 560, 92
    badge = Image.new("RGBA", (r * 2 + 8, r * 2 + 8), (0, 0, 0, 0))
    bd = ImageDraw.Draw(badge)
    bd.ellipse([4, 4, r * 2 + 4, r * 2 + 4], fill=TEAL_0 + (238,), outline=ACCENT, width=4)
    layer.alpha_composite(badge, (cx - r - 4, cy - r - 4))
    _glow(layer, cx, cy - 40, "VS", _font(76), ACCENT, radius=14)

    d = ImageDraw.Draw(layer)
    d.text((56, hero_bottom - 96), f"{score_a:.4f}", font=_font(64), fill=INK)
    d.text(
        (W - 56, hero_bottom - 96),
        f"{score_b:.4f}",
        font=_font(64),
        fill=MUTED if winner == "a" else INK,
        anchor="ra",
    )
    if note_a:
        d.text((56, hero_bottom - 30), note_a.upper(), font=_font(24), fill=MUTED)
    if note_b:
        d.text((W - 56, hero_bottom - 30), note_b.upper(), font=_font(24), fill=MUTED, anchor="ra")

    # нижний блок на чистом фоне — как инфо-панель в витрине стриминга
    d.rectangle([0, hero_bottom, W, H], fill=TEAL_0 + (255,))
    d.line([(40, hero_bottom + 1), (W - 40, hero_bottom + 1)], fill=TEAL_2, width=2)

    pill = f"{verdict.upper()}  ·  {gap:.4f}"
    f_pill = _font(40)
    b = d.textbbox((0, 0), pill, font=f_pill)
    pw = b[2] - b[0] + 76
    px0 = (W - pw) // 2
    d.rounded_rectangle(
        [px0, hero_bottom + 54, px0 + pw, hero_bottom + 146],
        radius=46,
        fill=TEAL_1 + (255,),
        outline=ACCENT,
        width=3,
    )
    d.text((px0 + 38 - b[0], hero_bottom + 78 - b[1]), pill, font=f_pill, fill=MINT)

    _text_center(d, W // 2, hero_bottom + 190, "Сетка по баллам МОГ", _font(30), INK)
    _text_center(d, W // 2, hero_bottom + 238, "игра, не оценка внешности", _font(26), MUTED)

    out = io.BytesIO()
    layer.convert("RGB").save(out, "JPEG", quality=90, optimize=True)
    return out.getvalue()


def champion_card(
    thumb: bytes,
    *,
    slot: int,
    score: float,
    tier: str,
    path: str = "",
    crown: str = "",
) -> bytes:
    """Финальный постер: фото в край, CHAMPION, балл, путь. JPEG bytes."""
    import math

    from PIL import Image, ImageDraw, ImageFilter

    layer = _base()
    hero_bottom = 900

    img = _cover(thumb, W, hero_bottom)
    layer.alpha_composite(img.convert("RGBA"), (0, 0))

    # лучи из-под короны — бирюзовые, почти невидимые
    rays = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rays)
    cx, cy = W // 2, 430
    for i in range(24):
        ang = math.pi * 2 * i / 24
        wdt = 0.05
        rd.polygon(
            [
                (cx, cy),
                (cx + H * math.cos(ang - wdt), cy + H * math.sin(ang - wdt)),
                (cx + H * math.cos(ang + wdt), cy + H * math.sin(ang + wdt)),
            ],
            fill=ACCENT + (14,),
        )
    layer.alpha_composite(rays.filter(ImageFilter.GaussianBlur(3)))

    _scrim(layer, 420, hero_bottom, TEAL_0, peak=1.0)
    _scrim(layer, 0, 190, TEAL_0, peak=0.8)

    _chip(layer, 40, 34, "MOG", size=28)
    _chip(layer, 40, hero_bottom - 150, f"#{slot}", size=30)
    _glow(layer, W // 2, hero_bottom - 300, "CHAMPION", _font(84), ACCENT, radius=18)

    d = ImageDraw.Draw(layer)
    d.rectangle([0, hero_bottom, W, H], fill=TEAL_0 + (255,))
    d.line([(40, hero_bottom + 1), (W - 40, hero_bottom + 1)], fill=TEAL_2, width=2)

    _text_center(d, W // 2, hero_bottom + 48, f"{score:.4f}", _font(84), INK)
    _text_center(d, W // 2, hero_bottom + 152, (tier or "").upper(), _font(32), ACCENT)
    if path:
        _text_center(d, W // 2, hero_bottom + 216, path, _font(30), MUTED)
    if crown:
        _text_center(d, W // 2, hero_bottom + 268, crown, _font(28), INK)
    _text_center(
        d, W // 2, H - 54, "MOG · баллы случайны, не оценка внешности", _font(24), FAINT
    )

    out = io.BytesIO()
    layer.convert("RGB").save(out, "JPEG", quality=90, optimize=True)
    return out.getvalue()
