"""Собирает статику для Netlify из game/legal.py.

Netlify отдаёт ТОЛЬКО статику (страница согласия + лендинг) — бот там жить не
может (serverless, нет длительного процесса, long-polling не работает).
Бот остаётся на Render (webhook), а эта страница — вечный, бесплатный,
не засыпающий адрес для ссылки «правила/согласие».

Запуск:  python tools/build_netlify.py
Результат: netlify/public/{index.html, legal/index.html}
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import legal  # noqa: E402  (после правки sys.path)

OUT = os.path.join(ROOT, "netlify", "public")

BOT_URL = os.environ.get("MOG_BOT_URL", "https://t.me/lookernice_bot")

INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MOG — узнай свой МОГ (18+)</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:radial-gradient(1200px 600px at 50% -10%, #1b2130, #0b0d12 60%);
         color:#e7e9ee; font:16px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif; padding:24px; }}
  .wrap {{ max-width:560px; text-align:center; }}
  h1 {{ font-size:40px; margin:0 0 8px; letter-spacing:.5px; }}
  p {{ color:#9aa3b2; margin:8px 0 24px; }}
  .btn {{ display:inline-block; background:#2b6cff; color:#fff; text-decoration:none;
          padding:14px 26px; border-radius:14px; font-weight:600; font-size:18px; }}
  .btn:hover {{ background:#3d79ff; }}
  .links {{ margin-top:28px; font-size:14px; }}
  .links a {{ color:#7ab7ff; text-decoration:none; margin:0 8px; }}
  .pill {{ display:inline-block; background:#212a3a; border-radius:999px; padding:2px 10px;
           font-size:13px; color:#cfe3ff; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>MOG</h1>
  <p>Кинь фото в бота — получи свой игровой «МОГ» и печать. <span class="pill">18+</span></p>
  <a class="btn" href="{bot}">Открыть бота в Telegram</a>
  <div class="links">
    <a href="/legal/">Правила и согласие</a>
    <a href="/legal/#8">18+</a>
  </div>
</div>
</body>
</html>
"""


def main() -> None:
    os.makedirs(os.path.join(OUT, "legal"), exist_ok=True)
    with open(os.path.join(OUT, "legal", "index.html"), "w", encoding="utf-8") as f:
        f.write(legal.page_html(base_url=""))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX_HTML.format(bot=BOT_URL))
    print("netlify static built ->", OUT)
    print("  ", os.path.join(OUT, "index.html"))
    print("  ", os.path.join(OUT, "legal", "index.html"))


if __name__ == "__main__":
    main()
