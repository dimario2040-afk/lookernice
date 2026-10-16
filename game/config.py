"""Константы MOG: ролл, экономика, пути, дисклеймер."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Токен бота (нужен для проверки WebApp initData). Совпадает с TELEGRAM_TOKEN бота.
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# SQLite база игры (создаётся сама).
DB_PATH = os.getenv("MOG_DB_PATH", str(BASE_DIR / "mog.db"))

# Папка со статикой мини-аппа.
WEBAPP_DIR = BASE_DIR / "webapp"

# ---------------------------------------------------------------- лимиты роллов
FREE_ROLLS_PER_DAY = 3           # бесплатных роллов в день
INVITE_ROLLS_CAP = 3             # максимум рероллов за инвайты в день

# ---------------------------------------------------------------- математика ролла
ROLL_SIGMA = 0.18                # std шума: ~90% попадают в ±0.3
CRIT_CHANCE = 0.02               # 2% крит
CRIT_BONUS_MIN = 0.5
CRIT_BONUS_MAX = 0.8
FAIL_CHANCE = 0.03               # 3% провал
FAIL_PENALTY = 0.5
SCORE_MIN = 1.0
SCORE_MAX = 5.0
BASE_MIN = 2.0                   # база из хэша фото
BASE_MAX = 4.4

# Границы тиров (по score; Mythic — только крит).
TIER_RARE = 2.5
TIER_EPIC = 3.2
TIER_LEGENDARY = 3.91

# TTL хранения байтов фото в памяти (не на диске!).
PHOTO_TTL_SECONDS = 15 * 60

# ---------------------------------------------------------------- дисклеймер
DISCLAIMER = (
    "MOG — развлекательная игра. Балл генерируется случайно, не является "
    "научной, медицинской или объективной оценкой внешности."
)
DISCLAIMER_SHORT = "MOG: развлекательная игра. Балл случаен, не оценка внешности."

# ---------------------------------------------------------------- каталог Stars
# (первый шаг монетизации; инвойсы подключим в фазе 2)
STARS_CATALOG = {
    "starter": {"title": "Стартовый паблик", "stars": 99, "rolls": 25, "crit": 1},
    "rolls_5": {"title": "Reroll x5", "stars": 25, "rolls": 5},
    "rolls_25": {"title": "Reroll x25", "stars": 99, "rolls": 25},
    "rolls_100": {"title": "Reroll x100", "stars": 299, "rolls": 100},
    "crit_token": {"title": "Крит-токен", "stars": 49, "crit": 1},
    "second_chance": {"title": "Второй шанс", "stars": 15, "rolls": 1},
}
