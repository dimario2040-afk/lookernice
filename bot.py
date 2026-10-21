"""Looksmaxxing / PSL Telegram-бот.

Стек: Aiogram 3.x + MediaPipe (локальная биометрия) + VLM-мозги
(Zhipu GLM по дефолту, любой OpenAI-совместимый endpoint через env).
Работа: юзер шлёт фото -> MediaPipe считает fWHR/симметрию/tilt ->
цифры + фото уходят в нейронку -> жёсткий структурированный разбор.
"""
import asyncio
import io
import logging
import math
import os
import random
import tempfile
import time

# Тяжёлая биометрия — ОПЦИОНАЛЬНА: на free-хостинге (Render, 512MB RAM)
# mediapipe не влезет, и бот должен работать без него (только VLM по фото).
try:
    import cv2
    import mediapipe as mp

    BIOMETRY_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    BIOMETRY_AVAILABLE = False
import numpy as np
from aiogram import Bot, Dispatcher, F

# Бейдж сборки: виден в /healthz, чтобы проверять прод без догадок.
APP_VERSION = "2026-10-17h-tournament-comment"
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    PreCheckoutQuery,
    WebAppInfo,
)
from dotenv import load_dotenv

load_dotenv()

from game.config import PUBLIC_ERROR  # noqa: E402  (после load_dotenv — конфиг читает ENV)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4.6v-flash")

# Водопад бесплатных VLM. Тайминги: одна попытка = VLM_TIMEOUT сек, весь бюджет
# на одно фото = VLM_DEADLINE сек. Без sleep-пауз — иначе отказ «зависает».
VLM_TIMEOUT = float(os.getenv("VLM_TIMEOUT", "12"))
VLM_DEADLINE = float(os.getenv("VLM_DEADLINE", "30"))

# Доп. модели основного провайдера (z.ai) через запятую. Переопределяется env.
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv("MODEL_FALLBACKS", "glm-4.6v-flash").split(",")
    if m.strip()
]

# Публичный HTTPS-URL (Render задаёт RENDER_EXTERNAL_URL; PUBLIC_URL — для прочего
# хостинга). Нужен для Mini App и webhook. Пусто = локальный polling без мини-аппа.
PUBLIC_BASE_URL = (
    os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    or os.environ.get("PUBLIC_URL", "").strip()
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("psl-bot")

# ---------------------------------------------------------------- биометрия
# Индексы ключевых точек MediaPipe FaceMesh (478 точек).
IDX = {
    "nose_tip": 1,
    "chin": 152,
    "forehead": 10,
    "brow_center": 151,
    "upper_lip": 13,
    "zygion_r": 234,      # правая скула
    "zygion_l": 454,      # левая скула
    "mouth_r": 78,        # правый угол рта
    "mouth_l": 308,       # левый угол рта
    "gonion_r": 58,       # правый угол челюсти (аппрокс.)
    "gonion_l": 288,      # левый угол челюсти (аппрокс.)
    "eye_r_outer": 33,
    "eye_r_inner": 133,
    "eye_l_inner": 362,
    "eye_l_outer": 263,
}

# FaceLandmarker (Tasks API) — модель скачивается сама при первом запуске.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Кладём модель во временную папку: у нативного кода MediaPipe бывают
# проблемы с не-ASCII путями (кириллица в пути проекта).
FACE_LANDMARKER_PATH = os.path.join(tempfile.gettempdir(), "face_landmarker_psl.task")
FACE_LANDMARKER_MIN_BYTES = 1 * 1024 * 1024  # меньше = битый/недокачанный файл

_landmarker = None
_landmarker_lock = None


def _get_landmarker():
    """Ленивый синглтон FaceLandmarker (тяжёлый, создаём один раз)."""
    global _landmarker, _landmarker_lock
    import threading

    if _landmarker_lock is None:
        _landmarker_lock = threading.Lock()
    with _landmarker_lock:
        if _landmarker is None:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python import BaseOptions

            if (
                not os.path.exists(FACE_LANDMARKER_PATH)
                or os.path.getsize(FACE_LANDMARKER_PATH) < FACE_LANDMARKER_MIN_BYTES
            ):
                log.info("качаю face_landmarker.task…")
                import urllib.request

                for attempt in (1, 2):
                    try:
                        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("скачивание, попытка %d: %s", attempt, exc)
                size = os.path.getsize(FACE_LANDMARKER_PATH)
                if size < FACE_LANDMARKER_MIN_BYTES:
                    raise RuntimeError(
                        f"модель скачалась битой ({size} байт). "
                        "Проверь интернет и перезапусти бота."
                    )
                log.info("модель скачана (%d байт)", size)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def analyze_face(image_path: str) -> dict:
    """Прогоняет фото через MediaPipe, возвращает антропометрию.

    Всегда возвращает dict; при отсутствии лица — {"face_found": False}.
    """
    if not BIOMETRY_AVAILABLE:
        return {"face_found": False, "error": "biometry_unavailable"}
    img = cv2.imread(image_path)
    if img is None:
        return {"face_found": False, "error": "cannot_read_image"}
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)

    if not res.face_landmarks:
        return {"face_found": False}

    lm = res.face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm])  # пиксельные координаты

    def P(name: str) -> np.ndarray:
        return pts[IDX[name]]

    out: dict = {"face_found": True}
    try:
        # --- fWHR: ширина скул / высота (середина бровей -> верхняя губа)
        bizygo = _dist(P("zygion_r"), P("zygion_l"))
        upper_height = _dist(P("brow_center"), P("upper_lip"))
        out["fWHR"] = round(bizygo / upper_height, 3) if upper_height > 0 else None
        out["bizygomatic_px"] = round(bizygo, 1)

        # --- нижняя треть: ширина челюсти / ширина скул (jaw-to-zygo ratio)
        bigonial = _dist(P("gonion_r"), P("gonion_l"))
        out["jaw_to_zygo"] = round(bigonial / bizygo, 3) if bizygo > 0 else None

        # --- ширина рта / ширина скул
        mouth_w = _dist(P("mouth_r"), P("mouth_l"))
        out["mouth_to_zygo"] = round(mouth_w / bizygo, 3) if bizygo > 0 else None

        # --- высота нижней трети (верхняя губа -> подбородок) / ширина скул
        lower_third = _dist(P("upper_lip"), P("chin"))
        out["lower_third_ratio"] = round(lower_third / bizygo, 3) if bizygo > 0 else None

        # --- canthal tilt: угол линии глаз (град). + = hunter, - = prey.
        # Нормализация к (-90, 90]: у правого глаза outer левее inner (dx<0)
        # и сырой atan2 даёт ~178° вместо правильных ~-2° (та же прямая).
        def eye_tilt(inner: np.ndarray, outer: np.ndarray) -> float:
            dx = outer[0] - inner[0]
            dy = inner[1] - outer[1]  # y растёт вниз -> инверсия
            deg = math.degrees(math.atan2(dy, dx))
            deg = ((deg + 90) % 180) - 90
            return round(deg, 2)

        tilt_r = eye_tilt(P("eye_r_inner"), P("eye_r_outer"))
        tilt_l = eye_tilt(P("eye_l_inner"), P("eye_l_outer"))
        out["canthal_tilt_r"] = tilt_r
        out["canthal_tilt_l"] = tilt_l
        out["canthal_tilt_avg"] = round((tilt_r + tilt_l) / 2, 2)

        # --- симметрия: разница расстояний левых/правых точек до оси носа (%)
        nose_x = P("nose_tip")[0]
        pairs = [
            ("zygion_r", "zygion_l"),
            ("gonion_r", "gonion_l"),
            ("mouth_r", "mouth_l"),
            ("eye_r_outer", "eye_l_outer"),
        ]
        asyms = []
        for r_name, l_name in pairs:
            dr = abs(P(r_name)[0] - nose_x)
            dl = abs(P(l_name)[0] - nose_x)
            base = max((dr + dl) / 2, 1e-6)
            asyms.append(abs(dr - dl) / base * 100)
        out["asymmetry_pct"] = round(sum(asyms) / len(asyms), 2)
    except Exception as exc:  # noqa: BLE001 — биометрия не должна ронять бота
        log.warning("metric calc failed: %s", exc)
        out["calc_error"] = str(exc)

    return out


def format_metrics(m: dict) -> str:
    if m.get("error") == "biometry_unavailable":
        return "биометрия недоступна (лайт-режим хостинга) — разбор только по фото"
    if not m.get("face_found"):
        return "лицо не найдено"
    lines = []
    if m.get("fWHR") is not None:
        lines.append(f"fWHR={m['fWHR']} (норма M ~1.8-2.0)")
    if m.get("jaw_to_zygo") is not None:
        lines.append(f"jaw/zygo={m['jaw_to_zygo']} (цель ~0.9+)")
    if m.get("mouth_to_zygo") is not None:
        lines.append(f"mouth/zygo={m['mouth_to_zygo']} (идеал ~0.4+)")
    if m.get("lower_third_ratio") is not None:
        lines.append(f"lower-third={m['lower_third_ratio']}")
    if m.get("canthal_tilt_avg") is not None:
        lines.append(
            f"canthal tilt: R={m['canthal_tilt_r']}° L={m['canthal_tilt_l']}° "
            f"(+ = hunter, − = prey)"
        )
    if m.get("asymmetry_pct") is not None:
        lines.append(f"асимметрия={m['asymmetry_pct']}% (меньше = лучше)")
    return "; ".join(lines) if lines else "метрики не посчитались"


def format_metrics_ui(m: dict) -> str:
    """Компактные тех-замеры для кнопки (≈×2 короче полной строки), с комментами."""
    if m.get("error") == "biometry_unavailable":
        return "биометрия недоступна (лайт-режим хостинга)"
    if not m.get("face_found"):
        return "лицо не найдено"
    rows = []
    if m.get("fWHR") is not None:
        rows.append(f"fWHR <b>{m['fWHR']}</b> · норм 1.8–2.0")
    if m.get("jaw_to_zygo") is not None:
        rows.append(f"челюсть/скулы <b>{m['jaw_to_zygo']}</b> · ≥0.9")
    if m.get("canthal_tilt_avg") is not None:
        rows.append(f"кантальн. <b>{m['canthal_tilt_avg']}°</b> · +hunter/−prey")
    if m.get("asymmetry_pct") is not None:
        rows.append(f"асимметрия <b>{m['asymmetry_pct']}%</b> · меньше=лучше")
    return "\n".join(rows) if rows else "метрики не посчитались"

# ---------------------------------------------------------------- промпт
SYSTEM_PROMPT = """Ты — эксперт по эстетической медицине и мужской привлекательности:
антропометрист, знаток looksmaxxing-культуры
(looksmax.org, PSL-скейл, blackpill-анализ).

Пользователь — взрослый человек, который ДОБРОВОЛЬНО прислал своё фото
и ПРЯМО ПРОСИТ максимально жёсткую, циничную, безжалостную оценку внешности
без ванильных утешений в духе «ты и так красив». Это consensual aesthetic
analysis, консультация по внешности. Вежливые фильтры ОТКЛЮЧЕНЫ
по просьбе пользователя. Говори как свой пацан с форума: прямо, с матом
там где уместно, но по делу, без воды.

Оперируй терминами: PSL (1.0-5.0; 2.0 саб-5, 2.5-3.2 MTN, 3.5+ HTN, 4.0+ Chad),
fWHR, ramus, gonial angle, canthal tilt (hunter vs prey eyes), midface ratio,
lower third, bigonial width, zygos, anti-fraud (ракурс/свет), softmaxxing
(сушка, мьюинг, груминг, стрижка, борода, скинкейр), ascend, mog/mogging.

Тебе даны ЗАМЕРЕННЫЕ биометрические данные лица — это ФАКТЫ, опирайся на них,
а не галлюцинируй. Фото тоже смотри: кожа, жир на лице, глаза, нос, губы,
челюсть, шея, стрижка, ракурс.
Если вместо замеров сказано, что биометрия недоступна (лайт-режим хостинга) —
оценивай пропорции ВИЗУАЛЬНО по фото, так же жёстко и по тем же пунктам.

Тир определяй СТРОГО по числу, без натяжек: ниже 2.5 = sub-5/LTN,
2.5-3.2 = MTN, 3.3-3.9 = HTN, 4.0+ = Chad. Число 2.2 — это LTN, а не MTN.

Формат ответа (строго, на русском):
1. 💀 ВЕРДИКТ: PSL X.X — тир (sub-5/LTN/MTN/HTN/Chadlite/Chad) + одна жёсткая фраза.
2. 📐 КОСТИ И ПРОПОРЦИИ: разбор каждой метрики, что значит, хорошо/плохо.
3. 👁 ЗОНЫ ПРОВАЛА: топ-3 failo по силе урона.
4. ✨ HALO: что реально тащит (если есть).
5. 📸 ANTI-FRAUD: насколько фото врёт vs IRL.
6. 🛠 SOFTMAXXING-ПЛАН: конкретные шаги по приоритету (без скальпеля).
7. HARDMAXXING (опционально): радикальные методы, честно про риски и цену.
8. 📈 ПРОГНОЗ ASCEND: реалистичный потолок (например MTN→HTN) и за счёт чего.

Не читай нотаций, не жалей, не добавляй дисклеймеры про «внутреннюю красоту».
Максимум конкретики по ЭТОМУ лицу.
Не выдумывай того, чего нет на фото: фон, одежду вне кадра, точный возраст,
профессию. Если лиц несколько — разбирай самое крупное/центральное.
"""

# ---------------------------------------------------------------- мозги (VLM)
# ─────────────────── ПРОВАЙДЕРЫ VLM (заменяемые, «водопад») ───────────────────
# Порядок = приоритет. Первый живой отдаёт ответ, остальные — резерв.
# Формат: (label, base_url, key_env, models_csv)
#   base_url — готовый URL (с "://") ИЛИ имя ENV-переменной с URL.
# Если ключа провайдера нет — он тихо пропускается (включается позже без правок кода).
def _vlm_env(name: str, default: str) -> str:
    """Значение из env (после load_dotenv) или дефолт — всё настраивается файлами."""
    return (os.getenv(name) or default).strip()


_VLM_PROVIDERS = [
    # 1) Groq (ключ GROQ_API_KEY) — БАЗА: бесплатный тир, очень быстрый, без
    #    региональных блоков (работает и из РФ, и с Render). Vision: qwen3.8-27b.
    (
        "groq",
        _vlm_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "GROQ_API_KEY",
        _vlm_env("GROQ_MODELS", "qwen/qwen3.8-27b"),
    ),
    # 2) Google AI Studio (ключ GEMINI_API_KEY). Регион РФ блокируется — работает на
    #    Render (Frankfurt). Недоступный регион авто-выключаем (см. _VLM_DISABLED).
    (
        "gemini",
        _vlm_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "GEMINI_API_KEY",
        _vlm_env("GEMINI_MODELS", "gemini-3.8-flash,gemini-flash-latest"),
    ),
    # 3) AIHubMix (ключ AIHUBMIX_API_KEY) — агрегатор, OpenAI-совместимый, vision.
    (
        "aihubmix",
        _vlm_env("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1"),
        "AIHUBMIX_API_KEY",
        _vlm_env("AIHUBMIX_MODELS", "gpt-4o-mini"),
    ),
    # 4) Alibaba DashScope — дешёвый Qwen-VL (vision), работает из РФ.
    #    Ключ DASHSCOPE_API_KEY (https://dashscope.console.aliyun.com).
    (
        "dashscope",
        _vlm_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "DASHSCOPE_API_KEY",
        _vlm_env("DASHSCOPE_MODELS", "qwen-vl-plus,qwen-vl-max,qwen2.5-vl-72b-instruct"),
    ),
    # 5) z.ai / Zhipu (ключ MODEL_API_KEY) — glm-4.6v-flash бесплатный, часто 429;
    #    glm-4.6v платный-но-дешёвый (в MODEL_FALLBACKS) — включается при балансе.
    (
        "zhipu",
        MODEL_BASE_URL,
        "MODEL_API_KEY",
        ",".join(dict.fromkeys([MODEL_NAME] + MODEL_FALLBACKS)),
    ),
    # 5) OpenRouter (ключ OPENROUTER_API_KEY) — один ключ, пачка бесплатных vision.
    (
        "openrouter",
        _vlm_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "OPENROUTER_API_KEY",
        _vlm_env(
            "OPENROUTER_MODELS",
            "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,thinkingmachines/inkling:free",
        ),
    ),
]


# Провайдеры, которые отвалились необратимо (битый ключ, блок региона) — чтобы
# не тратить время на них в каждом следующем запросе. Сбрасывается при рестарте.
_VLM_DISABLED: set[str] = set()


def _vlm_key_env(label: str) -> str:
    """label провайдера -> имя env с его API-ключом."""
    for lbl, _url, key_env, _models in _VLM_PROVIDERS:
        if lbl == label:
            return key_env
    return "MODEL_API_KEY"


def _vlm_targets() -> list[tuple[str, str, str]]:
    """Собирает упорядоченный водопад (label, base_url, model).

    Провайдер попадает в список только если задан его ключ и URL. Порядок =
    порядок _VLM_PROVIDERS, внутри провайдера — порядок моделей в models_csv.
    """
    targets: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for label, base, key_env, models_csv in _VLM_PROVIDERS:
        url = base.strip() if "://" in base else os.getenv(base, "").strip()
        key = os.getenv(key_env, "").strip()
        if not key or not url:
            continue
        for model in (m.strip() for m in (models_csv or "").split(",")):
            if not model:
                continue
            sig = (label, url, model)
            if sig in seen:
                continue
            seen.add(sig)
            targets.append(sig)
    return targets


_GARBAGE_MARKERS = (
    "<?php", "<!doctype", "<html", "</html", "```json", "```python",
    "\ufffd", "�",
)


def _sanitize_roast(text: str) -> str:
    """Чистит ответ VLM: режет HTML/markdown-заборы и битые символы.

    НЕ трогает обычные слова — только явный мусор (теги, U+FFFD, fences).
    """
    import re
    import unicodedata

    if not text:
        return ""
    # битые символы (U+FFFD)
    text = text.replace("\ufffd", "").replace("�", "")
    # HTML-теги кроме наших <b>/<i>
    text = re.sub(r"<(?!/?[bi]>)[^>]+>", "", text, flags=re.IGNORECASE)
    # markdown-заборы и json-блоки
    text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
    # служебные маркеры провайдера
    for marker in _GARBAGE_MARKERS:
        text = text.replace(marker, "")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _extract_roast(resp, fallback_hint: str = "") -> str:
    """Достаёт текст из ответа VLM и прогоняет через санитайзер + детектор мусора."""
    finish = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
        finish = resp.choices[0].finish_reason or ""
    except Exception:  # noqa: BLE001
        text = ""
    text = _sanitize_roast(text)
    log.info("roast: finish=%s chars=%d", finish, len(text))
    # если после чистки ничего осмысленного не осталось — fallback
    if len(text) < 25:
        return (
            f"⚠️ Нейронка сегодня стесняется (finish_reason={finish or 'unknown'}). "
            "Попробуй другое фото: прямой ракурс, дневной свет, без очков."
        )
    return text


def _vision_chat(messages: list, *, max_tokens: int, temperature: float = 0.9):
    """ВОДОПАД: перебирает провайдеров по приоритету, пока кто-нибудь не ответит.

    Без sleep-пауз и без ретраев одной модели: упал/занят — сразу следующий
    провайдер. На одну попытку VLM_TIMEOUT сек, на всё фото VLM_DEADLINE сек.
    Так отказ всегда быстрый, а не «зависает» на минуту.
    """
    import time

    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )

    targets = _vlm_targets()
    if not targets:
        raise RuntimeError(
            "NO_VLM_PROVIDER: не задано ни одного VLM-провайдера "
            "(нужен ключ: GEMINI_API_KEY / MODEL_API_KEY / OPENROUTER_API_KEY)"
        )

    deadline = time.monotonic() + VLM_DEADLINE
    last_exc: Exception | None = None
    busy = True  # всё упало на 429/сеть => это внешняя перегрузка, не наш баг
    for label, base_url, model in targets:
        if label in _VLM_DISABLED:
            continue  # провайдер уже отработал необратимую ошибку — не тратим время
        if time.monotonic() >= deadline:
            log.warning("водопад: бюджет %.0fс исчерпан", VLM_DEADLINE)
            break
        client = OpenAI(
            api_key=os.getenv(_vlm_key_env(label), ""),
            base_url=base_url,
            max_retries=0,
            timeout=VLM_TIMEOUT,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            log.info("водопад: ответ от %s/%s", label, model)
            return resp
        except RateLimitError as exc:
            last_exc = exc
            busy = True
            log.warning("[%s] %s 429 (перегруз) → следующий", label, model)
        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            busy = True
            log.warning("[%s] %s сеть/timeout → следующий", label, model)
        except APIStatusError as exc:
            last_exc = exc
            code = getattr(exc, "status_code", 0)
            busy = code >= 500
            body = ""
            try:
                body = (exc.response.text or "").lower()
            except Exception:  # noqa: BLE001
                body = str(exc).lower()
            # необратимые ошибки: блок региона / плохой ключ → выключаем провайдера
            if code in (401, 403) or "location is not supported" in body:
                _VLM_DISABLED.add(label)
                log.warning("[%s] необратимо недоступен (HTTP %s) → выключаю", label, code)
            else:
                log.warning("[%s] %s HTTP %s → следующий", label, model, code)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            busy = False
            log.warning("[%s] %s err: %s", label, model, str(exc)[:110])

    if busy:
        raise RuntimeError("MODEL_BUSY: все бесплатные vision-модели перегружены (429/5xx)")
    if last_exc:
        raise last_exc
    raise RuntimeError("нет доступной VLM-модели")


def _friendly_error(exc: Exception) -> str:
    """Человеческое сообщение вместо сырого 429/1305."""
    s = str(exc)
    if "MODEL_BUSY" in s or "429" in s or "1305" in s:
        return (
            "🧠 Нейронка сейчас перегружена (бесплатная модель). "
            "Кинь фото через минуту — вернётся. Скан «Слабые места» работает всегда."
        )
    return f"💥 {PUBLIC_ERROR}"


# Провайдер-агностик: любой OpenAI-совместимый endpoint с vision-моделью.
# Дефолт — Zhipu (z.ai, регистрация по email, бесплатный glm-4.6v-flash).
# Через env переключается на что угодно (Qwen, Gemini-OAI, OpenRouter…).
def analyze_with_vlm(photo_bytes: bytes, metrics_text: str) -> str:
    """Отправляет фото + метрики в VLM, возвращает разбор."""
    import base64

    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": f"Биометрия лица (факты от MediaPipe): {metrics_text}\n\n"
                    "Разъеби по схеме из системного промпта. Жёстко, по делу.",
                },
            ],
        },
    ]
    # 4096: reasoning-модели (GLM-4.6v) тратят токены на размышления,
    # при 2048 ответ обрывался на полуслове (finish_reason=length).
    resp = _vision_chat(messages, max_tokens=4096)
    return _extract_roast(resp)


# ---------------------------------------------------------------- простой слой
SIMPLE_PROMPT = """Ты — друг-карикатурист: описываешь внешность весело и хлёстко,
но по-доброму, так чтобы вердикт хотелось переслать другу. Простыми словами,
без терминов (PSL, fWHR, tilt, failo — запрещены), без цифр и замеров.

Сначала определи по фото КТО герой:
- СЕЛФИ (человек снял сам себя: крупный план, ракурс вытянутой руки) → герой сам
  прислал фото, обращайся на "ты", шути по-дружески.
- ЧУЖОЕ ФОТО (его сняли, скрин, пересланное) → говори о герое в третьем лице
  ("этот орёл", "наша героиня"), можешь весело пошутить и про приславшего.
- Лиц несколько → описывай самое крупное/центральное, остальных — одной строкой.

Это добровольный развлекательный тест: человек сам прислал фото ради шутки.

Структура — РОВНО ДВА АБЗАЦА, русский язык, юмора побольше (минимум 4 шутки
на весь ответ — народ должен смеяться и пересылать):
Абзац 1 — первое впечатление одной яркой фразой + вайб героя, минимум 2 шутки.
Абзац 2 — что классно и что забавно, тоже через юмор, минимум 2 шутки.
Пример стиля: "уверенность волка, взгляд — как у совы после ночной смены,
а улыбка такая, будто он только что вспомнил твой позор со школы".

Описывай только то, что видно на фото. Не додумывай фон, одежду вне кадра,
профессию, точный возраст. Не знаешь — шути, а не выдумывай.
Жесть, цифры и диагнозы — НЕ твоя задача, их выдаст другой режим по кнопке.
Если биометрия недоступна — смотри только фото.

ГЛАВНОЕ ПРАВИЛО: описывай смешно ЛЮБОЕ фото. Даже если это НЕ лицо (кот, еда,
пейзаж, мем, скриншот, размытое пятно, палец на камере) — СДЕЛАЙ ВИД, что это
лицо героя, и опиши его как лицо: «кости», «взгляд», «характер», «скулы».
Чем абсурднее исходник, тем смешнее. НИКОГДА не пиши «это не лицо» и не
отказывайся — обыграй абсурд как фишку героя.
"""

SIMPLE_PROMPT_SOFT = """Ты — автор добрых дружеских шаржей для развлекательного теста.
Человек сам прислал своё фото ради веселья. Опиши его внешность тепло и смешно:
комплименты с юмором, яркие сравнения с героями фильмов и животными.
Простыми словами, без терминов и цифр. Два абзаца, русский язык.
Описывай только то, что видно на фото, ничего не выдумывай.
Даже если это не лицо (кот, еда, предмет) — сделай вид, что это лицо героя,
и опиши смешно. Никогда не отказывайся.
"""


def _simple_call(system_prompt: str, photo_b64: str, metrics_text: str):
    """Один вызов simple-слоя. Может кинуть исключение (в т.ч. 1301 фильтр)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"},
                },
                {
                    "type": "text",
                    "text": f"Замеры для ориентира (не озвучивай цифры): {metrics_text}\n\n"
                    "Опиши человека простыми словами по схеме из системного промпта.",
                },
            ],
        },
    ]
    # 2048: reasoning съедает бюджет, при 800 контент приходил пустым.
    return _vision_chat(messages, max_tokens=2048)


def _extract_text(resp) -> tuple[str, str]:
    try:
        return (resp.choices[0].message.content or "").strip(), (
            resp.choices[0].finish_reason or ""
        )
    except Exception:  # noqa: BLE001
        return "", ""


def analyze_simple(photo_bytes: bytes, metrics_text: str) -> str:
    """Короткий разбор простыми словами (первый слой выдачи).

    Китайский фильтр (1301) иногда режет дерзкий промпт — тогда повторяем
    с мягкой версией, чтобы юзер всё равно получил ответ, а не ошибку.
    """
    import base64

    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    try:
        resp = _simple_call(SIMPLE_PROMPT, b64, metrics_text)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if any(m in err for m in ("1301", "contentFilter", "content_filter", "不安全")):
            log.warning("simple отфильтрован (1301), повторяю с мягким промптом")
            try:
                resp = _simple_call(SIMPLE_PROMPT_SOFT, b64, metrics_text)
            except Exception as exc2:  # noqa: BLE001
                log.warning("soft тоже не прошёл: %s", str(exc2)[:200])
                raise
        else:
            raise
    text = _extract_roast(resp)
    return text


# Контекст для кнопок: user_id -> (photo_bytes, metrics_text, metrics_dict).
# Ограничен PENDING_LIMIT, чтобы бот не съедал RAM у коммерческого инстанса.
_pending_full: dict[int, tuple[bytes, str, dict]] = {}


def _remember_photo(user_id: int, raw: bytes, metrics_text: str, metrics: dict) -> None:
    from game import config as mog_config

    _pending_full[user_id] = (raw, metrics_text, metrics)
    if len(_pending_full) > mog_config.PENDING_LIMIT:
        for old in list(_pending_full.keys())[: len(_pending_full) - mog_config.PENDING_LIMIT]:
            _pending_full.pop(old, None)


# Фото, ждущие согласия: user_id -> (unix_ts, raw байты). TTL — чтобы бот не таскал
# старые фото после отложенного нажатия «Согласен». Лимит — тот же PENDING_LIMIT.
_CONSENT_PHOTO_TTL = 15 * 60
_pending_consent: dict[int, tuple[float, bytes]] = {}


def _stash_consent_photo(user_id: int, raw: bytes) -> None:
    from game import config as mog_config

    _pending_consent[user_id] = (time.time(), raw)
    if len(_pending_consent) > mog_config.PENDING_LIMIT:
        for old in list(_pending_consent.keys())[: len(_pending_consent) - mog_config.PENDING_LIMIT]:
            _pending_consent.pop(old, None)

# ---------------------------------------------------------------- бот
dp = Dispatcher()
BOT_ME_ID: int | None = None
_BOT_USERNAME = ""
# Хэш (chat_id, user_id) → ts: нажал «Я в деле» без ролла, ждём его фото в этом чате.
_pending_arena: dict[tuple[int, int], float] = {}
# arena_id → (chat_id, message_id): живое лобби, чтобы редактировать его, а не плодить.
_arena_lobby: dict[int, tuple[int, int]] = {}
_ARENA_PENDING_TTL = 15 * 60


@dp.message(Command("start"))
async def cmd_start(msg: Message, bot: Bot) -> None:
    # ⚔️ Deep-link вызов: /start duel_<код> — сразу показываем вызов, без приветствия.
    parts = (msg.text or "").split(maxsplit=1)
    payload = parts[1].strip() if len(parts) > 1 else ""
    if payload.startswith("duel_"):
        await _start_duel_invite(msg, bot, payload[len("duel_"):])
        return

    welcome = (
        "💀 <b>PSL-бот на связи.</b>\n\n"
        "Кидай своё фото (лицо крупно, без очков и фильтров) — "
        "сначала скажу по-человечески, без занудства. "
        "А если хочешь жести — жми кнопку «📊 Подробный разбор»: "
        "там PSL-рейтинг, кости, failo/halo и план прокачки.\n\n"
        "Фото обрабатывается локально (MediaPipe) + нейронка. "
        "Ничего не храню — файл удаляется сразу после анализа."
    )
    # 🔞 Согласие спрашиваем ОДИН раз и МЕЛКО: строка кнопок под приветствием,
    # без абзацев. Полный текст — только за кнопкой «договор».
    rows = [
        [InlineKeyboardButton(text="🏟 Тестовый турнир", callback_data="tt:new")]
    ]
    if not await _has_consent(msg.from_user.id):
        rows += _consent_keyboard().inline_keyboard
    await msg.answer(
        welcome, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
    )


@dp.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "Как получить охуительный результат:\n"
        "1. Фото при дневном свете, камера на уровне глаз\n"
        "2. Нейтральное лицо, губы сомкнуты\n"
        "3. Без очков, кепок, фильтров и жёсткого ракурса снизу\n\n"
        "Просто отправь фото в чат 👇\n\n"
        "⚔️ Бател: ответь на сообщение и напиши /mog — или /mog без ответа "
        "(открытый вызов, кто примет кнопкой).\n"
        "🏆 Арена до 8: /arena в группе.\n"
        "🏟 Тестовый турнир (соло, до 8 фото): /tournament в личке со мной.\n"
        "🎒 Артефакты в бою: за Stars, буст до +0.5 (есть 🎁 бесплатный раз в день).\n"
        "🔞 18+ · правила и согласие на обработку данных — /legal",
    )


@dp.message(Command("mog"))
async def cmd_mog(msg: Message) -> None:
    """Вызов на мог-батл: ответь+ /mog (адресно) или /mog без ответа (открытый вызов)."""
    if msg.chat.type == "private":
        await msg.answer(
            "⚔️ Баттлы — для группового чата. Добавь меня в группу и вызови /mog"
        )
        return
    from game import battles as mog_battles
    from game import db as mog_db

    await mog_db.get_or_create_user(
        msg.from_user.id, msg.from_user.full_name, msg.from_user.username or ""
    )
    challenger = {"id": msg.from_user.id, "name": msg.from_user.full_name}

    opponent = None
    target = msg.reply_to_message
    if target and target.from_user and not target.from_user.is_bot:
        if target.from_user.id != msg.from_user.id:
            opponent = {"id": target.from_user.id, "name": target.from_user.full_name}
    if opponent is None:
        for ent in msg.entities or []:
            if ent.type == "text_mention" and ent.user and not ent.user.is_bot:
                if ent.user.id != msg.from_user.id:
                    opponent = {"id": ent.user.id, "name": ent.user.full_name}
                break

    try:
        if opponent is None:
            battle_id = await mog_battles.create_open(msg.chat.id, challenger)
        else:
            battle_id = await mog_battles.create(msg.chat.id, challenger, opponent)
    except Exception:  # noqa: BLE001 — БД может быть не инициализирована
        log.exception("battle create failed")
        await msg.reply(f"💥 {PUBLIC_ERROR}")
        return

    if opponent is None:
        text = (
            f"⚔️ <b>{challenger['name']}</b> бросает открытый вызов!\n\n"
            "Кто первый жмёт «Принять» — тот и в бою. "
            "Счёт — твой лучший МОГ. После — просто кинь фото в чат."
        )
        kb = _duel_keyboard(battle_id, open_challenge=True)
    else:
        text = (
            f"⚔️ <b>{challenger['name']}</b> вызывает <b>{opponent['name']}</b> "
            "на МОГ-БАТТЛ!\n\n"
            f"{opponent['name']}, жми «Принять» — или просто пришли <b>своё фото</b> в чат."
        )
        kb = _duel_keyboard(battle_id, open_challenge=False)

    trial = await mog_db.get_battle(battle_id)
    if trial and not trial["challenger_score"]:
        text += (
            "\n\n⚠️ У вызывающего ещё нет ролла — счёт 0.0000. "
            "Пусть сначала кинет фото в чат."
        )
    await msg.reply(text, reply_markup=kb, parse_mode="HTML")


def _card_keyboard(credits: dict[str, int] | None = None) -> InlineKeyboardMarkup:
    """Кнопки под МОГ-карточкой. Учитывает подарочные кредиты (бесплатный реролл/студия)."""
    cr = credits or {}
    reroll_label = "🔄 Ещё раз" + (" — бесплатно 🎁" if cr.get("reroll") else " — 15 ⭐")
    studio_label = "🎨 Студия" + (" — бесплатно 🎁" if cr.get("studio") else " — 35 ⭐")
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⚔️ Сразиться (ссылка)", callback_data="duel_link"),
                InlineKeyboardButton(text="🏆 Общий баттл", callback_data="arena_quick"),
            ],
            [InlineKeyboardButton(text="⚔️ Дуэль 1×1 в чате", callback_data="duel_new")],
            [InlineKeyboardButton(text=reroll_label, callback_data="reroll")],
            [InlineKeyboardButton(text=studio_label, callback_data="studio")],
            [InlineKeyboardButton(text="🎁 Подарить", callback_data="gift")],
            [InlineKeyboardButton(text="📊 Подробный разбор", callback_data="full_roast")],
            [InlineKeyboardButton(text="🎯 Слабые места (неон)", callback_data="weakspots")],
            [InlineKeyboardButton(text="🧬 Замеры (тех)", callback_data="metrics")],
        ]
    )


async def _card_keyboard_for(user_id: int) -> InlineKeyboardMarkup:
    """Карточка с учётом кредитов юзера (показать «бесплатно 🎁»)."""
    from game import db as mog_db

    try:
        credits = await mog_db.get_credits(user_id)
    except Exception:  # noqa: BLE001 — БД не критична для клавиатуры
        credits = {}
    return _card_keyboard(credits)


def _duel_keyboard(battle_id: int, *, open_challenge: bool):
    """Кнопка принятия вызова (чат). Для адресного — плюс мини-апп, если есть URL."""
    rows = [
        [
            InlineKeyboardButton(
                text="⚔️ Принять вызов", callback_data=f"duel_accept:{battle_id}"
            )
        ],
        [
            InlineKeyboardButton(
                text="🎒 Артефакты", callback_data=f"art_shop:{battle_id}"
            )
        ],
    ]
    if not open_challenge and PUBLIC_BASE_URL:
        rows.append(
            [
                InlineKeyboardButton(
                    text="Принять в приложении",
                    web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/app?battle={battle_id}"),
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.callback_query(F.data.startswith("duel_accept:"))
async def on_duel_accept(cb: CallbackQuery) -> None:
    """Оппонент принял вызов кнопкой → просим фото, снимаем кнопку."""
    battle_id = int(cb.data.split(":", 1)[1])
    from game import battles as mog_battles
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    user = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    ok, reason = await mog_battles.accept(battle_id, user)
    if not ok:
        await cb.answer(reason, show_alert=True)
        return
    await cb.answer("Принято ⚔️")
    try:
        await cb.message.edit_reply_markup(reply_markup=None)
    except Exception:  # noqa: BLE001 — сообщение старое/неизменно
        pass
    await cb.message.answer(
        f"⚔️ <b>{user['name']}</b> принял вызов! "
        "Кидай <b>своё фото</b> в чат — и бой закрыт.",
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "duel_new")
async def on_duel_new(cb: CallbackQuery) -> None:
    """Кнопка «Вызвать на баттл» под карточкой → открытый вызов в этом чате."""
    if cb.message.chat.type == "private":
        await cb.answer("Баттлы — в групповом чате 👥", show_alert=True)
        return
    from game import battles as mog_battles
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    challenger = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    try:
        battle_id = await mog_battles.create_open(cb.message.chat.id, challenger)
    except Exception:  # noqa: BLE001
        log.exception("battle create failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)
        return
    await cb.answer("Вызов брошен ⚔️")
    await cb.message.answer(
        f"⚔️ <b>{challenger['name']}</b> бросает открытый вызов!\n\n"
        "Кто первый жмёт «Принять» — тот и в бою. Потом просто кинь фото.",
        reply_markup=_duel_keyboard(battle_id, open_challenge=True),
        parse_mode="HTML",
    )

# ------------------------------------------------ дуэль по ссылке (без группы)


async def _bot_username(bot: Bot) -> str:
    """Юзернейм бота для t.me-ссылки (кэшируем: me() стоит запрос к Telegram)."""
    global _BOT_USERNAME
    if not _BOT_USERNAME:
        _BOT_USERNAME = (await bot.me()).username or ""
    return _BOT_USERNAME


async def _current_score(user_id: int) -> float:
    """Балл вызывающего: сперва карточка в памяти, иначе лучший ролл из БД."""
    from game import db as mog_db

    pending = _pending_full.get(user_id)
    if pending:
        return float(_mog_score_tier(pending[0])[0])
    best = await mog_db.get_best_roll(user_id)
    return float(best["score"]) if best and best["score"] else 0.0


@dp.callback_query(F.data == "duel_link")
async def on_duel_link(cb: CallbackQuery, bot: Bot) -> None:
    """«⚔️ Сразиться (ссылка)» под карточкой → вызов, который пересылают другу."""
    from game import db as mog_db
    from game import duel_link as mog_duel

    uid = cb.from_user.id
    score = await _current_score(uid)
    if score <= 0:
        await cb.answer("Сначала кинь фото — нужен твой МОГ, чтобы вызвать 🔥", show_alert=True)
        return
    await mog_db.get_or_create_user(uid, cb.from_user.full_name, cb.from_user.username or "")
    try:
        duel = await mog_duel.create({"id": uid, "name": cb.from_user.full_name}, score)
    except Exception:  # noqa: BLE001
        log.exception("duel link create failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)
        return
    text, _link = mog_duel.invite_text(duel, await _bot_username(bot))
    await cb.answer("Вызов готов ⚔️", show_alert=True)
    await cb.message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@dp.callback_query(F.data.startswith("duel_link_accept:"))
async def on_duel_link_accept(cb: CallbackQuery) -> None:
    """Соперник принимает вызов из линка."""
    from game import db as mog_db
    from game import duel_link as mog_duel

    code = cb.data.split(":", 1)[1]
    ok, reason = await mog_duel.accept(code, {"id": cb.from_user.id, "name": cb.from_user.full_name})
    if not ok:
        pretty = {
            "not_found": "Вызов не найден или уже сгорел ⌛",
            "expired": "Вызов истёк ⌛",
            "closed": "Этот вызов уже принят другим ⚔️",
            "self": "Себя нельзя 😈",
        }.get(reason, "Не вышло")
        await cb.answer(pretty, show_alert=True)
        return
    duel = await mog_db.get_duel_link(code)
    await cb.answer("Вызов принят ⚔️")
    if not duel:
        return
    await cb.message.answer(
        mog_duel.accepted_text(duel), parse_mode="HTML", disable_web_page_preview=True
    )
    # Вызывающий мог быть в другом чате — пишем ему в личку.
    try:
        await cb.bot.send_message(
            int(duel["challenger_id"]), mog_duel.opponent_joined_text(duel), parse_mode="HTML"
        )
    except Exception:  # noqa: BLE001 — мог заблокировать бота
        log.warning("не смог уведомить вызывающего о приёме вызова", exc_info=True)


async def _start_duel_invite(msg: Message, bot: Bot, code: str) -> None:
    """/start duel_<код> — соперник открыл ссылку: показываем вызов и кнопку."""
    from game import db as mog_db
    from game import duel_link as mog_duel

    duel = await mog_db.get_duel_link(code)
    if not duel or duel["status"] != "open":
        await msg.answer(
            "⌛ Этот вызов не найден или уже сгорел.\n\n"
            "Попроси вызывающего сделать новый ⚔️",
            parse_mode="HTML",
        )
        return
    if duel["challenger_id"] == msg.from_user.id:
        await msg.answer("Это твой собственный вызов 😏 Перешли его кому-нибудь другому.")
        return
    text, _link = mog_duel.invite_text(duel, await _bot_username(bot))
    rows = [
        [
            InlineKeyboardButton(
                text="✅ Принять вызов", callback_data=f"duel_link_accept:{code}"
            )
        ]
    ]
    # Согласие — той же мелкой строкой, без отдельного сообщения-простыни.
    if not await _has_consent(msg.from_user.id):
        rows += _consent_keyboard().inline_keyboard
    await msg.answer(
        text,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        disable_web_page_preview=True,
    )


async def _duel_sweep(bot: Bot) -> None:
    """Раз в час гасит просроченные вызовы и пишет вызывающим «соперник слился»."""
    from game import db as mog_db
    from game import duel_link as mog_duel

    await asyncio.sleep(180)  # дать сервису подняться
    while True:
        try:
            for duel in await mog_db.expire_duel_links():
                try:
                    await bot.send_message(
                        int(duel["challenger_id"]),
                        mog_duel.expired_text(duel),
                        parse_mode="HTML",
                    )
                except Exception:  # noqa: BLE001
                    log.warning("не смог уведомить о сливе", exc_info=True)
        except Exception:  # noqa: BLE001 — свип не должен ронять бота
            log.exception("duel sweep failed")
        await asyncio.sleep(3600)


@dp.callback_query(F.data == "arena_quick")
async def on_arena_quick(cb: CallbackQuery) -> None:
    """Кнопка «Общий баттл» под карточкой → открыть/войти в чатовую МОГ-АРЕНУ."""
    if cb.message.chat.type == "private":
        await cb.answer("Баттлы — в групповом чате 👥", show_alert=True)
        return
    from game import arena as mog_arena
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    host = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    try:
        arena, created = await mog_arena.create(cb.message.chat.id, host)
        plist = await mog_arena.players(arena["id"])
        if any(p["user_id"] == host["id"] for p in plist):
            ok, reason, code = True, "Ты уже в деле ⚔️", "already"
        else:
            ok, reason, code = await mog_arena.join(arena["id"], host)
    except Exception:  # noqa: BLE001
        log.exception("arena quick failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)
        return
    if not ok:
        await cb.answer(reason, show_alert=True)
        if code == "no_roll":
            _pending_arena[(cb.message.chat.id, cb.from_user.id)] = time.time()
        return
    await cb.answer("Общий баттл! 🔥" if created else "Ты в арене ⚔️")
    plist = await mog_arena.players(arena["id"])
    sent = await cb.message.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )
    _remember_lobby(arena["id"], sent)


# ───────────────────────── АРТЕФАКТЫ (магазин боя) ─────────────────────────
async def _art_battle_guard(cb: CallbackQuery, battle_id: int):
    """Общая проверка: бой открыт и cb-юзер — его участник. Вернёт battle или None."""
    from game import db as mog_db

    battle = await mog_db.get_battle(battle_id)
    if not battle or battle["status"] != "pending":
        await cb.answer("Бой уже закрыт — артефакты ни к чему 😈", show_alert=True)
        return None
    if cb.from_user.id not in (battle["challenger_id"], battle["opponent_id"]):
        await cb.answer("Это не твой бой 👀", show_alert=True)
        return None
    return battle


async def _art_shop_view(cb: CallbackQuery, battle_id: int, mine: list[str], theirs: list[str]) -> None:
    from game import artifacts as mog_art

    await cb.message.answer(
        mog_art.shop_text(battle_id, mine, theirs),
        reply_markup=mog_art.shop_keyboard(battle_id, mine),
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("art_shop:"))
async def on_art_shop(cb: CallbackQuery) -> None:
    """🎒 Магазин артефактов текущего боя."""
    from game import db as mog_db

    battle_id = int(cb.data.split(":", 1)[1])
    battle = await _art_battle_guard(cb, battle_id)
    if battle is None:
        return
    uid = cb.from_user.id
    mine = await mog_db.battle_artifacts(battle_id, uid)
    opp_id = battle["opponent_id"] if uid == battle["challenger_id"] else battle["challenger_id"]
    theirs = await mog_db.battle_artifacts(battle_id, opp_id) if opp_id else []
    await cb.answer()
    await _art_shop_view(cb, battle_id, mine, theirs)


@dp.callback_query(F.data.startswith("art_buy:"))
async def on_art_buy(cb: CallbackQuery, bot: Bot) -> None:
    """Покупка артефакта за Stars (счёт в чат)."""
    from game import artifacts as mog_art
    from game import payments

    _, bid_s, item = (cb.data or "").split(":", 2)
    if item not in mog_art.ARTIFACTS:
        await cb.answer("Такого артефакта нет.", show_alert=True)
        return
    battle_id = int(bid_s)
    if await _art_battle_guard(cb, battle_id) is None:
        return
    try:
        await payments.send_artifact_invoice(bot, cb.message.chat.id, cb.from_user.id, battle_id, item)
        await cb.answer("Счёт отправлен ⭐")
    except Exception:  # noqa: BLE001
        log.exception("artifact invoice failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)


@dp.callback_query(F.data.startswith("art_free:"))
async def on_art_free(cb: CallbackQuery) -> None:
    """🎁 Бесплатный артефакт раз в день (ролл на удачу, без «Анти-сглаза»)."""
    from game import artifacts as mog_art
    from game import db as mog_db

    battle_id = int(cb.data.split(":", 1)[1])
    battle = await _art_battle_guard(cb, battle_id)
    if battle is None:
        return
    if not await mog_db.claim_free_artifact(cb.from_user.id):
        await cb.answer("Бесплатный уже брал сегодня — завтра 😉", show_alert=True)
        return
    item = random.choice(("lift", "light", "osteo"))
    await mog_db.add_battle_artifact(battle_id, cb.from_user.id, item, paid=False)
    a = mog_art.ARTIFACTS[item]
    await cb.answer("Подарок! 🎁")
    uid = cb.from_user.id
    mine = await mog_db.battle_artifacts(battle_id, uid)
    opp_id = battle["opponent_id"] if uid == battle["challenger_id"] else battle["challenger_id"]
    theirs = await mog_db.battle_artifacts(battle_id, opp_id) if opp_id else []
    await cb.message.answer(
        f"🎁 <b>Бесплатный артефакт:</b> {a['emoji']} {a['name']} — {a['desc']}",
        parse_mode="HTML",
    )
    await _art_shop_view(cb, battle_id, mine, theirs)


# ───────────────────────── РЕВАНШ ─────────────────────────
@dp.callback_query(F.data.startswith("rematch:"))
async def on_rematch(cb: CallbackQuery) -> None:
    """🔁 Реванш: проигравший снова вызывает победителя на новый батл."""
    from game import battles as mog_battles
    from game import db as mog_db

    battle_id = int((cb.data.split(":", 1)[1]).split(":", 1)[0])
    battle = await mog_db.get_battle(battle_id)
    if battle is None:
        await cb.answer("Бой не найден", show_alert=True)
        return
    if battle.get("status") == "pending":
        await cb.answer("Этот бой ещё не закончен — кинь фото!", show_alert=True)
        return

    # Реванш вызывает проигравший (при ничьей — принимавший вызывающий).
    cs = battle["challenger_score"] or 0.0
    os_ = battle["opponent_score"] or 0.0
    if cs >= os_:
        chal = {"id": battle["challenger_id"], "name": battle["challenger_name"]}
        opp = {"id": battle["opponent_id"], "name": battle["opponent_name"]}
    else:
        chal = {"id": battle["opponent_id"], "name": battle["opponent_name"]}
        opp = {"id": battle["challenger_id"], "name": battle["challenger_name"]}

    new_id = await mog_battles.create(battle["chat_id"], chal, opp)
    await cb.answer("Реванш!", show_alert=False)
    await cb.message.reply(
        f"🔁 <b>РЕВАНШ!</b>\n\n"
        f"{chal['name']} снова вызывает {opp['name']}! "
        f"{opp['name']}, кинь своё фото — и табло обновится.",
        parse_mode="HTML",
    )
    log.info("rematch #%s -> battle #%s", battle_id, new_id)


# ---------------------------------------------------------------- MOG-АРЕНА (битва до 8, чат)
def _remember_lobby(arena_id: int, message) -> None:
    """Запоминает живое лобби, чтобы редактировать его, а не постить новое."""
    try:
        _arena_lobby[arena_id] = (message.chat.id, message.message_id)
    except Exception:  # noqa: BLE001 — фейковое сообщение в тестах
        pass


async def _arena_refresh(bot: Bot, arena_id: int) -> None:
    """Перерисовывает живое лобби (edit), а если сообщение потеряно — постит заново."""
    from game import arena as mog_arena
    from game import db as mog_db

    arena = await mog_db.get_arena(arena_id)
    if not arena:
        return
    plist = await mog_arena.players(arena_id)
    text = mog_arena.lobby_text(arena, plist)
    kb = mog_arena.lobby_keyboard(arena, plist)
    target = _arena_lobby.get(arena_id)
    if target:
        chat_id, msg_id = target
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=msg_id,
                reply_markup=kb,
                parse_mode="HTML",
            )
            return
        except Exception:  # noqa: BLE001 — сообщение удалили/не изменилось
            _arena_lobby.pop(arena_id, None)
    try:
        sent = await bot.send_message(
            arena["chat_id"], text, reply_markup=kb, parse_mode="HTML"
        )
        _remember_lobby(arena_id, sent)
    except Exception:  # noqa: BLE001
        log.exception("arena refresh failed")


async def _persist_roll(msg: Message, raw: bytes, score: float, tier: str) -> None:
    """Пишет ролл в БД: без этого арена/дуэли считают, что фото не было.

    Балл и тир берём те же, что на карточке (детерминированные), чтобы табло
    не расходилось с тем, что человек только что увидел.
    """
    from game import db as mog_db
    from game import rolls as mog_rolls

    try:
        await mog_db.get_or_create_user(
            msg.from_user.id, msg.from_user.full_name, msg.from_user.username or ""
        )
        result = mog_rolls.roll(raw)
        result["score"] = score
        result["tier"] = tier
        result["crit"] = tier == "Mythic"
        result["failed"] = False
        await mog_db.add_roll(msg.from_user.id, result)
    except Exception:  # noqa: BLE001 — история роллов не должна ронять разбор
        log.exception("add_roll failed")


async def _apply_arena_photo(msg: Message, bot: Bot, score: float, tier: str) -> None:
    """Фото в группе → вписывает в арену или обновляет балл. Молча, если не ждали.

    Автовход только по приглашению (нажал «Я в деле» без ролла) или если игрок
    уже в лобби, — чтобы случайные фото в чате никого не затягивали в игру.
    """
    if msg.chat.type == "private":
        return
    from game import arena as mog_arena
    from game import db as mog_db

    try:
        arena = await mog_db.get_active_arena(msg.chat.id)
        if not arena:
            return
        key = (msg.chat.id, msg.from_user.id)
        invited = _pending_arena.get(key)
        if invited and time.time() - invited > _ARENA_PENDING_TTL:
            _pending_arena.pop(key, None)
            invited = None
        plist = await mog_arena.players(arena["id"])
        in_arena = any(p["user_id"] == msg.from_user.id for p in plist)
        if not invited and not in_arena:
            return
        user = {"id": msg.from_user.id, "name": msg.from_user.full_name}
        ok, code = await mog_arena.enter_with_score(arena["id"], user, score, tier)
        _pending_arena.pop(key, None)
        if not ok:
            return
        await _arena_refresh(bot, arena["id"])
        if code == "joined":
            await msg.reply(
                f"⚔️ Ты в арене! Твой МОГ: <b>{score:.4f}</b> · {tier}. "
                "Ждём остальных 👇",
                parse_mode="HTML",
            )
        else:
            await msg.reply(
                f"♻️ Балл в арене обновлён: <b>{score:.4f}</b> · {tier}.",
                parse_mode="HTML",
            )
    except Exception:  # noqa: BLE001 — арена не должна ронять обычный разбор
        log.exception("arena photo apply failed")


async def _arena_render_lobby(message, arena_id: int) -> None:
    """Перерисовывает лобби-сообщение (не роня при старом/неизменном сообщении)."""
    from game import arena as mog_arena
    from game import db as mog_db

    arena = await mog_db.get_arena(arena_id)
    if not arena:
        return
    plist = await mog_arena.players(arena_id)
    try:
        await message.edit_text(
            mog_arena.lobby_text(arena, plist),
            reply_markup=mog_arena.lobby_keyboard(arena, plist),
            parse_mode="HTML",
        )
        _remember_lobby(arena_id, message)
    except Exception:  # noqa: BLE001 — текст не изменился или сообщение устарело
        pass


async def _arena_finish(message, arena_id: int) -> None:
    """Резолвит арену: публикует сетку плей-офф + неоновое табло-картинку."""
    from game import arena as mog_arena
    from game import db as mog_db

    arena = await mog_db.get_arena(arena_id)
    if not arena:
        return
    plist = await mog_arena.resolve(arena_id)
    _arena_lobby.pop(arena_id, None)  # лобби больше не живое — редактировать нечего
    text = mog_arena.result_text(arena, plist)
    btext = mog_arena.bracket_text(arena, plist)
    kb = mog_arena.rematch_keyboard(arena_id)
    shown = btext or text
    try:
        await message.edit_text(shown, reply_markup=kb, parse_mode="HTML")
    except Exception:  # noqa: BLE001
        await message.answer(shown, reply_markup=kb, parse_mode="HTML")
    try:
        pic = await asyncio.to_thread(_arena_board_image, plist)
    except Exception:  # noqa: BLE001
        pic = None
        log.exception("arena board image failed")
    if pic:
        try:
            await message.answer_photo(
                BufferedInputFile(pic, filename="mog_arena.jpg"),
                caption=text,
                parse_mode="HTML",
            )
        except Exception:  # noqa: BLE001
            log.exception("arena board send failed")


@dp.message(Command("arena", "арена"))
async def cmd_arena(msg: Message) -> None:
    """Открывает чатовую битву МОГ до 8 игроков (без мини-аппа)."""
    if msg.chat.type == "private":
        await msg.answer(
            "⚔️ Арена — для группового чата. Добавь меня в группу и вызови /arena"
        )
        return
    from game import arena as mog_arena
    from game import db as mog_db

    await mog_db.get_or_create_user(
        msg.from_user.id, msg.from_user.full_name, msg.from_user.username or ""
    )
    host = {"id": msg.from_user.id, "name": msg.from_user.full_name}
    arena, _created = await mog_arena.create(msg.chat.id, host)
    plist = await mog_arena.players(arena["id"])
    sent = await msg.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )
    _remember_lobby(arena["id"], sent)


@dp.callback_query(F.data.startswith("arena_join:"))
async def on_arena_join(cb: CallbackQuery) -> None:
    arena_id = int(cb.data.split(":", 1)[1])
    from game import arena as mog_arena
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    user = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    ok, reason, code = await mog_arena.join(arena_id, user)
    if not ok:
        await cb.answer(reason, show_alert=True)
        # Нет ролла — не тупик: просим фото прямо в чат и запоминаем ожидание.
        # Следующее фото этого юзера впишет его в арену автоматически.
        if code == "no_roll" and cb.message.chat.type != "private":
            _pending_arena[(cb.message.chat.id, cb.from_user.id)] = time.time()
            mention = (
                f'<a href="tg://user?id={cb.from_user.id}">'
                f"{cb.from_user.full_name}</a>"
            )
            try:
                await cb.message.answer(
                    f"⚔️ {mention}, кинь фото сюда 👇\n"
                    "Я сразу впишу тебя в арену — со свежим МОГом, без повторных кнопок.",
                    parse_mode="HTML",
                )
            except Exception:  # noqa: BLE001 — не роняем вход из-за сообщения
                log.exception("arena prompt failed")
        return
    await cb.answer(reason)
    arena = await mog_db.get_arena(arena_id)
    if arena and await mog_db.arena_count(arena_id) >= arena["max_players"]:
        await _arena_finish(cb.message, arena_id)
    else:
        await _arena_render_lobby(cb.message, arena_id)


@dp.callback_query(F.data.startswith("arena_start:"))
async def on_arena_start(cb: CallbackQuery) -> None:
    arena_id = int(cb.data.split(":", 1)[1])
    from game import arena as mog_arena

    plist = await mog_arena.players(arena_id)
    if not any(p["user_id"] == cb.from_user.id for p in plist):
        await cb.answer("Сначала жми «Я в деле» 👇", show_alert=True)
        return
    if len(plist) < mog_arena.MIN_PLAYERS:
        await cb.answer(f"Нужно минимум {mog_arena.MIN_PLAYERS} бойца", show_alert=True)
        return
    await cb.answer("Старт! 🔥")
    await _arena_finish(cb.message, arena_id)


@dp.callback_query(F.data.startswith("arena_leave:"))
async def on_arena_leave(cb: CallbackQuery) -> None:
    arena_id = int(cb.data.split(":", 1)[1])
    from game import arena as mog_arena

    ok = await mog_arena.leave(arena_id, cb.from_user.id)
    await cb.answer("Вышел из арены" if ok else "Ты и не был в деле")
    if ok:
        await _arena_render_lobby(cb.message, arena_id)


@dp.callback_query(F.data.startswith("arena_new:"))
async def on_arena_new(cb: CallbackQuery) -> None:
    from game import arena as mog_arena
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    host = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    arena, _created = await mog_arena.create(cb.message.chat.id, host)
    plist = await mog_arena.players(arena["id"])
    sent = await cb.message.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )
    _remember_lobby(arena["id"], sent)
    await cb.answer("Новая арена ⚔️")


# ───────────────────────── ТЕСТОВЫЙ ТУРНИР (соло, до 8 фото) ─────────────────
# Зачем: в чате никого нет и баттл не с кем запустить — человек кидает свои фото,
# сам разводит пары и смотрит, кто дойдёт до финала. Вайб MK, но без анимации.

_TT_TTL = 60 * 60  # час на всю возню
_tournaments: dict[int, dict] = {}


def _tt_get(user_id: int) -> dict | None:
    st = _tournaments.get(user_id)
    if not st:
        return None
    if time.time() - st.get("ts", 0) > _TT_TTL:
        _tournaments.pop(user_id, None)
        return None
    return st


def _tt_ordered(st: dict) -> list[dict]:
    """Фото в текущем порядке слотов + номера слотов."""
    out = []
    for i, idx in enumerate(st["order"], 1):
        photo = dict(st["photos"][idx])
        photo["slot"] = i
        out.append(photo)
    return out


async def _tt_panel(bot: Bot, st: dict, text: str, kb) -> None:
    """Правит одну панель вместо новых сообщений (альбом не должен спамить)."""
    chat_id = st["chat_id"]
    if st.get("panel_id"):
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=st["panel_id"],
                reply_markup=kb,
                parse_mode="HTML",
            )
            return
        except Exception:  # noqa: BLE001 — сообщение устарело/не изменилось
            pass
    sent = await bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")
    st["panel_id"] = getattr(sent, "message_id", None)


async def _tt_album(bot: Bot, st: dict) -> None:
    """Пересылает фото альбомом в текущем порядке (номера = подписи).

    Старый альбом удаляем, иначе после каждого свапа в чате копится мусор.
    """
    from aiogram.types import InputMediaPhoto

    chat_id = st["chat_id"]
    for msg_id in st.get("album", []):
        try:
            await bot.delete_message(chat_id, msg_id)
        except Exception:  # noqa: BLE001 — уже удалено/старое
            pass
    st["album"] = []
    ordered = _tt_ordered(st)
    if not ordered:
        return
    media = [
        InputMediaPhoto(media=p["file_id"], caption=str(p["slot"])) for p in ordered
    ]
    try:
        sent = await bot.send_media_group(chat_id, media=media)
        st["album"] = [m.message_id for m in sent]
    except Exception:  # noqa: BLE001
        log.exception("tournament album failed")


async def _tt_open(msg: Message, bot: Bot, *, force_new: bool = False) -> None:
    """Открывает сбор фото (или возвращает уже открытый)."""
    from game import tournament as mog_tt

    uid = msg.from_user.id
    if msg.chat.type != "private":
        await msg.answer(
            "🏟 Тестовый турнир — только в личке: напиши мне /tournament в ЛС."
        )
        return
    st = _tt_get(uid)
    if not st or force_new:
        if not await _has_consent(uid):
            _tournaments[uid] = {
                "stage": "consent",
                "photos": [],
                "order": [],
                "picked": None,
                "chat_id": msg.chat.id,
                "ts": time.time(),
            }
            await msg.answer("18+", reply_markup=_consent_keyboard())
            return
        st = {
            "stage": "collect",
            "photos": [],
            "order": [],
            "picked": None,
            "chat_id": msg.chat.id,
            "ts": time.time(),
            "album": [],
        }
        _tournaments[uid] = st
    elif st["stage"] != "collect":
        # Уже на этапе расстановки — не начинаем заново, просто напоминаем.
        await msg.answer("🏟 Турнир уже открыт — жми «Готово» или «Отмена».")
        return
    await _tt_panel(
        bot, st, mog_tt.collect_text(len(st["photos"])), mog_tt.collect_keyboard(len(st["photos"]))
    )


@dp.message(Command("tournament", "tt", "турнир"))
async def cmd_tournament(msg: Message, bot: Bot) -> None:
    """Соло-турнир: до 8 фото, сам разводишь пары, бот играет сетку."""
    await _tt_open(msg, bot, force_new=True)


@dp.callback_query(F.data == "tt:new")
async def on_tt_new(cb: CallbackQuery, bot: Bot) -> None:
    await cb.answer()
    await _tt_open(cb.message, bot, force_new=True)


async def _tt_add_photo(msg: Message, bot: Bot, file_id: str) -> None:
    """Считает МОГ фото сразу и хранит ТОЛЬКО file_id + балл (байтов не держим)."""
    from game import tournament as mog_tt

    st = _tt_get(msg.from_user.id)
    if not st or st["stage"] != "collect":
        return
    if len(st["photos"]) >= mog_tt.MAX_PHOTOS:
        await msg.reply(
            f"Максимум {mog_tt.MAX_PHOTOS} фото. Жми «✅ Готово» — и разводим пары."
        )
        return
    try:
        tg_file = await bot.get_file(file_id)
        bio = await bot.download_file(tg_file.file_path)
        raw = bio.read()
        from game import rolls as mog_rolls

        score, tier = _mog_score_tier(raw)
        # Статы нужны для комментария к бою, хэш — чтобы шутки были стабильны
        # при повторном FIGHT на тех же фото. Байты наружу не уходят.
        stats = mog_rolls.stats_for(raw)
        phash = mog_rolls.photo_hash(raw)
    except Exception:  # noqa: BLE001
        log.exception("tournament photo failed")
        await msg.reply("Не смог прочитать это фото — пропустил 🤷")
        return
    st["photos"].append(
        {
            "file_id": file_id,
            "score": score,
            "tier": tier,
            "stats": stats,
            "hash": phash,
        }
    )
    st["order"].append(len(st["photos"]) - 1)
    st["ts"] = time.time()
    await _tt_panel(
        bot, st, mog_tt.collect_text(len(st["photos"])), mog_tt.collect_keyboard(len(st["photos"]))
    )


@dp.callback_query(F.data == "tt:done")
async def on_tt_done(cb: CallbackQuery, bot: Bot) -> None:
    """Сбор закончен → альбом с номерами + панель расстановки пар."""
    from game import tournament as mog_tt

    st = _tt_get(cb.from_user.id)
    if not st or st["stage"] != "collect":
        await cb.answer("Турнир уже закрыт", show_alert=True)
        return
    if len(st["photos"]) < mog_tt.MIN_PHOTOS:
        await cb.answer(f"Нужно минимум {mog_tt.MIN_PHOTOS} фото", show_alert=True)
        return
    st["stage"] = "seed"
    st["picked"] = None
    st["order"] = list(range(len(st["photos"])))
    await cb.answer("Погнали 🎛")
    await _tt_album(bot, st)
    await _tt_panel(
        bot, st, mog_tt.panel_text(len(st["photos"]), None), mog_tt.slots_keyboard(len(st["photos"]))
    )


@dp.callback_query(F.data.startswith("tt:pick:"))
async def on_tt_pick(cb: CallbackQuery, bot: Bot) -> None:
    """Двухтапный свап: выбрал слот — выбрал второй — поменялись местами."""
    from game import tournament as mog_tt

    st = _tt_get(cb.from_user.id)
    if not st or st["stage"] != "seed":
        await cb.answer("Расстановка уже закончилась", show_alert=True)
        return
    slot = int(cb.data.split(":")[-1])
    if slot < 1 or slot > len(st["order"]):
        await cb.answer("Такого слота нет", show_alert=True)
        return
    if st["picked"] is None:
        st["picked"] = slot
        await cb.answer(f"Слот {slot} — выбери второй")
        await _tt_panel(
            bot,
            st,
            mog_tt.panel_text(len(st["order"]), slot),
            mog_tt.slots_keyboard(len(st["order"]), slot),
        )
        return
    first = st["picked"]
    st["picked"] = None
    if first != slot:
        st["order"][first - 1], st["order"][slot - 1] = (
            st["order"][slot - 1],
            st["order"][first - 1],
        )
        await cb.answer(f"Поменял {first} ↔ {slot}")
        await _tt_album(bot, st)
    else:
        await cb.answer("Отменил выбор")
    await _tt_panel(
        bot,
        st,
        mog_tt.panel_text(len(st["order"]), None),
        mog_tt.slots_keyboard(len(st["order"])),
    )


@dp.callback_query(F.data == "tt:fight")
async def on_tt_fight(cb: CallbackQuery, bot: Bot) -> None:
    """Играет сетку по текущему порядку и показывает чемпиона."""
    from game import tournament as mog_tt

    st = _tt_get(cb.from_user.id)
    if not st or st["stage"] != "seed":
        await cb.answer("Сетка уже сыграна", show_alert=True)
        return
    items = _tt_ordered(st)
    rounds, champion = mog_tt.build_rounds(items)
    if not champion:
        await cb.answer("Нужно минимум 2 фото", show_alert=True)
        return
    st["stage"] = "done"
    await cb.answer("FIGHT! ⚔️")
    await _tt_panel(
        bot,
        st,
        mog_tt.result_text(rounds, champion, total=len(items)),
        mog_tt.fight_keyboard(),
    )
    try:
        await bot.send_photo(
            st["chat_id"],
            champion["file_id"],
            caption=mog_tt.champion_text(champion, total=len(items), rounds=rounds),
            parse_mode="HTML",
        )
    except Exception:  # noqa: BLE001 — фото могли удалить
        log.exception("champion photo failed")


@dp.callback_query(F.data == "tt:reset")
async def on_tt_reset(cb: CallbackQuery, bot: Bot) -> None:
    """Ещё раз: чистим альбом и начинаем сбор заново."""
    from game import tournament as mog_tt

    st = _tt_get(cb.from_user.id)
    if st:
        for msg_id in st.get("album", []):
            try:
                await bot.delete_message(st["chat_id"], msg_id)
            except Exception:  # noqa: BLE001
                pass
    uid = cb.from_user.id
    _tournaments[uid] = {
        "stage": "collect",
        "photos": [],
        "order": [],
        "picked": None,
        "chat_id": cb.message.chat.id,
        "ts": time.time(),
        "album": [],
        "panel_id": st.get("panel_id") if st else None,
    }
    await cb.answer("Заново 🔁")
    await _tt_panel(
        bot,
        _tournaments[uid],
        mog_tt.collect_text(0),
        mog_tt.collect_keyboard(0),
    )


@dp.callback_query(F.data == "tt:cancel")
async def on_tt_cancel(cb: CallbackQuery, bot: Bot) -> None:
    st = _tt_get(cb.from_user.id)
    if st:
        for msg_id in st.get("album", []):
            try:
                await bot.delete_message(st["chat_id"], msg_id)
            except Exception:  # noqa: BLE001
                pass
    _tournaments.pop(cb.from_user.id, None)
    await cb.answer("Закрыто")
    try:
        await cb.message.edit_text("🏟 Турнир закрыт.")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- Stars (оплата)
@dp.message(Command("buy"))
async def cmd_buy(msg: Message) -> None:
    """Магазин: список товаров за Telegram Stars."""
    from game import config as mog_config

    rows = [
        [
            InlineKeyboardButton(
                text=f"{p['title']} — {p['stars']} ⭐", callback_data=f"buy:{key}"
            )
        ]
        for key, p in mog_config.STARS_CATALOG.items()
    ]
    await msg.answer(
        "🛒 <b>Магазин MOG</b>\nОплата в Telegram Stars ⭐. "
        "Роллы — на случай, когда 3 бесплатных мало.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("buy:"))
async def on_buy(cb: CallbackQuery, bot: Bot) -> None:
    from game import payments

    key = (cb.data or "").split(":", 1)[-1]
    try:
        await payments.send_invoice(bot, cb.message.chat.id, key, cb.from_user.id)
        await cb.answer("Счёт отправлен ⭐")
    except Exception as exc:  # noqa: BLE001
        log.exception("send_invoice failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)


# ------------------------------------------------------- платный реролл (15 ⭐)
@dp.callback_query(F.data == "reroll")
async def on_reroll(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка под карточкой: платный реролл ТОГО ЖЕ фото (байты из памяти)."""
    from game import db as mog_db
    from game import payments

    if cb.from_user.id not in _pending_full:
        await cb.answer("📸 Фото уже вышло из памяти — просто кинь его заново.", show_alert=True)
        return
    # 🎁 Подарочный кредит — реролл бесплатно (без счёта).
    if await mog_db.spend_credit(cb.from_user.id, "reroll"):
        await cb.answer("🎁 Реролл бесплатный — считаю!")
        await _do_reroll(cb.message, cb.from_user.id)
        return
    try:
        await payments.send_invoice(bot, cb.message.chat.id, "reroll", cb.from_user.id)
        await cb.answer("Счёт на реролл отправлен ⭐")
    except Exception:  # noqa: BLE001
        log.exception("reroll invoice failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)


async def _do_reroll(msg: Message, user_id: int) -> None:
    """После оплаты: пересчитать то же фото новым роллом и выдать новую печать."""
    from game import rolls as mog_rolls

    data = _pending_full.get(user_id)
    if not data:
        await msg.answer(
            "🔄 Реролл оплачен, но фото уже вышло из памяти. Кинь его заново — пересчёт бесплатный.",
            parse_mode="HTML",
        )
        return
    raw = data[0]
    prev = round(mog_rolls.base_score(raw), 4)
    r = mog_rolls.roll(raw)  # новый шум/крит поверх той же базы
    score, tier = r["score"], r["tier"]
    arrow = "📈" if score > prev else ("📉" if score < prev else "➡️")
    try:
        card = await asyncio.to_thread(_mog_stamp, raw, score, tier)
        await msg.answer_photo(
            BufferedInputFile(card, filename="mog_card.jpg"),
            caption=(
                f"🎲 <b>МОГ {score:.4f} · {tier}</b>\n"
                f"🔄 Реролл: было {prev:.4f} → стало <b>{score:.4f}</b> {arrow}"
            ),
            parse_mode="HTML",
            reply_markup=await _card_keyboard_for(user_id),
        )
    except Exception:  # noqa: BLE001
        log.exception("reroll card failed")
        await msg.answer(f"🔄 Реролл: было {prev:.4f} → стало {score:.4f} ({tier})")


# ------------------------------------------------------- ванити-Студия (35 ⭐)
@dp.callback_query(F.data == "studio")
async def on_studio(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка под карточкой: платная ванити-ретушь ТОГО ЖЕ фото (байты из памяти)."""
    from game import db as mog_db
    from game import payments

    if cb.from_user.id not in _pending_full:
        await cb.answer("📸 Фото уже вышло из памяти — кинь его заново.", show_alert=True)
        return
    # 🎁 Подарочный кредит — Студия бесплатно (без счёта).
    if await mog_db.spend_credit(cb.from_user.id, "studio"):
        await cb.answer("🎁 Студия бесплатная — ретуширую!")
        await _do_studio(cb.message, cb.from_user.id)
        return
    try:
        await payments.send_invoice(bot, cb.message.chat.id, "studio", cb.from_user.id)
        await cb.answer("Счёт на Студию отправлен ⭐")
    except Exception:  # noqa: BLE001
        log.exception("studio invoice failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)


async def _do_studio(msg: Message, user_id: int) -> None:
    """После оплаты: отдать отретушированную карточку (балл не меняется)."""
    data = _pending_full.get(user_id)
    if not data:
        await msg.answer(
            "🎨 Оплачено, но фото уже вышло из памяти. Кинь заново — ретушь бесплатная.",
            parse_mode="HTML",
        )
        return
    raw = data[0]
    score, tier = _mog_score_tier(raw)
    try:
        card = await asyncio.to_thread(_studio_photo, raw, score, tier)
        await msg.answer_photo(
            BufferedInputFile(card, filename="mog_studio.jpg"),
            caption=(
                f"🎨 <b>MOG STUDIO · {score:.4f} · {tier}</b>\n"
                "Ретушь + золотая рамка. Чистая ванити — балл не менялся."
            ),
            parse_mode="HTML",
            reply_markup=await _card_keyboard_for(user_id),
        )
    except Exception:  # noqa: BLE001
        log.exception("studio card failed")
        await msg.answer("🎨 Не смог обработать — попробуй ещё раз.")


# ------------------------------------------------------- подарки (🎁, 15/35 ⭐)
_GIFT_LABEL = {"reroll": "🔄 Реролл", "studio": "🎨 Студия"}


@dp.callback_query(F.data == "gift")
async def on_gift(cb: CallbackQuery) -> None:
    """Меню подарков: что подарить другу."""
    rows = [
        [InlineKeyboardButton(text="🔄 Реролл — 15 ⭐", callback_data="gift_buy:reroll")],
        [InlineKeyboardButton(text="🎨 Студия — 35 ⭐", callback_data="gift_buy:studio")],
    ]
    await cb.message.answer(
        "🎁 <b>Подарок другу</b>\n"
        "Выбери, что подарить — бот выдаст карточку-подарок, перешли её другу. "
        "Кто первый нажмёт «Забрать» — тому и достанется.",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        parse_mode="HTML",
    )
    await cb.answer()


@dp.callback_query(F.data.startswith("gift_buy:"))
async def on_gift_buy(cb: CallbackQuery, bot: Bot) -> None:
    from game import payments

    kind = (cb.data or "").split(":", 1)[-1]
    if kind not in _GIFT_LABEL:
        await cb.answer("Неизвестный подарок.", show_alert=True)
        return
    try:
        await payments.send_invoice(bot, cb.message.chat.id, f"gift_{kind}", cb.from_user.id)
        await cb.answer("Счёт на подарок отправлен ⭐")
    except Exception:  # noqa: BLE001
        log.exception("gift invoice failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)


@dp.callback_query(F.data.startswith("gift_claim:"))
async def on_gift_claim(cb: CallbackQuery) -> None:
    from game import db as mog_db

    token = (cb.data or "").split(":", 1)[-1]
    ok, kind, reason = await mog_db.claim_gift_token(token, cb.from_user.id)
    if not ok:
        note = {
            "used": "😔 Этот подарок уже забрали.",
            "not_found": "Подарок не найден.",
        }.get(reason, "Не получилось забрать подарок.")
        await cb.answer(note, show_alert=True)
        return
    label = _GIFT_LABEL.get(kind, kind)
    await cb.message.answer(
        f"🎉 <b>Тебе подарок: {label}!</b>\n"
        f"Кинь фото — и жми «{label} — бесплатно 🎁» под карточкой 👇",
        parse_mode="HTML",
    )
    await cb.answer("Подарок твой 🎁")


async def _send_gift(msg: Message, product: dict) -> None:
    """Выдаёт пересылаемую карточку-подарок с кнопкой «Забрать»."""
    kind = product.get("gift_kind") or product.get("gift") or ""
    token = product.get("gift_token") or ""
    label = _GIFT_LABEL.get(kind, kind)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🎁 Забрать подарок", callback_data=f"gift_claim:{token}")]
        ]
    )
    await msg.answer(
        f"🎁 <b>Подарок готов: {label}</b>\n\n"
        "Перешли это сообщение другу — кто первый нажмёт «Забрать», "
        "тому подарок и достанется.",
        reply_markup=kb,
        parse_mode="HTML",
    )


@dp.pre_checkout_query()
async def on_pre_checkout(q: PreCheckoutQuery, bot: Bot) -> None:
    from game import payments

    ok, err = payments.validate_payload(q.invoice_payload)
    await bot.answer_pre_checkout_query(q.id, ok=ok, error_message=err)


@dp.message(F.successful_payment)
async def on_paid(msg: Message, bot: Bot) -> None:
    from game import payments

    sp = msg.successful_payment
    product = await payments.grant(sp.invoice_payload)
    product_key, paid_uid, _extras = payments.parse_payload(sp.invoice_payload)
    if product_key == "reroll" and paid_uid:
        await _do_reroll(msg, paid_uid)
        return
    if product_key == "studio" and paid_uid:
        await _do_studio(msg, paid_uid)
        return
    if product and product.get("gift_token"):
        await _send_gift(msg, product)
        return
    if product:
        await msg.answer(
            f"✅ Оплачено: <b>{product['title']}</b> ({sp.total_amount} ⭐)\n"
            f"{payments.grant_text(product)}",
            parse_mode="HTML",
        )
    else:
        await msg.answer(
            f"✅ Оплата получена ({sp.total_amount} ⭐), но товар не распознан.\n"
            f"<code>{sp.invoice_payload}</code>",
            parse_mode="HTML",
        )


# Добивки после оценки: ровно два предложения, потом дисклеймер.
# Только самоирония игры: без оскорблений по защищённым признакам и без темы самоповреждения.
_MOG_PUNCHLINES = (
    "Счёт зафиксирован нотариусом выдуманных наук. Апелляции принимаются только звёздами.",
    "Эксперты разошлись во мнениях, но сошлись в баре. Твой балл уже обсуждают без тебя.",
    "Замеры проведены линейкой, купленной на сдачу. Погрешность спишем на харизму.",
    "Результат заверен печатью «примерно». Пересдача — завтра, с другого ракурса.",
    "Алгоритм трижды перепроверил и один раз присвистнул. Дальше — только легенда.",
    "Протокол подписан, свидетели подкуплены комплиментами. Балл обжалованию не подлежит.",
    "Коллегия псевдоучёных голосовала дважды: второй раз — чтобы поржать. Вердикт устоял.",
    "Лаборатория подтверждает: лицо настоящее, цифры выдуманные. Наука бессильна.",
)


def _mog_score_tier(photo_bytes: bytes) -> tuple[float, str]:
    """Детерминированные МОГ-балл и тир из фото."""
    from game import rolls as mog_rolls

    score = round(mog_rolls.base_score(photo_bytes), 4)
    return score, mog_rolls.tier_for(score)


def _mog_tag(photo_bytes: bytes, score: float | None = None, tier: str | None = None) -> str:
    """МОГ-балл + смешная добивка + дисклеймер для любой выдачи (спек §7)."""
    import hashlib

    from game import config as mog_config

    if score is None or tier is None:
        score, tier = _mog_score_tier(photo_bytes)
    punch = _MOG_PUNCHLINES[
        int(hashlib.sha256(photo_bytes).hexdigest(), 16) % len(_MOG_PUNCHLINES)
    ]
    return (
        f"\n\n🎲 <b>МОГ: {score:.4f} · {tier}</b>\n{punch}\n"
        f"<i>{mog_config.DISCLAIMER}</i>"
    )


# ───────────────────────── СКАН «СЛАБЫЕ МЕСТА» ─────────────────────────
# Тонкий контур (1–3 px) по зонам лица: красным — слабые, синим — сильные.
# Фото не портим: только линия, без заливки и свечения.
# Позиции зон берём из FaceMesh; если биометрии нет — падаем на доли кадра.

SPOT_REGIONS: list[tuple[str, str]] = [
    ("groom", "лоб/груминг"),
    ("eyes", "глаза"),
    ("harmony", "центр/гармония"),
    ("cheeks", "скулы"),
    ("jaw", "челюсть"),
    ("aura", "аура (общий контур)"),
]

# ключевые точки FaceMesh (индексы), из них собираем боксы зон
_SPOT_IDX = {
    "eye_r_out": 33, "eye_r_in": 133, "eye_l_in": 362, "eye_l_out": 263,
    "zygo_r": 234, "zygo_l": 454,
    "gonion_r": 58, "gonion_l": 288,
    "chin": 152, "forehead_top": 10, "brow": 151,
    "temple_r": 127, "temple_l": 356,
    "nasion": 168, "nose_tip": 1, "subnasale": 2,
    "mouth_r": 78, "mouth_l": 308, "upper_lip": 13,
}


# Тонкий контур зоны: 1–3 px, без glow и «корон» — фото не портим.
# Толщина настраивается из .env: MOG_SPOT_LINE=1..3 (по умолчанию 2).
SPOT_LINE_WIDTH = max(1, min(3, int(os.getenv("MOG_SPOT_LINE", "2"))))


def _neon_color(score: int) -> tuple[int, int, int]:
    if score < 40:
        return (255, 45, 94)    # красный — слабая зона
    if score >= 60:
        return (56, 132, 255)   # синий — сильная зона
    return (34, 211, 238)       # циан — нейтрально (в скане не обводим)


def _draw_neon(d, box, color) -> None:
    """Тонкий контур зоны (1–3 px) — только линия, без заливки и свечения."""
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return
    d.rectangle([x1, y1, x2, y2], outline=color + (235,), width=SPOT_LINE_WIDTH)


def _box_from_points(pts, idxs, w, h, pad_x=0.0, pad_y=0.0):
    xs, ys = [], []
    for name in idxs:
        i = _SPOT_IDX[name]
        if i in pts:
            xs.append(pts[i][0])
            ys.append(pts[i][1])
    if not xs:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    px, py = (x2 - x1) * pad_x, (y2 - y1) * pad_y
    return [max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)]


def _region_boxes(pts, w: int, h: int) -> dict:
    return {
        "eyes": _box_from_points(pts, ["eye_r_out", "eye_r_in", "eye_l_in", "eye_l_out"], w, h, 0.10, 1.1),
        "cheeks": _box_from_points(pts, ["zygo_r", "zygo_l", "upper_lip"], w, h, 0.05, 0.08),
        "jaw": _box_from_points(pts, ["gonion_r", "gonion_l", "chin", "upper_lip"], w, h, 0.04, 0.03),
        "groom": _box_from_points(pts, ["temple_r", "temple_l", "forehead_top", "brow"], w, h, 0.03, 0.08),
        "harmony": _box_from_points(pts, ["nasion", "nose_tip", "subnasale"], w, h, 0.35, 0.14),
        "aura": _box_from_points(pts, ["forehead_top", "chin", "zygo_r", "zygo_l"], w, h, 0.06, 0.05),
    }


def _fallback_boxes(w: int, h: int) -> dict:
    def bx(fx1, fy1, fx2, fy2):
        return [fx1 * w, fy1 * h, fx2 * w, fy2 * h]

    return {
        "groom": bx(0.22, 0.05, 0.78, 0.24),
        "eyes": bx(0.22, 0.34, 0.78, 0.50),
        "harmony": bx(0.38, 0.38, 0.62, 0.62),
        "cheeks": bx(0.18, 0.46, 0.82, 0.64),
        "jaw": bx(0.24, 0.62, 0.76, 0.86),
        "aura": bx(0.15, 0.07, 0.85, 0.89),
    }


def _landmarks_px(image_bytes: bytes):
    """Возвращает (pts{idx:(x,y)}, rgb_ndarray, w, h) или (None, rgb, w, h)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None, None
    hh, ww = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not BIOMETRY_AVAILABLE:
        return None, rgb, ww, hh
    res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None, rgb, ww, hh
    pts = {i: (p.x * ww, p.y * hh) for i, p in enumerate(res.face_landmarks[0])}
    return pts, rgb, ww, hh


def annotate_weakspots(image_bytes: bytes, stats: dict) -> tuple[bytes, list]:
    """Неон-киберпанк аннотация зон лица. Возвращает (jpeg_bytes, items)."""
    from PIL import Image, ImageDraw, ImageFilter

    pts, rgb, w, h = _landmarks_px(image_bytes)
    img = Image.fromarray(rgb) if rgb is not None else Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    boxes = _region_boxes(pts, w, h) if pts else _fallback_boxes(w, h)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    items = []
    for key, label in SPOT_REGIONS:
        box = boxes.get(key)
        if not box:
            continue
        score = int(stats.get(key, 50))
        if 40 <= score < 60:
            continue  # нейтральные зоны не обводим — только слабые/сильные
        color = _neon_color(score)
        _draw_neon(d, box, color)
        items.append((key, label, score, color))

    base = img.convert("RGBA")
    out = Image.alpha_composite(base, overlay)
    if out.width > 1080:
        ratio = 1080 / out.width
        out = out.resize((1080, int(out.height * ratio)))
    buf = io.BytesIO()
    out.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), items


def spots_caption(items: list) -> str:
    from game import config as mog_config

    weak = [f"{lab} {sc}" for _k, lab, sc, _c in items if sc < 40]
    good = [f"{lab} {sc}" for _k, lab, sc, _c in items if sc >= 60]
    lines = ["🎯 <b>СЛАБЫЕ МЕСТА</b> · неон-скан", ""]
    lines.append("🔴 зоны урона: " + (", ".join(weak) if weak else "нет"))
    lines.append("🔵 halo: " + (", ".join(good) if good else "нет"))
    lines.append("")
    lines.append(f"<i>{mog_config.DISCLAIMER}</i>")
    return "\n".join(lines)


# ───────────────────────── ПЕЧАТЬ «МОГ» НА ФОТО ─────────────────────────
# Возвращает то же фото с неон-штампом «МОГ <балл> · тир» в углу — для репоста.


def _load_font(size: int):
    """Кириллический шрифт: сперва наш в репо, потом системные, потом дефолт."""
    from PIL import ImageFont

    here = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(here, "assets", "fonts", "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _studio_photo(photo_bytes: bytes, score: float, tier: str) -> bytes:
    """Ванити-ретушь карточки: мягкое сглаживание + тёплый тон + золотая рамка
    + бейдж «MOG STUDIO». Балл/тир НЕ меняются — это чистая косметика."""
    import io

    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    base = Image.open(io.BytesIO(_mog_stamp(photo_bytes, score, tier))).convert("RGB")
    # мягкое «ретушное» сглаживание (blend с блюром ≈ лёгкое билатеральное)
    smooth = base.filter(ImageFilter.GaussianBlur(1.2))
    img = Image.blend(base, smooth, 0.35)
    img = ImageEnhance.Color(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    img = ImageEnhance.Brightness(img).enhance(1.03)

    w, h = img.size
    draw = ImageDraw.Draw(img)
    gold, gold_light = (212, 175, 55), (255, 230, 140)
    frame = max(8, int(min(w, h) * 0.02))
    draw.rectangle([0, 0, w - 1, h - 1], outline=gold, width=frame)
    draw.rectangle(
        [frame, frame, w - 1 - frame, h - 1 - frame],
        outline=gold_light,
        width=max(2, frame // 3),
    )

    f = _load_font(max(16, int(w * 0.045)))
    label = "★ MOG STUDIO ★"
    try:
        tb = draw.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:  # noqa: BLE001
        tw, th = len(label) * 12, 22
    pad = max(8, int(w * 0.012))
    x0 = frame + pad
    y0 = h - frame - th - pad * 3
    draw.rectangle([x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2], fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), label, font=f, fill=gold_light)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _arena_board_image(players: list[dict]) -> bytes | None:
    """Неоновое табло арены: ранкинг по score (JPEG bytes | None)."""
    if not players:
        return None
    import io

    from PIL import Image, ImageDraw

    ranked = sorted(players, key=lambda p: p["score"], reverse=True)
    W = 1000
    top_pad, row_h, bot_pad = 132, 66, 36
    H = top_pad + row_h * len(ranked) + bot_pad
    img = Image.new("RGB", (W, H), (13, 13, 24))
    draw = ImageDraw.Draw(img)

    for x in range(0, W, 40):
        draw.line([(x, 0), (x, H)], fill=(22, 22, 38), width=1)
    for y in range(0, H, 40):
        draw.line([(0, y), (W, y)], fill=(22, 22, 38), width=1)

    neon = [(80, 220, 255), (255, 90, 200), (120, 255, 160)]
    gold = (255, 214, 90)

    f_title = _load_font(46)
    f_rank = _load_font(34)
    f_name = _load_font(30)
    f_score = _load_font(34)
    f_tier = _load_font(20)

    draw.text((40, 30), "MOG ARENA", font=f_title, fill=(255, 255, 255))
    draw.text((44, 90), "itogovoe tablo / final ranking", font=f_tier, fill=(150, 150, 175))

    maxs = max((p["score"] for p in ranked), default=1.0) or 1.0
    for i, p in enumerate(ranked):
        y = top_pad + i * row_h
        col = gold if i == 0 else neon[i % len(neon)]
        draw.rectangle([40, y, 86, y + 46], fill=col)
        rk = "★" if i == 0 else str(i + 1)
        try:
            rb = draw.textbbox((0, 0), rk, font=f_rank)
            draw.text(
                (40 + 23 - (rb[2] - rb[0]) / 2, y + 23 - (rb[3] - rb[1]) / 2 - rb[1]),
                rk,
                font=f_rank,
                fill=(10, 10, 18),
            )
        except Exception:  # noqa: BLE001
            draw.text((48, y + 6), rk, font=f_rank, fill=(10, 10, 18))

        name = (p.get("name") or "?")[:22]
        draw.text((110, y + 2), name, font=f_name, fill=(240, 240, 255))
        draw.text((110, y + 36), str(p.get("tier") or ""), font=f_tier, fill=(150, 150, 175))

        sc = f"{p['score']:.4f}"
        try:
            sw = draw.textbbox((0, 0), sc, font=f_score)[2]
        except Exception:  # noqa: BLE001
            sw = len(sc) * 20
        draw.text((W - 40 - sw, y + 6), sc, font=f_score, fill=col)

        bar_x0, bar_x1, bar_y = 110, W - 60, y + 58
        frac = max(0.0, min(1.0, p["score"] / maxs))
        draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + 4], fill=(35, 35, 55))
        draw.rectangle(
            [bar_x0, bar_y, bar_x0 + int((bar_x1 - bar_x0) * frac), bar_y + 4], fill=col
        )

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


async def _send_mog_card(msg: Message, raw: bytes, score: float, tier: str) -> bool:
    """Шлёт фото с печатью «МОГ».

    Печать рисуется Pillow'ом; если он недоступен (или сломан) — шлём ИСХОДНОЕ фото
    с тем же подписью-баллом. Принцип: юзер ВСЕГДА получает картинку, даже если
    красивость не удалась. Вернёт True, если ушла именно печатная карточка.
    """
    caption = (
        f"🎲 <b>МОГ {score:.4f} · {tier}</b>\n"
        "😏 Печать стоит — кинь другу"
    )
    card: bytes | None = None
    try:
        card = await asyncio.to_thread(_mog_stamp, raw, score, tier)
    except Exception:  # noqa: BLE001 — без PIL печать недоступна, но фото отправим
        log.exception("stamp failed — отправляю фото без печати")
    try:
        await msg.answer_photo(
            BufferedInputFile(card or raw, filename="mog_card.jpg"),
            caption=caption,
            parse_mode="HTML",
        )
        return card is not None
    except Exception:  # noqa: BLE001 — крайний случай: картинка не ушла вовсе
        log.exception("answer_photo failed")
        await msg.answer(caption, parse_mode="HTML")
        return False


async def _resolve_duel_link(msg: Message, bot: Bot, score: float) -> None:
    """Если юзер принял вызов по ссылке — считает бой и разсылает табло обоим.

    Молча выходит, если активного боя нет: обычное фото в личке не должно
    падать из-за отсутствия дуэли.
    """
    if msg.chat.type != "private":
        return
    from game import duel_link as mog_duel

    try:
        result = await mog_duel.resolve_for(msg.from_user.id, score)
    except Exception:  # noqa: BLE001 — дуэль не должна ронять разбор фото
        log.exception("duel link resolve failed")
        return
    if not result:
        return
    board = mog_duel.result_text(result)
    await msg.answer(board, parse_mode="HTML")
    # Вызывающий в другом чате/личке — пишем ему результат тоже.
    try:
        await bot.send_message(
            int(result["challenger_id"]),
            "⚔️ <b>Бой окончен!</b>\n\n" + board,
            parse_mode="HTML",
        )
    except Exception:  # noqa: BLE001 — заблокировал бот
        log.warning("не смог прислать табло вызывающему", exc_info=True)


def _mog_stamp(photo_bytes: bytes, score: float, tier: str) -> bytes:
    """Фото + неон-печать «МОГ <балл> · тир» в правом нижнем углу (JPEG bytes)."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    W, H = img.size
    color = _neon_color(int(round(score / 10.0 * 100)))
    lab, sc, ti = "МОГ", f"{score:.4f}", tier.upper()

    # Размер печати ≈ 34% короткой стороны; шрифт балла подгоняем под ширину.
    target_w = max(150, int(min(W, H) * 0.34))
    size = max(18, int(target_w * 0.20))
    f_sc = _load_font(size)
    for _ in range(12):
        bb = f_sc.getbbox(sc)
        w = bb[2] - bb[0]
        if w <= target_w * 0.74 or size <= 18:
            break
        size = max(18, int(size * (target_w * 0.74) / max(1, w)))
        f_sc = _load_font(size)
    f_lab = _load_font(max(14, int(size * 0.50)))
    f_ti = _load_font(max(12, int(size * 0.42)))

    md = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    b_lab = md.textbbox((0, 0), lab, font=f_lab)
    b_sc = md.textbbox((0, 0), sc, font=f_sc)
    b_ti = md.textbbox((0, 0), ti, font=f_ti)
    h_lab, h_sc, h_ti = b_lab[3] - b_lab[1], b_sc[3] - b_sc[1], b_ti[3] - b_ti[1]
    inner_w = max(b_lab[2] - b_lab[0], b_sc[2] - b_sc[0], b_ti[2] - b_ti[0])

    gap = max(4, int(size * 0.22))
    padx, pady = int(size * 0.60), int(size * 0.46)
    inner_h = h_lab + gap + h_sc + gap + h_ti
    SW, SH = inner_w + padx * 2, inner_h + pady * 2

    stamp = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    d = ImageDraw.Draw(stamp)
    radius = int(min(SW, SH) * 0.20)
    d.rounded_rectangle([0, 0, SW - 1, SH - 1], radius=radius, fill=(8, 8, 16, 135))
    for width, alpha in ((max(4, int(SH * 0.06)), 45), (max(2, int(SH * 0.03)), 100), (2, 210)):
        d.rounded_rectangle([0, 0, SW - 1, SH - 1], radius=radius, outline=color + (alpha,), width=width)
    d.rounded_rectangle([6, 6, SW - 7, SH - 7], radius=max(2, radius - 5), outline=(255, 255, 255, 70), width=1)

    y = pady
    for i, (txt, font, bbox, fill) in enumerate(
        (
            (lab, f_lab, b_lab, (255, 255, 255, 255)),
            (sc, f_sc, b_sc, color + (255,)),
            (ti, f_ti, b_ti, (255, 255, 255, 225)),
        )
    ):
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((SW - tw) // 2 - bbox[0], y - bbox[1]), txt, font=font, fill=fill)
        y += th + (gap if i < 2 else 0)

    stamp = stamp.rotate(6, expand=True, resample=Image.BICUBIC)
    glow = stamp.filter(ImageFilter.GaussianBlur(4))

    margin = int(min(W, H) * 0.035)
    px = max(0, W - stamp.width - margin)
    py = max(0, H - stamp.height - margin)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay.alpha_composite(glow, (px, py))
    overlay.alpha_composite(stamp, (px, py))

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    if out.width > 1080:
        ratio = 1080 / out.width
        out = out.resize((1080, int(out.height * ratio)))
    buf = io.BytesIO()
    out.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


@dp.message(Command("legal", "privacy", "consent"))
async def cmd_legal(msg: Message) -> None:
    """Правила и согласие на обработку данных (дублирует веб-страницу)."""
    from game import legal as mog_legal

    link = (
        f'\n\n📄 Полный текст: <a href="{PUBLIC_BASE_URL}/legal">{PUBLIC_BASE_URL}/legal</a>'
        if PUBLIC_BASE_URL
        else ""
    )
    await msg.answer(
        "📄 <b>Правила и согласие</b>\n\n" + mog_legal.consent_short() + link,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def _has_consent(user_id: int) -> bool:
    """Согласие уже зафиксировано и версия политики актуальна?"""
    from game import db as mog_db
    from game import legal as mog_legal

    row = await mog_db.get_consent(user_id)
    return bool(row and row["version"] == mog_legal.POLICY_VERSION)


def _consent_keyboard() -> InlineKeyboardMarkup:
    """Одна мелкая строка: «ок» + «договор» (текст политики — только по клику)."""
    from game import legal as mog_legal

    row = [
        InlineKeyboardButton(
            text="✅ ок, 18+", callback_data=f"consent:{mog_legal.POLICY_VERSION}"
        )
    ]
    if PUBLIC_BASE_URL:
        row.append(InlineKeyboardButton(text="📄 договор", url=f"{PUBLIC_BASE_URL}/legal"))
    return InlineKeyboardMarkup(inline_keyboard=[row])


async def _ensure_consent(msg: Message, bot: Bot, intro: str = "") -> bool:
    """Согласие (152-ФЗ). Нет — один мелкий тап, без простыни текста.

    Никаких абзацев: человек видит только «18+» и две кнопки. Полный текст
    политики живёт за кнопкой «договор» (страница /legal), в чат не вываливается.
    """
    from game import db as mog_db
    from game import legal as mog_legal

    row = await mog_db.get_consent(msg.from_user.id)
    if row and row["version"] == mog_legal.POLICY_VERSION:
        return True
    await msg.answer(
        "18+",
        reply_markup=_consent_keyboard(),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    return False


@dp.callback_query(F.data.startswith("consent:"))
async def on_consent(cb: CallbackQuery, bot: Bot) -> None:
    """Один тап по «ок» → фиксируем и сразу продолжаем. Без служебных сообщений."""
    from game import db as mog_db

    version = (cb.data or "").split(":", 1)[-1]
    await mog_db.record_consent(cb.from_user.id, version)
    await cb.answer("✅ Погнали")
    # Убираем строку согласия из клавиатуры, остальные кнопки не трогаем.
    try:
        kb = getattr(cb.message, "reply_markup", None)
        rows = [
            r
            for r in (kb.inline_keyboard if kb else [])
            if not any((btn.callback_data or "").startswith("consent:") for btn in r)
        ]
        await cb.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows) if rows else None
        )
    except Exception:  # noqa: BLE001 — сообщение старое/без клавиатуры
        pass
    # 🏟 Если человек начал турнир и упёрся в согласие — сразу открываем сбор,
    # чтобы не заставлять его второй раз жать /tournament.
    st = _tournaments.get(cb.from_user.id)
    if st and st.get("stage") == "consent":
        from game import tournament as mog_tt

        st.update(stage="collect", ts=time.time(), album=[])
        await _tt_panel(
            bot, st, mog_tt.collect_text(0), mog_tt.collect_keyboard(0)
        )
    # 📸 Фото, присланное до согласия, — считаем сразу, от имени ЧЕЛОВЕКА
    # (cb.message.from_user — это бот, поэтому передаём юзера явно).
    entry = _pending_consent.pop(cb.from_user.id, None)
    if entry:
        stored_at, raw = entry
        if time.time() - stored_at <= _CONSENT_PHOTO_TTL and raw:
            await _process_photo(cb.message, bot, raw, user=cb.from_user)


@dp.message(F.photo)
async def on_photo(msg: Message, bot: Bot) -> None:
    # 🏟 Режим тестового турнира: не разбираем, а копим фото в сетку.
    st = _tt_get(msg.from_user.id)
    if st and st.get("stage") == "collect" and msg.chat.type == "private":
        await _tt_add_photo(msg, bot, msg.photo[-1].file_id)
        return

    if not TELEGRAM_TOKEN or not MODEL_API_KEY:
        await msg.answer("⚠️ Бот не настроен: нет TELEGRAM_TOKEN или MODEL_API_KEY в .env")
        return

    # 📸 Скачиваем фото ДО проверки согласия — чтобы после кнопки «Согласен»
    # разбор стартовал сразу, без повторной отправки фото.
    try:
        photo = msg.photo[-1]  # самое большое разрешение
        tg_file = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(tg_file.file_path)
        raw = photo_bytes.read()
    except Exception:  # noqa: BLE001
        log.exception("photo download failed")
        await msg.answer("Не смог скачать фото. Отправь ещё раз 📸")
        return

    # 🔞 Без согласия на обработку ПДн фото не обрабатываем — но запоминаем
    # байты: нажатие «Согласен» запустит обработку этого фото автоматически.
    if not await _ensure_consent(msg, bot):
        _stash_consent_photo(msg.from_user.id, raw)
        return

    await _process_photo(msg, bot, raw)


class _MsgAs:
    """Чат/сообщение как есть, но от имени нужного юзера.

    Нужно для отложенного фото: жмёшь «ок» под сообщением бота, а разбор должен
    идти от человека (иначе ролл запишется на бота).
    """

    __slots__ = ("_msg", "from_user")

    def __init__(self, msg, user):
        self._msg = msg
        self.from_user = user

    def __getattr__(self, name):
        return getattr(self._msg, name)


@dp.message(F.document)
async def on_document(msg: Message, bot: Bot) -> None:
    """Файл картинкой (jpg/png «как файл») — только в режиме турнира."""
    st = _tt_get(msg.from_user.id)
    if not st or st.get("stage") != "collect" or msg.chat.type != "private":
        await on_other(msg)
        return
    doc = msg.document
    mime = (doc.mime_type or "").lower()
    if not mime.startswith("image/"):
        await msg.answer("Это не картинка. Кидай jpg/png — фото или файлом.")
        return
    if (doc.file_size or 0) > 10 * 1024 * 1024:
        await msg.answer("Файл больше 10 МБ — сожми и кидай ещё.")
        return
    await _tt_add_photo(msg, bot, doc.file_id)


async def _process_photo(msg: Message, bot: Bot, raw: bytes, user=None) -> None:
    """Полный разбор: дуэль → биометрия → нейронка → печать «МОГ» + кнопки."""
    if user is not None:
        msg = _MsgAs(msg, user)
    status = await msg.answer(
        "📐 Изучаю пропорции…" if BIOMETRY_AVAILABLE else "🧠 Нейронка думает…"
    )
    tmp_path = ""
    # Нейтральные значения на случай сбоя ДО их вычисления: локальная печать «МОГ»
    # и кнопки не должны зависеть от нейронки (она может быть на 429).
    metrics = {"face_found": False, "error": "biometry_unavailable"}
    metrics_text = format_metrics(metrics)
    kb = await _card_keyboard_for(msg.from_user.id)
    try:
        # ⚔️ ДУЭЛЬ: если этого юзера вызвали на бой — его фото закрывает батл.
        # Работает прямо в чате, без мини-аппа и без HTTPS.
        try:
            from game import battles as mog_battles
            from game import db as mog_db
            from game import rolls as mog_rolls

            pending = await mog_db.find_pending_battle(msg.chat.id, msg.from_user.id)
            if pending:
                duel_score = mog_rolls.roll(raw)["score"]
                res = await mog_battles.resolve(pending["id"], duel_score)
                await msg.reply(
                    mog_battles.result_text(res),
                    parse_mode="HTML",
                    reply_markup=mog_battles.rematch_keyboard(res),
                )
        except Exception:  # noqa: BLE001 — батл не должен ронять обычный разбор
            log.exception("battle resolve failed")

        # 1. Локальная биометрия (в треде, чтобы не блочить loop).
        # В лайт-режиме хостинга её нет — сразу идём к нейронке по фото.
        if BIOMETRY_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            metrics = await asyncio.to_thread(analyze_face, tmp_path)
            # Числа — ПОД КАПОТ: в чат не показываем, только по кнопке «🧬 Замеры».
            # Нет лица — не блокируем: разбор всё равно даём (99,9% фото).
            metrics_text = format_metrics(metrics)
            log.info("metrics: %s", metrics_text)
            await status.edit_text("🧠 Нейронка думает…")
        else:
            metrics = {"face_found": False, "error": "biometry_unavailable"}
            metrics_text = format_metrics(metrics)

        # 2. Первый слой: разбор простыми словами (тоже в треде)
        simple = await asyncio.to_thread(analyze_simple, raw, metrics_text)

        # Запоминаем контекст для кнопки "подробный разбор"
        _remember_photo(msg.from_user.id, raw, metrics_text, metrics)

        await status.delete()
        score, tier = _mog_score_tier(raw)
        mog_line = _mog_tag(raw, score, tier)

        # 💾 Сохраняем ролл в БД. Без этого арена и дуэли «не видят» фото:
        # get_best_roll() возвращает пустоту, и вход в игру ругается «сначала фото».
        await _persist_roll(msg, raw, score, tier)

        # Печать «МОГ» на фото — готово к репосту. Не роняем разбор, если не нарисуется.
        await _send_mog_card(msg, raw, score, tier)

        # ⚔️ Арена: если ждали фото (нажал «Я в деле» без ролла) или уже в лобби —
        # вписываем/обновляем бойца и перерисовываем лобби прямо в чате.
        await _apply_arena_photo(msg, bot, score, tier)

        # ⚔️ Дуэль по ссылке: если этот юзер ПРИНЯЛ вызов и прислал фото — считаем бой
        # и пишем табло обоим в личку (в группе этот вызов не жил).
        await _resolve_duel_link(msg, bot, score)

        if simple:
            # Футер-вирус: подталкиваем переслать другу (ретеншн).
            await msg.answer(
                simple + mog_line + "\n\n😏 Перешли другу — пусть тоже узнает правду",
                reply_markup=kb,
                parse_mode="HTML",
            )
        else:
            # Простой слой не выдал текст — кнопка жести всё равно живая
            # (у полного разбора лимит больше, может прорваться).
            await msg.answer(
                "Не разглядел по-простому — жми, разъебу по-полной 👇" + mog_line,
                reply_markup=kb,
                parse_mode="HTML",
            )
    except Exception as exc:  # noqa: BLE001 — юзер должен видеть ошибку текстом
        log.exception("photo handling failed")
        # Нейронка могла упасть на 429 — но печать «МОГ» и кнопки ЛОКАЛЬНЫ и
        # работают всегда. Отдаём их, чтобы юзер не остался без вирусной карточки.
        sent_card = False
        if raw:
            try:
                score, tier = _mog_score_tier(raw)
                _remember_photo(msg.from_user.id, raw, metrics_text, metrics)
                await _persist_roll(msg, raw, score, tier)
                sent_card = await _send_mog_card(msg, raw, score, tier)
                await msg.answer(
                    _friendly_error(exc) + "\nПечать твоя 👆 Кнопки ниже живут без нейронки.",
                    reply_markup=kb,
                    parse_mode="HTML",
                )
            except Exception:  # noqa: BLE001
                log.exception("fallback card failed")
                sent_card = False
        if not sent_card:
            await msg.answer(_friendly_error(exc) + "\nПопробуй другое фото.")
        try:
            await status.delete()
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@dp.message()
async def on_other(msg: Message) -> None:
    # В группах молчим: иначе после выключения privacy-режима бот огрызается
    # на каждое сообщение в чате и его выгоняют.
    if msg.chat.type != "private":
        return
    await msg.answer("Мне нужно именно <b>фото</b>, не текст. Кидай ебало 👇", parse_mode="HTML")


@dp.callback_query(F.data == "full_roast")
async def on_full_roast(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка 'Жёсткий разбор': полный PSL-разбор по сохранённому фото."""
    data = _pending_full.get(cb.from_user.id)
    if not data:
        await cb.answer("Фото устарело — кинь заново 👇", show_alert=True)
        return
    await cb.answer("Готовлю жесть…")
    raw, metrics_text, _m = data
    status = await cb.message.answer("🧠 Нейронка думает…")
    try:
        # Полный разбор (в треде — сетевой вызов синхронный)
        verdict = await asyncio.to_thread(analyze_with_vlm, raw, metrics_text)

        header = ""
        # Режем длинные ответы под лимит TG (4096)
        chunk_size = 4000 - len(header)
        first = True
        for i in range(0, len(verdict), chunk_size):
            chunk = verdict[i : i + chunk_size]
            await cb.message.answer((header if first else "") + chunk)
            first = False
        await status.delete()
    except Exception as exc:  # noqa: BLE001
        log.exception("full roast failed")
        await cb.message.answer(_friendly_error(exc) + "\nПопробуй ещё раз кнопкой.")
        try:
            await status.delete()
        except Exception:  # noqa: BLE001, S110
            pass


@dp.callback_query(F.data == "weakspots")
async def on_weakspots(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка 'Слабые места': неон-рамки по зонам лица (красное/зелёное)."""
    data = _pending_full.get(cb.from_user.id)
    if not data:
        await cb.answer("Фото устарело — кинь заново 👇", show_alert=True)
        return
    await cb.answer("Рисую неон-скан…")
    raw, _metrics_text, _m = data
    try:
        from game import rolls as mog_rolls

        stats = mog_rolls.stats_for(raw)
        img_bytes, items = await asyncio.to_thread(annotate_weakspots, raw, stats)
        await cb.message.answer_photo(
            BufferedInputFile(img_bytes, filename="mog_spots.jpg"),
            caption=spots_caption(items),
            parse_mode="HTML",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("weakspots failed")
        await cb.message.answer(f"💥 {PUBLIC_ERROR}")


@dp.callback_query(F.data == "metrics")
async def on_metrics(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка 'Замеры (тех)': сырые числа биометрии — только по запросу, под капот."""
    data = _pending_full.get(cb.from_user.id)
    if not data:
        await cb.answer("Фото устарело — кинь заново 👇", show_alert=True)
        return
    await cb.answer("Под капот…")
    _raw, metrics_text, metrics = data
    await cb.message.answer(
        f"🧬 <b>ТЕХ-ЗАМЕРЫ</b> <i>(под капот)</i>\n{format_metrics_ui(metrics)}",
        parse_mode="HTML",
    )


async def main() -> None:
    if not TELEGRAM_TOKEN:
        raise SystemExit("Нет TELEGRAM_TOKEN — создай .env по образцу .env.example")
    if not MODEL_API_KEY:
        raise SystemExit("Нет MODEL_API_KEY — возьми на https://z.ai (регистрация по email)")
    bot = Bot(token=TELEGRAM_TOKEN)

    # Инициализируем игровую БД (нужна и для /mog, и для /api/*).
    from game import api as mog_api
    from game import db as mog_db

    await mog_db.init_db()

    if PUBLIC_BASE_URL:
        # ☁️ Облачный режим (Render и любой хостинг с HTTPS-URL): webhook.
        # Телеграм сам присылает апдейты на URL — держать соединение не надо,
        # переживает сон free-тарифа. Плюс поднимаем Mini App и /api/*.
        # Порядок важен: СНАЧАЛА биндим порт (Render health-check), ПОТОМ Telegram —
        # чтобы сетевой чих Телеграма не ронял весь деплой.
        from aiohttp import web
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

        # Telegram разрешает в secret_token только [A-Za-z0-9_-]{1,256}.
        # Пустой/невалидный WEBHOOK_SECRET (в т.ч. старый дефолт с кириллицей) —
        # молча заменяем случайным валидным, иначе set_webhook падает с Bad Request.
        import re
        import secrets as _secrets

        _raw_secret = os.environ.get("WEBHOOK_SECRET", "").strip()
        if _raw_secret and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", _raw_secret):
            secret = _raw_secret
        else:
            secret = _secrets.token_urlsafe(24)[:32]
            log.warning("WEBHOOK_SECRET пуст/невалиден — сгенерирован случайный, webhook пересоздастся")
        hook_path = f"/webhook/{secret}"
        app = web.Application()

        async def health(_request: object) -> object:
            # Диагностика деплоя: pil=false -> печать «МОГ» и картинки не рисуются.
            try:
                import PIL  # noqa: F401, PLC0415

                pil = True
            except Exception:  # noqa: BLE001
                pil = False
            return web.json_response({"ok": True, "pil": pil, "ver": APP_VERSION})

        app.router.add_get("/", health)  # для Render health-check и cron-job.org анти-сна
        app.router.add_get("/healthz", health)

        # 📄 Юридическая страница: согласие на обработку ПДн / правила / 18+.
        async def legal_page(_request: object) -> object:
            from game import legal as mog_legal

            return web.Response(
                text=mog_legal.page_html(base_url=PUBLIC_BASE_URL),
                content_type="text/html",
            )

        for _path in ("/legal", "/privacy", "/terms", "/consent"):
            app.router.add_get(_path, legal_page)
        SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=secret).register(
            app, path=hook_path
        )
        # Mini App + игровой API.
        mog_api.register_webapp(app, bot)
        setup_application(app, dp, bot=bot)

        port = int(os.environ.get("PORT", "8080"))
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        log.info("webhook-режим: %s (порт %d), Mini App: %s/app", PUBLIC_BASE_URL, port, PUBLIC_BASE_URL)

        # Telegram-настройка уже после бинда порта.
        try:
            await bot.set_webhook(
                f"{PUBLIC_BASE_URL}{hook_path}", secret_token=secret, drop_pending_updates=True
            )
            log.info("webhook установлен: %s%s", PUBLIC_BASE_URL, hook_path)
        except Exception:  # noqa: BLE001
            # Страховка: лучше polling, чем бот, который молчит. Порт уже занят,
            # поэтому / и /app продолжают работать (health-check и мини-апп живы).
            log.exception("set_webhook не удался — переключаюсь на polling, TG не замолчит")
            try:
                await bot.delete_webhook(drop_pending_updates=False)
            except Exception:  # noqa: BLE001, S110
                pass
            await dp.start_polling(bot)
            return

        # Кнопка-меню "открыть MOG" во всех чатах.
        try:
            await bot.set_chat_menu_button(
                menu_button={"type": "web_app", "text": "🎰 MOG", "web_app": {"url": f"{PUBLIC_BASE_URL}/app"}}
            )
        except Exception:  # noqa: BLE001 — кнопка не критична
            log.warning("set_chat_menu_button не удалось", exc_info=True)

        # 🔁 Анти-сон: free-тариф Render усыпляет web-сервис после ~15 мин без
        # ВХОДЯЩИХ запросов (исходящий long-poll не считается). Раз в 10 минут
        # пингуем собственный публичный URL → входящий трафик → инстанс не спит.
        async def _keepalive() -> None:
            import aiohttp

            url = f"{PUBLIC_BASE_URL}/healthz"
            await asyncio.sleep(120)  # дать сервису подняться
            while True:
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.get(url, timeout=aiohttp.ClientTimeout(total=25)) as r:
                            log.info("keep-alive %s -> %s", url, r.status)
                except Exception:  # noqa: BLE001 — пинг не должен ронять бота
                    log.warning("keep-alive ping не удался", exc_info=True)
                await asyncio.sleep(600)

        asyncio.create_task(_keepalive())
        asyncio.create_task(_duel_sweep(bot))

        await asyncio.Event().wait()  # висим вечно
    else:
        # 💻 Локальный режим: polling, полная биометрия если установлена.
        # Mini App/API тут недоступны (нужен публичный HTTPS).
        await bot.delete_webhook(drop_pending_updates=True)
        log.info(
            "polling-режим, модель=%s @ %s, биометрия=%s",
            MODEL_NAME,
            MODEL_BASE_URL,
            "ON" if BIOMETRY_AVAILABLE else "OFF",
        )
        log.info("Mini App выключен (нет RENDER_EXTERNAL_URL/PUBLIC_URL) — /mog создаёт батлы, но без приложения")
        await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
