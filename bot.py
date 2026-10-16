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
    # 1) Google AI Studio (ключ GEMINI_API_KEY). Регион РФ блокируется — работает на
    #    Render (Frankfurt). Недоступный регион авто-выключаем (см. _VLM_DISABLED).
    (
        "gemini",
        _vlm_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "GEMINI_API_KEY",
        _vlm_env("GEMINI_MODELS", "gemini-3.8-flash,gemini-flash-latest"),
    ),
    # 2) Groq (ключ GROQ_API_KEY) — быстрый, бесплатный тир, vision Llama 4 Scout.
    (
        "groq",
        _vlm_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "GROQ_API_KEY",
        _vlm_env("GROQ_MODELS", "qwen/qwen3.8-27b"),
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

# ---------------------------------------------------------------- бот
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "💀 <b>PSL-бот на связи.</b>\n\n"
        "Кидай своё фото (лицо крупно, без очков и фильтров) — "
        "сначала скажу по-человечески, без занудства. "
        "А если хочешь жести — жми кнопку «📊 Подробный разбор»: "
        "там PSL-рейтинг, кости, failo/halo и план прокачки.\n\n"
        "Фото обрабатывается локально (MediaPipe) + нейронка. "
        "Ничего не храню — файл удаляется сразу после анализа.",
        parse_mode="HTML",
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
        "🎒 Артефакты в бою: за Stars, буст до +0.5 (есть 🎁 бесплатный раз в день).",
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
            ok, reason = True, "Ты уже в деле ⚔️"
        else:
            ok, reason = await mog_arena.join(arena["id"], host)
    except Exception:  # noqa: BLE001
        log.exception("arena quick failed")
        await cb.answer(PUBLIC_ERROR, show_alert=True)
        return
    if not ok:
        await cb.answer(reason, show_alert=True)
        return
    await cb.answer("Общий баттл! 🔥" if created else "Ты в арене ⚔️")
    plist = await mog_arena.players(arena["id"])
    await cb.message.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )


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
    except Exception:  # noqa: BLE001 — текст не изменился или сообщение устарело
        pass


async def _arena_finish(message, arena_id: int) -> None:
    """Резолвит арену и публикует финальное табло."""
    from game import arena as mog_arena
    from game import db as mog_db

    arena = await mog_db.get_arena(arena_id)
    if not arena:
        return
    plist = await mog_arena.resolve(arena_id)
    text = mog_arena.result_text(arena, plist)
    kb = mog_arena.rematch_keyboard(arena_id)
    try:
        await message.edit_text(text, reply_markup=kb, parse_mode="HTML")
    except Exception:  # noqa: BLE001
        await message.answer(text, reply_markup=kb, parse_mode="HTML")


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
    await msg.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("arena_join:"))
async def on_arena_join(cb: CallbackQuery) -> None:
    arena_id = int(cb.data.split(":", 1)[1])
    from game import arena as mog_arena
    from game import db as mog_db

    await mog_db.get_or_create_user(
        cb.from_user.id, cb.from_user.full_name, cb.from_user.username or ""
    )
    user = {"id": cb.from_user.id, "name": cb.from_user.full_name}
    ok, reason = await mog_arena.join(arena_id, user)
    if not ok:
        await cb.answer(reason, show_alert=True)
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
    await cb.message.answer(
        mog_arena.lobby_text(arena, plist),
        reply_markup=mog_arena.lobby_keyboard(arena, plist),
        parse_mode="HTML",
    )
    await cb.answer("Новая арена ⚔️")


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


@dp.pre_checkout_query()
async def on_pre_checkout(q: PreCheckoutQuery, bot: Bot) -> None:
    from game import payments

    ok, err = payments.validate_payload(q.invoice_payload)
    await bot.answer_pre_checkout_query(q.id, ok=ok, error_message=err)


@dp.message(F.successful_payment)
async def on_paid(msg: Message) -> None:
    from game import payments

    sp = msg.successful_payment
    product = await payments.grant(sp.invoice_payload)
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


@dp.message(F.photo)
async def on_photo(msg: Message, bot: Bot) -> None:
    if not TELEGRAM_TOKEN or not MODEL_API_KEY:
        await msg.answer("⚠️ Бот не настроен: нет TELEGRAM_TOKEN или MODEL_API_KEY в .env")
        return

    status = await msg.answer(
        "📐 Изучаю пропорции…" if BIOMETRY_AVAILABLE else "🧠 Нейронка думает…"
    )
    tmp_path = ""
    raw = b""
    # Нейтральные значения на случай сбоя ДО их вычисления: локальная печать «МОГ»
    # и кнопки не должны зависеть от нейронки (она может быть на 429).
    metrics = {"face_found": False, "error": "biometry_unavailable"}
    metrics_text = format_metrics(metrics)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⚔️ Дуэль 1×1", callback_data="duel_new"),
                InlineKeyboardButton(text="🏆 Общий баттл", callback_data="arena_quick"),
            ],
            [InlineKeyboardButton(text="📊 Подробный разбор", callback_data="full_roast")],
            [InlineKeyboardButton(text="🎯 Слабые места (неон)", callback_data="weakspots")],
            [InlineKeyboardButton(text="🧬 Замеры (тех)", callback_data="metrics")],
        ]
    )
    try:
        photo = msg.photo[-1]  # самое большое разрешение
        tg_file = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(tg_file.file_path)
        raw = photo_bytes.read()

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

        # Печать «МОГ» на фото — готово к репосту. Не роняем разбор, если не нарисуется.
        try:
            card = await asyncio.to_thread(_mog_stamp, raw, score, tier)
            await msg.answer_photo(
                BufferedInputFile(card, filename="mog_card.jpg"),
                caption=f"🎲 <b>МОГ {score:.4f} · {tier}</b>\n😏 Печать стоит — кинь другу",
                parse_mode="HTML",
            )
        except Exception:  # noqa: BLE001
            log.exception("stamp failed")

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
                card = await asyncio.to_thread(_mog_stamp, raw, score, tier)
                await msg.answer_photo(
                    BufferedInputFile(card, filename="mog_card.jpg"),
                    caption=f"🎲 <b>МОГ {score:.4f} · {tier}</b>\n😏 Печать стоит — кинь другу",
                    parse_mode="HTML",
                )
                await msg.answer(
                    _friendly_error(exc) + "\nПечать твоя 👆 Кнопки ниже живут без нейронки.",
                    reply_markup=kb,
                    parse_mode="HTML",
                )
                sent_card = True
            except Exception:  # noqa: BLE001
                log.exception("fallback card failed")
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
            return web.Response(text="ok")

        app.router.add_get("/", health)  # для Render health-check и cron-job.org анти-сна
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
