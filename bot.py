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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4.6v-flash")

# Free vision-модели по приоритету. На z.ai единственная бесплатная vision —
# glm-4.6v-flash (glm-4.6v/4.5v платные, прочие имена не существуют).
# Переопределяется env MODEL_FALLBACKS (через запятую) под другой провайдер.
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv("MODEL_FALLBACKS", "glm-4.6v-flash").split(",")
    if m.strip()
]
MODEL_CANDIDATES = [MODEL_NAME] + [m for m in MODEL_FALLBACKS if m != MODEL_NAME]

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
def _vision_chat(messages: list, *, max_tokens: int, temperature: float = 0.9):
    """Вызов VLM с авто-фолбэком по free-моделям и ретраями на 429/5xx.

    z.ai free-модели периодически отвечают 429 (code 1305, «перегружено») —
    short retry + переход к следующей модели, чтобы юзер получал ответ.
    """
    import time

    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )

    client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)
    last_exc: Exception | None = None
    busy = True  # если всё упало только на 429 — это перегрузка, а не наш баг
    for model in MODEL_CANDIDATES:
        for attempt in range(3):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                busy = True
                log.warning("model %s try %d: %s", model, attempt + 1, str(exc)[:120])
                time.sleep(min(6.0, 1.5 * (attempt + 1)))
            except APIStatusError as exc:
                last_exc = exc
                if getattr(exc, "status_code", 0) >= 500:
                    busy = True
                    log.warning("model %s 5xx try %d: %s", model, attempt + 1, str(exc)[:120])
                    time.sleep(1.5 * (attempt + 1))
                else:
                    busy = False
                    log.warning("model %s %s: %s", model, exc.status_code, str(exc)[:120])
                    break  # 4xx кроме 429 — сразу к следующей модели
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                busy = False
                log.warning("model %s err: %s", model, str(exc)[:120])
                break
    if busy:
        raise RuntimeError("MODEL_BUSY: бесплатная vision-модель перегружена (429/1305)")
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
    return f"💥 Упал с ошибкой: {exc}"


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
    finish = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
        finish = resp.choices[0].finish_reason or ""
    except Exception:  # noqa: BLE001 — кривой/пустой ответ провайдера
        text = ""
    log.info("full: finish=%s chars=%d", finish, len(text))
    if not text:
        text = (
            f"⚠️ Нейронка отфильтровала ответ (finish_reason={finish}). "
            "Попробуй другое фото: прямой ракурс, дневной свет, без очков."
        )
    return text


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
    text, finish = _extract_text(resp)
    log.info("simple: finish=%s chars=%d", finish, len(text))
    return text


# Контекст для кнопки "подробный разбор": user_id -> (photo_bytes, metrics_text).
_pending_full: dict[int, tuple[bytes, str]] = {}

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
        "Просто отправь фото в чат 👇",
    )


@dp.message(Command("mog"))
async def cmd_mog(msg: Message) -> None:
    """Публичный вызов на мог-батл: ответь на сообщение + /mog или /mog @user."""
    from game import battles as mog_battles
    from game import db as mog_db

    opponent = None
    target = msg.reply_to_message
    if target and target.from_user and not target.from_user.is_bot:
        fu = target.from_user
        if fu.id != msg.from_user.id:
            opponent = {"id": fu.id, "name": fu.full_name}
    if opponent is None:
        for ent in msg.entities or []:
            if ent.type == "text_mention" and ent.user and not ent.user.is_bot:
                if ent.user.id != msg.from_user.id:
                    opponent = {"id": ent.user.id, "name": ent.user.full_name}
                break
    if opponent is None:
        await msg.reply(
            "Ткни, кого могнуть: ответь на его сообщение и напиши /mog — "
            "или упомяни его через @username."
        )
        return

    challenger = {"id": msg.from_user.id, "name": msg.from_user.full_name}
    try:
        battle_id = await mog_battles.create(msg.chat.id, challenger, opponent)
    except Exception as exc:  # noqa: BLE001 — БД может быть не инициализирована
        log.exception("battle create failed")
        await msg.reply(f"💥 Не смог создать батл: {exc}")
        return

    text = (
        f"⚔️ <b>{challenger['name']}</b> вызывает <b>{opponent['name']}</b> "
        "на МОГ-БАТТЛ!\n\n"
        f"{opponent['name']}, просто пришли <b>своё фото</b> в этот чат — и бой закрыт. "
        "Кто кого смог."
    )
    trial = await mog_db.get_battle(battle_id)
    if trial and not trial["challenger_score"]:
        text += (
            "\n\n⚠️ У вызывающего ещё нет ролла — его счёт 0.0000. "
            "Пусть сначала кинет фото в чат."
        )
    kb = None
    if PUBLIC_BASE_URL:
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="⚔️ Принять бой (мини-апп)",
                        web_app=WebAppInfo(url=f"{PUBLIC_BASE_URL}/app?battle={battle_id}"),
                    )
                ]
            ]
        )
    await msg.reply(text, reply_markup=kb, parse_mode="HTML")

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
        await cb.answer(f"Не смог выставить счёт: {exc}", show_alert=True)


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


def _mog_tag(photo_bytes: bytes) -> str:
    """МОГ-балл + две смешные добивки + дисклеймер для любой выдачи (спек §7)."""
    import hashlib

    from game import config as mog_config
    from game import rolls as mog_rolls

    score = round(mog_rolls.base_score(photo_bytes), 4)
    tier = mog_rolls.tier_for(score)
    punch = _MOG_PUNCHLINES[
        int(hashlib.sha256(photo_bytes).hexdigest(), 16) % len(_MOG_PUNCHLINES)
    ]
    return (
        f"\n\n🎲 <b>МОГ: {score:.4f} · {tier}</b>\n{punch}\n"
        f"<i>{mog_config.DISCLAIMER}</i>"
    )


# ───────────────────────── НЕОН-СКАН «СЛАБЫЕ МЕСТА» ─────────────────────────
# Рисует киберпанк-рамки поверх зон лица: красным — стремные, зелёным — хорошие.
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


def _neon_color(score: int) -> tuple[int, int, int]:
    if score < 40:
        return (255, 45, 94)    # неон-красный — слабое место
    if score >= 60:
        return (57, 255, 136)   # неон-зелёный — сильная зона
    return (34, 211, 238)       # циан — нейтрально


def _draw_neon(d, box, color) -> None:
    """Рамка с «короной» по углам + многослойным glow."""
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return
    for width, alpha in ((14, 30), (8, 65), (3, 150)):
        d.rectangle([x1, y1, x2, y2], outline=color + (alpha,), width=width)
    d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 235), width=1)
    ln = max(10, int(min(x2 - x1, y2 - y1) * 0.18))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        d.line([(cx, cy), (cx + dx * ln, cy)], fill=color + (255,), width=3)
        d.line([(cx, cy), (cx, cy + dy * ln)], fill=color + (255,), width=3)


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
        color = _neon_color(score)
        _draw_neon(d, box, color)
        items.append((key, label, score, color))

    glow = overlay.filter(ImageFilter.GaussianBlur(6))
    base = img.convert("RGBA")
    out = Image.alpha_composite(Image.alpha_composite(base, glow), overlay)
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
    lines.append("🟢 halo: " + (", ".join(good) if good else "нет"))
    lines.append("")
    lines.append(f"<i>{mog_config.DISCLAIMER}</i>")
    return "\n".join(lines)


@dp.message(F.photo)
async def on_photo(msg: Message, bot: Bot) -> None:
    if not TELEGRAM_TOKEN or not MODEL_API_KEY:
        await msg.answer("⚠️ Бот не настроен: нет TELEGRAM_TOKEN или MODEL_API_KEY в .env")
        return

    status = await msg.answer(
        "📐 Изучаю пропорции…" if BIOMETRY_AVAILABLE else "🧠 Нейронка думает…"
    )
    tmp_path = ""
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
            metrics_text = format_metrics({"face_found": False, "error": "biometry_unavailable"})

        # 2. Первый слой: разбор простыми словами (тоже в треде)
        simple = await asyncio.to_thread(analyze_simple, raw, metrics_text)

        # Запоминаем контекст для кнопки "подробный разбор"
        _pending_full[msg.from_user.id] = (raw, metrics_text)

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📊 Подробный разбор", callback_data="full_roast")],
                [InlineKeyboardButton(text="🎯 Слабые места (неон)", callback_data="weakspots")],
                [InlineKeyboardButton(text="🧬 Замеры (тех)", callback_data="metrics")],
            ]
        )
        await status.delete()
        mog_line = _mog_tag(raw)
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
    raw, metrics_text = data
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
    raw, _metrics = data
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
        await cb.message.answer(f"💥 Не смог нарисовать скан: {exc}")


@dp.callback_query(F.data == "metrics")
async def on_metrics(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка 'Замеры (тех)': сырые числа биометрии — только по запросу, под капот."""
    data = _pending_full.get(cb.from_user.id)
    if not data:
        await cb.answer("Фото устарело — кинь заново 👇", show_alert=True)
        return
    await cb.answer("Под капот…")
    _raw, metrics_text = data
    await cb.message.answer(
        f"🧬 <b>ТЕХ-ЗАМЕРЫ</b> <i>(под капот)</i>\n<code>{metrics_text}</code>",
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
            log.exception("set_webhook не удался — апдейты не придут, но health-check жив")

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
