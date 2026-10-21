"""HTTP API MOG + раздача мини-аппа.

Подключается к aiohttp-приложению бота через register_webapp(app, bot).
Авторизация — заголовок X-Init-Data (Telegram.WebApp.initData).
"""
from __future__ import annotations

import base64
import logging
import os
import time
import uuid

from aiogram import Bot
from aiohttp import web

from . import auth, battles, config, db, payments, rolls

log = logging.getLogger("mog-api")

_bot: Bot | None = None

# Байты фото — только в памяти (не на диск), TTL ~15 мин.
_photos: dict[str, tuple[bytes, float]] = {}

MAX_PHOTO_BYTES = 12 * 1024 * 1024  # 12 MB


# ---------------------------------------------------------------- helpers


def _json(payload: dict | list, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def _gc_photos() -> None:
    now = time.time()
    for token in [t for t, (_, ts) in _photos.items() if now - ts > config.PHOTO_TTL_SECONDS]:
        _photos.pop(token, None)


PHOTOS_BYTES_MAX = 32 * 1024 * 1024  # ????????? ?? ????? ?????? ?? free-????????


def _store_photo(data: bytes) -> str:
    _gc_photos()
    token = uuid.uuid4().hex
    _photos[token] = (data, time.time())
    # ??? ?????? ????-??? ? ?????? ???????? ?? ?????? ?????? ??? ?????? 512MB-????????.
    total = sum(len(item[0]) for item in _photos.values())
    while total > PHOTOS_BYTES_MAX and len(_photos) > 1:
        oldest = min(_photos, key=lambda t: _photos[t][1])
        if oldest == token:
            break
        total -= len(_photos.pop(oldest)[0])
    return token


def _get_photo(token: str) -> bytes | None:
    item = _photos.get(token)
    if not item:
        return None
    data, ts = item
    if time.time() - ts > config.PHOTO_TTL_SECONDS:
        _photos.pop(token, None)
        return None
    return data


def _current_user(request: web.Request) -> dict | None:
    init_data = request.headers.get("X-Init-Data", "")
    data = auth.verify_init_data(init_data, config.BOT_TOKEN)
    if not data or not data.get("user"):
        return None
    return data["user"]


async def _body(request: web.Request) -> dict:
    try:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001 — кривое тело запроса
        return {}


def _decode_b64(raw: str) -> bytes | None:
    if not raw:
        return None
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- API handlers


async def h_health(_request: web.Request) -> web.Response:
    return _json(
        {
            "ok": True,
            "service": "mog",
            "version": config.APP_VERSION,
            "provider": os.getenv("MODEL_NAME", "zhipu"),
        }
    )


async def h_me(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    user = await db.get_or_create_user(u["id"], u.get("first_name", ""), u.get("username", ""))
    best = await db.get_best_roll(u["id"])
    history = await db.get_history(u["id"], 10)
    return _json(
        {
            "user": {
                "id": user["id"],
                "first_name": user["first_name"],
                "username": user["username"],
            },
            "rolls_left": user["rolls_left"],
            "bonus_rolls": user.get("bonus_rolls", 0),
            "crit_tokens": user.get("crit_tokens", 0),
            "rolls_total": user["rolls_total"],
            "crits": user["crits"],
            "invites_today": user["invites_today"],
            "best_roll": best,
            "history": history,
            "disclaimer": config.DISCLAIMER,
            "catalog": config.STARS_CATALOG,
        }
    )


async def h_photo(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    body = await _body(request)
    data = _decode_b64(body.get("photo_b64", ""))
    if not data:
        return _json({"error": "no_photo"}, 400)
    if len(data) > MAX_PHOTO_BYTES:
        return _json({"error": "too_large"}, 413)
    return _json({"photo_token": _store_photo(data)})


async def h_roll(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    body = await _body(request)
    data = _get_photo(body.get("photo_token", ""))
    if not data:
        return _json({"error": "photo_expired"}, 400)

    user = await db.get_or_create_user(u["id"], u.get("first_name", ""), u.get("username", ""))
    if user["rolls_left"] + user.get("bonus_rolls", 0) <= 0:
        return _json({"error": "no_rolls"}, 402)
    if not await db.spend_roll(u["id"]):
        return _json({"error": "no_rolls"}, 402)

    crit_used = False
    if body.get("use_crit"):
        crit_used = await db.use_crit_token(u["id"])

    result = rolls.roll(data, force_crit=crit_used)
    roll_id = await db.add_roll(u["id"], result)
    user = await db.get_or_create_user(u["id"])

    return _json(
        {
            "roll_id": roll_id,
            "score": result["score"],
            "tier": result["tier"],
            "crit": result["crit"],
            "crit_token_used": crit_used,
            "failed": result["failed"],
            "stats": result["stats"],
            "roast": result["roast"],
            "disclaimer": config.DISCLAIMER,
            "rolls_left": user["rolls_left"],
            "bonus_rolls": user.get("bonus_rolls", 0),
            "crit_tokens": user.get("crit_tokens", 0),
            "rolls_total": user["rolls_total"],
            "crits": user["crits"],
        }
    )


async def h_leaderboard(request: web.Request) -> web.Response:
    if not _current_user(request):
        return _json({"error": "unauthorized"}, 401)
    return _json({"top": await db.leaderboard(20)})


async def h_battle_get(request: web.Request) -> web.Response:
    if not _current_user(request):
        return _json({"error": "unauthorized"}, 401)
    try:
        battle_id = int(request.match_info["id"])
    except (KeyError, ValueError):
        return _json({"error": "bad_id"}, 400)

    b = await db.get_battle(battle_id)
    if not b:
        return _json({"error": "not_found"}, 404)
    return _json(
        {
            "id": b["id"],
            "status": b["status"],
            "challenger": {
                "id": b["challenger_id"],
                "name": b["challenger_name"],
                "score": b["challenger_score"],
            },
            "opponent": {
                "id": b["opponent_id"],
                "name": b["opponent_name"],
                "score": b["opponent_score"],
            },
            "winner_id": b["winner_id"],
        }
    )


async def h_battle_photo(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    try:
        battle_id = int(request.match_info["id"])
    except (KeyError, ValueError):
        return _json({"error": "bad_id"}, 400)

    b = await db.get_battle(battle_id)
    if not b:
        return _json({"error": "not_found"}, 404)
    if b["status"] != "pending":
        return _json({"error": "already_done"}, 409)
    if u["id"] != b["opponent_id"]:
        return _json({"error": "not_opponent"}, 403)

    body = await _body(request)
    data = _get_photo(body.get("photo_token", ""))
    if not data:
        return _json({"error": "photo_expired"}, 400)

    user = await db.get_or_create_user(u["id"], u.get("first_name", ""), u.get("username", ""))
    if user["rolls_left"] <= 0:
        return _json({"error": "no_rolls"}, 402)
    if not await db.spend_roll(u["id"]):
        return _json({"error": "no_rolls"}, 402)

    result = rolls.roll(data)
    await db.add_roll(u["id"], result)
    resolved = await battles.resolve(battle_id, result["score"])
    await _push_battle(resolved)

    return _json(
        {
            "winner_id": resolved["winner_id"],
            "result": resolved,
            "roll": {
                "score": result["score"],
                "tier": result["tier"],
                "crit": result["crit"],
                "stats": result["stats"],
                "disclaimer": config.DISCLAIMER,
            },
        }
    )


async def h_invite_claim(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    body = await _body(request)
    try:
        invited_id = int(body.get("invited_id", 0))
    except (TypeError, ValueError):
        invited_id = 0
    if invited_id == 0 or invited_id == u["id"]:
        return _json({"error": "bad_invite"}, 400)

    granted = await db.grant_invite_roll(u["id"])
    user = await db.get_or_create_user(u["id"])
    return _json({"granted": granted, "rolls_left": user["rolls_left"]})


async def h_buy(request: web.Request) -> web.Response:
    u = _current_user(request)
    if not u:
        return _json({"error": "unauthorized"}, 401)

    body = await _body(request)
    product_key = body.get("product", "")
    if product_key not in config.STARS_CATALOG:
        return _json({"error": "bad_product"}, 400)
    if _bot is None:
        return _json({"error": "no_bot"}, 503)

    try:
        link = await payments.create_link(_bot, product_key, u["id"])
    except Exception:  # noqa: BLE001
        log.exception("create_invoice_link failed")
        return _json({"error": "invoice_failed"}, 502)
    return _json({"invoice_link": link})


async def _push_battle(r: dict) -> None:
    if _bot is None:
        return
    try:
        await _bot.send_message(
            r["chat_id"],
            battles.result_text(r),
            parse_mode="HTML",
            reply_markup=battles.rematch_keyboard(r),
        )
    except Exception:  # noqa: BLE001 — чат мог удалить бота
        log.exception("battle push failed")


# ---------------------------------------------------------------- static


def _static(filename: str) -> web.Response:
    path = config.WEBAPP_DIR / filename
    if not path.exists():
        return web.Response(status=404, text=f"{filename} not found")
    return web.FileResponse(path, headers={"Cache-Control": "no-store"})


async def h_index(_request: web.Request) -> web.Response:
    return _static("index.html")


async def h_appjs(_request: web.Request) -> web.Response:
    return _static("app.js")


async def h_stylecss(_request: web.Request) -> web.Response:
    return _static("style.css")


# ---------------------------------------------------------------- register

def register_webapp(app: web.Application, bot: Bot) -> None:
    """Навешивает /api/* и раздачу /app на aiohttp-приложение бота."""
    global _bot
    _bot = bot

    app.router.add_get("/api/health", h_health)
    app.router.add_get("/api/me", h_me)
    app.router.add_post("/api/photo", h_photo)
    app.router.add_post("/api/roll", h_roll)
    app.router.add_get("/api/leaderboard", h_leaderboard)
    app.router.add_get("/api/battle/{id}", h_battle_get)
    app.router.add_post("/api/battle/{id}/photo", h_battle_photo)
    app.router.add_post("/api/invite/claim", h_invite_claim)
    app.router.add_post("/api/buy", h_buy)

    app.router.add_get("/app", h_index)
    app.router.add_get("/app/", h_index)
    app.router.add_get("/app/app.js", h_appjs)
    app.router.add_get("/app/style.css", h_stylecss)

    app.on_startup.append(_on_startup)

    log.info("/api/* и /app зарегистрированы")


async def _on_startup(_app: web.Application) -> None:
    await db.init_db()
