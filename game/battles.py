"""Баты MOG: создание, фиксация ролла оппонента, разрешение исхода."""
from __future__ import annotations

from . import db


async def create(chat_id: int, challenger: dict, opponent: dict) -> int:
    """Создаёт батл. challenger/opponent — dict с id и name.

    challenger_score берём из лучшего ролла челленджера (0, если ещё не крутил).
    """
    best = await db.get_best_roll(challenger["id"])
    challenger_score = best["score"] if best else 0.0
    return await db.create_battle(
        chat_id=chat_id,
        challenger_id=challenger["id"],
        challenger_name=challenger["name"],
        challenger_score=challenger_score,
        opponent_id=opponent["id"],
        opponent_name=opponent["name"],
    )


async def create_open(chat_id: int, challenger: dict) -> int:
    """Открытый вызов: оппонента нет (opponent_id=0) — кто первый примет, тот и бой."""
    best = await db.get_best_roll(challenger["id"])
    challenger_score = best["score"] if best else 0.0
    return await db.create_battle(
        chat_id=chat_id,
        challenger_id=challenger["id"],
        challenger_name=challenger["name"],
        challenger_score=challenger_score,
        opponent_id=0,
        opponent_name="",
    )


async def accept(battle_id: int, user: dict) -> tuple[bool, str]:
    """Принять вызов кнопкой. Работает и для открытого (0), и для адресного вызова."""
    battle = await db.get_battle(battle_id)
    if not battle or battle["status"] != "pending":
        return False, "Этот вызов уже закрыт."
    if battle["challenger_id"] == user["id"]:
        return False, "Себя нельзя — позови другого 😈"
    opp = battle["opponent_id"] or 0
    if opp and opp != user["id"]:
        return False, "Вызов уже принят другим."
    if not opp:
        await db.set_battle_opponent(battle_id, user["id"], user["name"])
    return True, "Принято ⚔️"


async def resolve(battle_id: int, opponent_score: float) -> dict:
    """Фиксирует ролл оппонента и определяет победителя (draw при равенстве).

    Учитывает купленные артефакты (модель B: capped-буст + контры «Анти-сглаз»).
    """
    from . import artifacts as mog_art

    await db.set_opponent_roll(battle_id, opponent_score)
    battle = await db.get_battle(battle_id)
    assert battle is not None

    cid = battle["challenger_id"]
    oid = battle["opponent_id"]

    # Соперник мог не принять вызов (opponent_id=0) — тогда бой без второй стороны.
    if not oid:
        cid_score = float(battle["challenger_score"] or 0.0)
        await db.finish_battle(battle_id, cid)
        return {
            "battle_id": battle_id,
            "chat_id": battle["chat_id"],
            "challenger_id": cid,
            "challenger_name": battle["challenger_name"],
            "challenger_score": cid_score,
            "opponent_id": 0,
            "opponent_name": "",
            "opponent_score": float(opponent_score or 0.0),
            "winner_id": cid,
            "challenger_comp": {},
            "opponent_comp": {},
        }

    c_items = await db.battle_artifacts(battle_id, cid)
    o_items = await db.battle_artifacts(battle_id, oid)
    comp = mog_art.compute(
        [
            {"id": cid, "score": battle["challenger_score"] or 0.0, "items": c_items},
            {"id": oid, "score": opponent_score or 0.0, "items": o_items},
        ]
    )
    cs = comp[cid]["eff"]
    os_ = comp[oid]["eff"]

    if cs > os_:
        winner_id = cid
    elif os_ > cs:
        winner_id = oid
    else:
        winner_id = None  # ничья

    await db.finish_battle(battle_id, winner_id)
    return {
        "battle_id": battle_id,
        "chat_id": battle["chat_id"],
        "challenger_id": cid,
        "challenger_name": battle["challenger_name"],
        "challenger_score": cs,
        "opponent_id": oid,
        "opponent_name": battle["opponent_name"],
        "opponent_score": os_,
        "winner_id": winner_id,
        "challenger_comp": comp[cid],
        "opponent_comp": comp[oid],
    }


def view_for(r: dict, user_id: int) -> dict:
    """Сводит результат батла к «ты / соперник» — для картинки и подписи.

    Нужен, потому что фото может прислать и вызывающий, и принимающий: смотрим
    на user_id, а не на роль в записи боя.
    """
    is_ch = r["challenger_id"] == user_id
    you = r["challenger_score"] if is_ch else r["opponent_score"]
    foe = r["opponent_score"] if is_ch else r["challenger_score"]
    you_n = r["challenger_name"] if is_ch else r["opponent_name"]
    foe_n = r["opponent_name"] if is_ch else r["challenger_name"]
    return {
        "you_name": (you_n or "ты").split()[0][:14] if you_n else "ты",
        "foe_name": (foe_n or "соперник").split()[0][:14] if foe_n else "соперник",
        "you_score": float(you or 0.0),
        "foe_score": float(foe or 0.0),
        "draw": r["winner_id"] is None,
        "you_won": r["winner_id"] == user_id,
    }


def result_text(r: dict) -> str:
    """Табло результата батла: два балла, разрыв, победитель и что дальше."""
    from html import escape

    # Имена приходят из Telegram как есть — экранируем, иначе «<» в нике
    # роняет отправку с parse_mode=HTML.
    c_name, o_name = escape(r["challenger_name"]), escape(r["opponent_name"])
    c = f'<a href="tg://user?id={r["challenger_id"]}">{c_name}</a>'
    o = f'<a href="tg://user?id={r["opponent_id"]}">{o_name}</a>'
    cs, os_ = r["challenger_score"], r["opponent_score"]
    margin = abs(cs - os_)

    if r["winner_id"] is None:
        verdict = f"🤝 <b>НИЧЬЯ</b> — оба на {cs:.4f}. Скучно, но честно."
        first, second = f"{c} — {cs:.4f}", f"{o} — {os_:.4f}"
    elif r["winner_id"] == r["challenger_id"]:
        verdict = f"👑 <b>ПОБЕДА: {c}</b> · разрыв {margin:.4f}"
        first, second = f"🥇 {c} — <b>{cs:.4f}</b>", f"🥈 {o} — {os_:.4f}"
    else:
        verdict = f"👑 <b>ПОБЕДА: {o}</b> · разрыв {margin:.4f}"
        first, second = f"🥇 {o} — <b>{os_:.4f}</b>", f"🥈 {c} — {cs:.4f}"

    boost_block = ""
    c_comp = r.get("challenger_comp") or {}
    o_comp = r.get("opponent_comp") or {}
    if c_comp or o_comp:
        from . import artifacts as mog_art

        blines = [
            mog_art.result_line(r["challenger_name"], c_comp),
            mog_art.result_line(r["opponent_name"], o_comp),
        ]
        blines = [b for b in blines if b]
        if blines:
            boost_block = "\n".join(blines) + "\n\n"

    return (
        f"🏒 <b>МОГ-БАТТЛ #{r['battle_id']}</b> · итог\n\n"
        f"{first}\n{second}\n\n"
        f"{boost_block}"
        f"{verdict}\n\n"
        "Хочешь отыграться — жми «🔁 Реванш» под этим сообщением.\n"
        "<i>MOG — игра. Баллы случайны, не оценка внешности.</i>"
    )


def rematch_keyboard(r: dict):
    """Кнопка реванша под результатом батла (ленивый импорт aiogram)."""
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔁 Реванш!",
                    callback_data=f"rematch:{r['battle_id']}",
                )
            ]
        ]
    )

