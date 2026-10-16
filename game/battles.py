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


async def resolve(battle_id: int, opponent_score: float) -> dict:
    """Фиксирует ролл оппонента и определяет победителя (draw при равенстве)."""
    await db.set_opponent_roll(battle_id, opponent_score)
    battle = await db.get_battle(battle_id)
    assert battle is not None

    cs = battle["challenger_score"] or 0.0
    os_ = opponent_score or 0.0

    if cs > os_:
        winner_id = battle["challenger_id"]
    elif os_ > cs:
        winner_id = battle["opponent_id"]
    else:
        winner_id = None  # ничья

    await db.finish_battle(battle_id, winner_id)
    return {
        "battle_id": battle_id,
        "chat_id": battle["chat_id"],
        "challenger_id": battle["challenger_id"],
        "challenger_name": battle["challenger_name"],
        "challenger_score": cs,
        "opponent_id": battle["opponent_id"],
        "opponent_name": battle["opponent_name"],
        "opponent_score": os_,
        "winner_id": winner_id,
    }


def result_text(r: dict) -> str:
    """Жёсткий текст результата батла для чата (тегает обоих)."""
    c = f'<a href="tg://user?id={r["challenger_id"]}">{r["challenger_name"]}</a>'
    o = f'<a href="tg://user?id={r["opponent_id"]}">{r["opponent_name"]}</a>'
    cs, os_ = r["challenger_score"], r["opponent_score"]

    if r["winner_id"] is None:
        head = f"🤝 <b>НИЧЬЯ</b> — оба на {cs:.2f}. Так себе мог."
    elif r["winner_id"] == r["challenger_id"]:
        head = f"💀 <b>{c} СМОГНУЛ {o}</b>"
    else:
        head = f"💀 <b>{o} СМОГНУЛ {c}</b>"

    return (
        f"⚔️ МОГ-БАТТЛ #{r['battle_id']}\n"
        f"{c}: <b>{cs:.2f}</b>  ·  {o}: <b>{os_:.2f}</b>\n\n"
        f"{head}\n\n"
        "<i>MOG — игра. Баллы случайны, не оценка внешности.</i>"
    )
