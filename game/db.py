"""SQLite-слой MOG (aiosqlite): users / rolls / battles."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import aiosqlite

from . import config

_db: aiosqlite.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    first_name    TEXT    NOT NULL DEFAULT '',
    username      TEXT    NOT NULL DEFAULT '',
    rolls_left    INTEGER NOT NULL DEFAULT 0,
    rolls_day     TEXT    NOT NULL DEFAULT '',
    bonus_rolls   INTEGER NOT NULL DEFAULT 0,
    crit_tokens   INTEGER NOT NULL DEFAULT 0,
    rolls_total   INTEGER NOT NULL DEFAULT 0,
    crits         INTEGER NOT NULL DEFAULT 0,
    invites_today INTEGER NOT NULL DEFAULT 0,
    invites_day   TEXT    NOT NULL DEFAULT '',
    best_score    REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS rolls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    score      REAL    NOT NULL,
    tier       TEXT    NOT NULL,
    crit       INTEGER NOT NULL DEFAULT 0,
    failed     INTEGER NOT NULL DEFAULT 0,
    stats_json TEXT    NOT NULL DEFAULT '{}',
    roast      TEXT    NOT NULL DEFAULT '',
    photo_hash TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_rolls_user ON rolls(user_id);
CREATE INDEX IF NOT EXISTS idx_rolls_score ON rolls(score DESC);

CREATE TABLE IF NOT EXISTS battles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id          INTEGER NOT NULL,
    challenger_id    INTEGER NOT NULL,
    challenger_name  TEXT    NOT NULL DEFAULT '',
    challenger_score REAL    NOT NULL DEFAULT 0,
    opponent_id      INTEGER NOT NULL,
    opponent_name    TEXT    NOT NULL DEFAULT '',
    opponent_score   REAL    NOT NULL DEFAULT 0,
    status           TEXT    NOT NULL DEFAULT 'pending',
    winner_id        INTEGER,
    created_at       TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS arenas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    host_id     INTEGER NOT NULL,
    host_name   TEXT    NOT NULL DEFAULT '',
    max_players INTEGER NOT NULL DEFAULT 8,
    status      TEXT    NOT NULL DEFAULT 'lobby',
    created_at  TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS arena_players (
    arena_id  INTEGER NOT NULL,
    user_id   INTEGER NOT NULL,
    name      TEXT    NOT NULL DEFAULT '',
    score     REAL    NOT NULL DEFAULT 0,
    tier      TEXT    NOT NULL DEFAULT 'Common',
    joined_at TEXT    NOT NULL DEFAULT '',
    UNIQUE(arena_id, user_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return date.today().isoformat()


async def init_db() -> None:
    global _db
    if _db is not None:
        return
    _db = await aiosqlite.connect(config.DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.executescript(SCHEMA)
    # Мягкая миграция: добавляем колонки, если база создана старой версией.
    for col, ddl in (
        ("bonus_rolls", "INTEGER NOT NULL DEFAULT 0"),
        ("crit_tokens", "INTEGER NOT NULL DEFAULT 0"),
    ):
        try:
            await _db.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        except Exception:  # noqa: BLE001 — колонка уже есть
            pass
    await _db.commit()


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None


# ---------------------------------------------------------------- users


async def _refresh_daily(user_id: int) -> None:
    """Сброс дневных счётчиков, если наступил новый день."""
    assert _db is not None
    today = _today()
    await _db.execute(
        "UPDATE users SET rolls_left=?, rolls_day=? WHERE id=? AND rolls_day!=?",
        (config.FREE_ROLLS_PER_DAY, today, user_id, today),
    )
    await _db.execute(
        "UPDATE users SET invites_today=0, invites_day=? WHERE id=? AND invites_day!=?",
        (today, user_id, today),
    )
    await _db.commit()


async def get_or_create_user(
    user_id: int, first_name: str = "", username: str = ""
) -> dict:
    assert _db is not None
    today = _today()
    cur = await _db.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = await cur.fetchone()

    if row is None:
        await _db.execute(
            "INSERT INTO users (id, first_name, username, rolls_left, rolls_day,"
            " rolls_total, crits, invites_today, invites_day, best_score, created_at)"
            " VALUES (?,?,?,?,?,0,0,0,?,0,?)",
            (user_id, first_name, username, config.FREE_ROLLS_PER_DAY, today, today, _now()),
        )
        await _db.commit()
    else:
        first_name = first_name or row["first_name"]
        username = username or row["username"]
        await _db.execute(
            "UPDATE users SET first_name=?, username=? WHERE id=?",
            (first_name, username, user_id),
        )
        await _db.commit()

    await _refresh_daily(user_id)
    cur = await _db.execute("SELECT * FROM users WHERE id=?", (user_id,))
    return dict(await cur.fetchone())


async def spend_roll(user_id: int) -> bool:
    """Списывает один ролл. Сначала бесплатные, потом купленные. False — если нет."""
    assert _db is not None
    await _refresh_daily(user_id)
    cur = await _db.execute(
        "UPDATE users SET rolls_left=rolls_left-1 WHERE id=? AND rolls_left>0",
        (user_id,),
    )
    if cur.rowcount > 0:
        await _db.commit()
        return True
    cur = await _db.execute(
        "UPDATE users SET bonus_rolls=bonus_rolls-1 WHERE id=? AND bonus_rolls>0",
        (user_id,),
    )
    await _db.commit()
    return cur.rowcount > 0


async def add_purchase(user_id: int, rolls: int = 0, crit: int = 0) -> None:
    """Начисляет купленное за Stars (постоянные роллы и крит-токены)."""
    assert _db is not None
    await _db.execute(
        "UPDATE users SET bonus_rolls=bonus_rolls+?, crit_tokens=crit_tokens+? WHERE id=?",
        (rolls, crit, user_id),
    )
    await _db.commit()


async def use_crit_token(user_id: int) -> bool:
    """Тратит крит-токен (гарант Mythic). False — если токенов нет."""
    assert _db is not None
    cur = await _db.execute(
        "UPDATE users SET crit_tokens=crit_tokens-1 WHERE id=? AND crit_tokens>0",
        (user_id,),
    )
    await _db.commit()
    return cur.rowcount > 0


async def grant_invite_roll(user_id: int) -> bool:
    """+1 ролл за приглашённого друга (в пределах дневного капа)."""
    assert _db is not None
    await _refresh_daily(user_id)
    user = await get_or_create_user(user_id)
    if user["invites_today"] >= config.INVITE_ROLLS_CAP:
        return False
    await _db.execute(
        "UPDATE users SET rolls_left=rolls_left+1, invites_today=invites_today+1 WHERE id=?",
        (user_id,),
    )
    await _db.commit()
    return True


async def add_roll(user_id: int, result: dict) -> int:
    assert _db is not None
    cur = await _db.execute(
        "INSERT INTO rolls (user_id, score, tier, crit, failed, stats_json, roast,"
        " photo_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            user_id,
            result["score"],
            result["tier"],
            int(bool(result["crit"])),
            int(bool(result["failed"])),
            json.dumps(result["stats"]),
            result["roast"],
            result["photo_hash"],
            _now(),
        ),
    )
    roll_id = cur.lastrowid
    await _db.execute(
        "UPDATE users SET rolls_total=rolls_total+1, crits=crits+? WHERE id=?",
        (int(bool(result["crit"])), user_id),
    )
    await _db.execute(
        "UPDATE users SET best_score=? WHERE id=? AND ? > best_score",
        (result["score"], user_id, result["score"]),
    )
    await _db.commit()
    return int(roll_id)


async def get_best_roll(user_id: int) -> dict | None:
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM rolls WHERE user_id=? ORDER BY score DESC LIMIT 1", (user_id,)
    )
    row = await cur.fetchone()
    return _roll_row(row) if row else None


async def get_history(user_id: int, limit: int = 10) -> list[dict]:
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM rolls WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)
    )
    return [_roll_row(r) for r in await cur.fetchall()]


def _roll_row(row: aiosqlite.Row) -> dict:
    try:
        stats = json.loads(row["stats_json"])
    except (json.JSONDecodeError, TypeError):
        stats = {}
    return {
        "id": row["id"],
        "score": row["score"],
        "tier": row["tier"],
        "crit": bool(row["crit"]),
        "failed": bool(row["failed"]),
        "stats": stats,
        "roast": row["roast"],
        "created_at": row["created_at"],
    }


async def leaderboard(limit: int = 20) -> list[dict]:
    assert _db is not None
    cur = await _db.execute(
        "SELECT u.first_name, u.username, MAX(r.score) AS score, r.tier"
        " FROM rolls r JOIN users u ON u.id=r.user_id"
        " GROUP BY r.user_id ORDER BY score DESC LIMIT ?",
        (limit,),
    )
    rows = await cur.fetchall()
    out = []
    for i, row in enumerate(rows, start=1):
        name = row["first_name"] or row["username"] or f"id{row['first_name']}"
        out.append(
            {"rank": i, "name": name, "score": row["score"], "tier": row["tier"]}
        )
    return out


# ---------------------------------------------------------------- battles


async def create_battle(
    chat_id: int,
    challenger_id: int,
    challenger_name: str,
    challenger_score: float,
    opponent_id: int,
    opponent_name: str,
) -> int:
    assert _db is not None
    cur = await _db.execute(
        "INSERT INTO battles (chat_id, challenger_id, challenger_name, challenger_score,"
        " opponent_id, opponent_name, status, created_at) VALUES (?,?,?,?,?,?,'pending',?)",
        (
            chat_id,
            challenger_id,
            challenger_name,
            challenger_score,
            opponent_id,
            opponent_name,
            _now(),
        ),
    )
    await _db.commit()
    return int(cur.lastrowid)


async def get_battle(battle_id: int) -> dict | None:
    assert _db is not None
    cur = await _db.execute("SELECT * FROM battles WHERE id=?", (battle_id,))
    row = await cur.fetchone()
    return dict(row) if row else None


async def set_opponent_roll(battle_id: int, score: float) -> None:
    assert _db is not None
    await _db.execute(
        "UPDATE battles SET opponent_score=? WHERE id=?", (score, battle_id)
    )
    await _db.commit()


async def finish_battle(battle_id: int, winner_id: int | None) -> None:
    assert _db is not None
    await _db.execute(
        "UPDATE battles SET status='done', winner_id=? WHERE id=?", (winner_id, battle_id)
    )
    await _db.commit()


async def find_pending_battle(chat_id: int, opponent_id: int) -> dict | None:
    """Свежий незакрытый батл в этом чате, где юзер — оппонент (ждёт его фото)."""
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM battles WHERE chat_id=? AND opponent_id=? AND status='pending'"
        " ORDER BY id DESC LIMIT 1",
        (chat_id, opponent_id),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- arena


async def create_arena(
    chat_id: int, host_id: int, host_name: str, max_players: int
) -> int:
    assert _db is not None
    cur = await _db.execute(
        "INSERT INTO arenas (chat_id, host_id, host_name, max_players, status, created_at)"
        " VALUES (?,?,?,?,'lobby',?)",
        (chat_id, host_id, host_name, max_players, _now()),
    )
    await _db.commit()
    return int(cur.lastrowid)


async def get_arena(arena_id: int) -> dict | None:
    assert _db is not None
    cur = await _db.execute("SELECT * FROM arenas WHERE id=?", (arena_id,))
    row = await cur.fetchone()
    return dict(row) if row else None


async def get_active_arena(chat_id: int) -> dict | None:
    """Свежайшее лобби в чате (status='lobby'), если есть."""
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM arenas WHERE chat_id=? AND status='lobby' ORDER BY id DESC LIMIT 1",
        (chat_id,),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def set_arena_status(arena_id: int, status: str) -> None:
    assert _db is not None
    await _db.execute("UPDATE arenas SET status=? WHERE id=?", (status, arena_id))
    await _db.commit()


async def add_arena_player(
    arena_id: int, user_id: int, name: str, score: float, tier: str
) -> bool:
    """True — добавили; False — уже был в игре (UNIQUE)."""
    assert _db is not None
    try:
        await _db.execute(
            "INSERT INTO arena_players (arena_id, user_id, name, score, tier, joined_at)"
            " VALUES (?,?,?,?,?,?)",
            (arena_id, user_id, name, score, tier, _now()),
        )
        await _db.commit()
        return True
    except Exception:  # noqa: BLE001 — UNIQUE(arena_id,user_id): уже в списке
        return False


async def remove_arena_player(arena_id: int, user_id: int) -> bool:
    assert _db is not None
    cur = await _db.execute(
        "DELETE FROM arena_players WHERE arena_id=? AND user_id=?", (arena_id, user_id)
    )
    await _db.commit()
    return cur.rowcount > 0


async def arena_players(arena_id: int) -> list[dict]:
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM arena_players WHERE arena_id=? ORDER BY score DESC", (arena_id,)
    )
    return [dict(r) for r in await cur.fetchall()]


async def arena_count(arena_id: int) -> int:
    assert _db is not None
    cur = await _db.execute(
        "SELECT COUNT(*) AS n FROM arena_players WHERE arena_id=?", (arena_id,)
    )
    row = await cur.fetchone()
    return int(row["n"]) if row else 0
