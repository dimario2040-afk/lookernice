"""SQLite-слой MOG (aiosqlite): users / rolls / battles."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

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

CREATE TABLE IF NOT EXISTS battle_artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    battle_id  INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    item       TEXT    NOT NULL,
    paid       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bart_battle ON battle_artifacts(battle_id);

CREATE TABLE IF NOT EXISTS artifact_free (
    user_id INTEGER NOT NULL,
    day     TEXT    NOT NULL,
    UNIQUE(user_id, day)
);

CREATE TABLE IF NOT EXISTS consents (
    user_id    INTEGER PRIMARY KEY,
    version    TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT ''
);

-- 🎁 Кредиты подарочных фич (реролл/студия), начисляются за подарок.
CREATE TABLE IF NOT EXISTS credits (
    user_id INTEGER NOT NULL,
    kind    TEXT    NOT NULL,
    qty     INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, kind)
);

-- 🎁 Одноразовые подарочные токены: создал — подарил — кто-то забрал.
CREATE TABLE IF NOT EXISTS gift_tokens (
    token      TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    created_by INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT '',
    claimed_by INTEGER
);

-- ⚔️ Дуэли по ссылке (без группы): вызывающий шлёт t.me-ссылку, соперник жмёт
-- и попадает в личку к боту. Живут отдельно от battles, т.к. чата у них нет —
-- итог бот пишет обоим в личку.
CREATE TABLE IF NOT EXISTS duel_links (
    code            TEXT PRIMARY KEY,
    challenger_id   INTEGER NOT NULL,
    challenger_name TEXT    NOT NULL DEFAULT '',
    challenger_score REAL   NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'open',  -- open|accepted|done|expired
    opponent_id     INTEGER,
    opponent_name   TEXT    NOT NULL DEFAULT '',
    opponent_score  REAL    NOT NULL DEFAULT 0,
    winner_id       INTEGER,
    created_at      TEXT    NOT NULL DEFAULT '',
    expires_at      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_duel_links_status ON duel_links(status, expires_at);
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


async def set_battle_opponent(
    battle_id: int, opponent_id: int, opponent_name: str
) -> None:
    """Назначает оппонента открытому вызову (opponent_id=0 → реальный юзер)."""
    assert _db is not None
    await _db.execute(
        "UPDATE battles SET opponent_id=?, opponent_name=? WHERE id=?",
        (opponent_id, opponent_name, battle_id),
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


async def list_open_battles(chat_id: int, exclude_id: int = 0) -> list[dict]:
    """Открытые вызовы в чате (никто ещё не принял), кроме вызовов самого юзера.

    Нужно, чтобы фото в чате могло закрыть открытый вызов без нажатия кнопки:
    «просто кинь фото» должно работать буквально, а не только на словах.
    """
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM battles WHERE chat_id=? AND opponent_id=0 AND status='pending'"
        " AND challenger_id<>? ORDER BY id DESC",
        (chat_id, exclude_id),
    )
    rows = await cur.fetchall()
    return [dict(r) for r in rows]


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


async def update_arena_player_score(
    arena_id: int, user_id: int, score: float, tier: str
) -> bool:
    """Обновляет снапшот балла в арене (кинул новое фото — пересчитали)."""
    assert _db is not None
    cur = await _db.execute(
        "UPDATE arena_players SET score=?, tier=? WHERE arena_id=? AND user_id=?",
        (score, tier, arena_id, user_id),
    )
    await _db.commit()
    return cur.rowcount > 0


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


# ---------------------------------------------------------------- артефакты (баттлы)


async def add_battle_artifact(
    battle_id: int, user_id: int, item: str, paid: bool = False
) -> int:
    """Прикрепляет артефакт к баттлу за конкретного игрока (моментально)."""
    assert _db is not None
    cur = await _db.execute(
        "INSERT INTO battle_artifacts (battle_id, user_id, item, paid, created_at)"
        " VALUES (?,?,?,?,?)",
        (battle_id, user_id, item, int(bool(paid)), _now()),
    )
    await _db.commit()
    return int(cur.lastrowid)


async def battle_artifacts(battle_id: int, user_id: int) -> list[str]:
    """Список ключей артефактов игрока в баттле (в порядке покупки)."""
    assert _db is not None
    cur = await _db.execute(
        "SELECT item FROM battle_artifacts WHERE battle_id=? AND user_id=? ORDER BY id",
        (battle_id, user_id),
    )
    return [r["item"] for r in await cur.fetchall()]


async def claim_free_artifact(user_id: int) -> bool:
    """True — если бесплатный артефакт сегодня ещё не брали (и теперь заняли)."""
    assert _db is not None
    try:
        await _db.execute(
            "INSERT INTO artifact_free (user_id, day) VALUES (?,?)", (user_id, _today())
        )
        await _db.commit()
        return True
    except Exception:  # noqa: BLE001 — UNIQUE(user_id, day): уже брали сегодня
        return False


# ---------------------------------------------------------------- согласия (ПДн)


async def get_consent(user_id: int) -> dict | None:
    """Текущая зафиксированная версия согласия юзера (или None)."""
    assert _db is not None
    cur = await _db.execute("SELECT * FROM consents WHERE user_id=?", (user_id,))
    row = await cur.fetchone()
    return dict(row) if row else None


async def record_consent(user_id: int, version: str) -> str:
    """Фиксирует согласие юзера на версию политики. Возвращает ISO-таймстамп."""
    assert _db is not None
    ts = _now()
    await _db.execute(
        "INSERT INTO consents (user_id, version, created_at) VALUES (?,?,?)"
        " ON CONFLICT(user_id) DO UPDATE SET version=excluded.version,"
        " created_at=excluded.created_at",
        (user_id, version, ts),
    )
    await _db.commit()
    return ts


# ---------------------------------------------------------------- кредиты/подарки


async def add_credit(user_id: int, kind: str, n: int = 1) -> int:
    """Начисляет кредит (реролл/студия). Возвращает новый остаток."""
    assert _db is not None
    await _db.execute(
        "INSERT INTO credits (user_id, kind, qty) VALUES (?,?,?)"
        " ON CONFLICT(user_id, kind) DO UPDATE SET qty = qty + excluded.qty",
        (user_id, kind, int(n)),
    )
    await _db.commit()
    cur = await _db.execute(
        "SELECT qty FROM credits WHERE user_id=? AND kind=?", (user_id, kind)
    )
    row = await cur.fetchone()
    return int(row["qty"]) if row else 0


async def get_credits(user_id: int) -> dict[str, int]:
    """Все ненулевые кредиты юзера: {kind: qty}."""
    assert _db is not None
    cur = await _db.execute("SELECT kind, qty FROM credits WHERE user_id=?", (user_id,))
    return {r["kind"]: int(r["qty"]) for r in await cur.fetchall() if int(r["qty"]) > 0}


async def spend_credit(user_id: int, kind: str) -> bool:
    """Списывает 1 кредит. True — если списание прошло (был остаток)."""
    assert _db is not None
    cur = await _db.execute(
        "UPDATE credits SET qty = qty - 1 WHERE user_id=? AND kind=? AND qty > 0",
        (user_id, kind),
    )
    await _db.commit()
    return cur.rowcount > 0


async def create_gift_token(kind: str, created_by: int) -> str:
    """Создаёт одноразовый подарочный токен. Возвращает token."""
    import secrets

    assert _db is not None
    token = secrets.token_urlsafe(9)
    await _db.execute(
        "INSERT INTO gift_tokens (token, kind, created_by, created_at) VALUES (?,?,?,?)",
        (token, kind, created_by, _now()),
    )
    await _db.commit()
    return token


async def claim_gift_token(token: str, user_id: int) -> tuple[bool, str, str]:
    """Атомарно забирает подарок. Возвращает (ok, kind, reason)."""
    assert _db is not None
    cur = await _db.execute(
        "SELECT kind, created_by, claimed_by FROM gift_tokens WHERE token=?", (token,)
    )
    row = await cur.fetchone()
    if not row:
        return False, "", "not_found"
    if row["claimed_by"] is not None:
        return False, row["kind"], "used"
    upd = await _db.execute(
        "UPDATE gift_tokens SET claimed_by=? WHERE token=? AND claimed_by IS NULL",
        (user_id, token),
    )
    await _db.commit()
    if upd.rowcount == 0:
        return False, row["kind"], "used"
    await add_credit(user_id, row["kind"], 1)
    return True, row["kind"], "ok"


# ------------------------------------------------- ????? ?? ?????? (duel_links)


async def create_duel_link(
    code: str, challenger_id: int, challenger_name: str, challenger_score: float, ttl_hours: int
) -> None:
    """??????? ????? ?? ??????. ???? ??????????? ??????????? ?? ?????? ??????."""
    assert _db is not None
    expires = (
        datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    ).isoformat(timespec="seconds")
    await _db.execute(
        "INSERT INTO duel_links (code, challenger_id, challenger_name, challenger_score,"
        " status, created_at, expires_at) VALUES (?,?,?,?,'open',?,?)",
        (code, challenger_id, challenger_name, challenger_score, _now(), expires),
    )
    await _db.commit()


async def get_duel_link(code: str) -> dict | None:
    assert _db is not None
    cur = await _db.execute("SELECT * FROM duel_links WHERE code=?", (code,))
    row = await cur.fetchone()
    return dict(row) if row else None


async def accept_duel_link(code: str, user_id: int, user_name: str) -> tuple[bool, str]:
    """???????? ????????? ?????. ?????????? (ok, reason)."""
    assert _db is not None
    cur = await _db.execute(
        "UPDATE duel_links SET status='accepted', opponent_id=?, opponent_name=?"
        " WHERE code=? AND status='open' AND opponent_id IS NULL",
        (user_id, user_name, code),
    )
    await _db.commit()
    return (cur.rowcount > 0), ("ok" if cur.rowcount > 0 else "closed")


async def resolve_duel_link(code: str, opponent_score: float, winner_id: int | None) -> None:
    assert _db is not None
    await _db.execute(
        "UPDATE duel_links SET status='done', opponent_score=?, winner_id=? WHERE code=?",
        (opponent_score, winner_id, code),
    )
    await _db.commit()


async def pending_duel_link_for(user_id: int) -> dict | None:
    """????????, ?? ?? ??????????? ????? ????? ????? ? ???? ??? ????."""
    assert _db is not None
    cur = await _db.execute(
        "SELECT * FROM duel_links WHERE opponent_id=? AND status='accepted'"
        " ORDER BY created_at DESC LIMIT 1",
        (user_id,),
    )
    row = await cur.fetchone()
    return dict(row) if row else None


async def expire_duel_links() -> list[dict]:
    """????? ???????????? ??????. ?????????? ?????????? ??? ???????????."""
    assert _db is not None
    now = _now()
    cur = await _db.execute(
        "SELECT * FROM duel_links WHERE status='open' AND expires_at<?", (now,)
    )
    rows = [dict(r) for r in await cur.fetchall()]
    if not rows:
        return []
    await _db.execute(
        "UPDATE duel_links SET status='expired' WHERE status='open' AND expires_at<?", (now,)
    )
    await _db.commit()
    return rows
