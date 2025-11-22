# app/gamification/leaderboard.py
import enum
from typing import Literal, List
import redis.asyncio as aioredis
from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User as DBUser  # assumes has id, total_points, etc.

# Redis connection (configure URL via settings)
redis = aioredis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)

LEADERBOARD_KEY = "lb:discipline:global"  # one global leaderboard for now

class EventType(str, enum.Enum):
    FOLLOWED_NUDGE = "followed_nudge"
    AVOIDED_IMPULSIVE_TRADE = "avoided_impulsive_trade"
    JOURNALED_TRADE = "journaled_trade"

EVENT_POINTS: dict[EventType, int] = {
    EventType.FOLLOWED_NUDGE: 10,
    EventType.AVOIDED_IMPULSIVE_TRADE: 20,
    EventType.JOURNALED_TRADE: 5,
}

async def award_points(db: AsyncSession, user_id: int, event_type: EventType) -> int:
    """Compute and award points for a given event, update Redis + DB."""
    points = EVENT_POINTS.get(event_type, 0)
    if points == 0:
        return 0

    # 1) Update Redis leaderboard (ZINCRBY)
    await redis.zincrby(LEADERBOARD_KEY, points, str(user_id))

    # 2) Update persistent DB total_points
    result = await db.execute(select(DBUser).where(DBUser.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return 0

    user.total_points = (user.total_points or 0) + points
    await db.commit()
    return points

async def get_top_n(n: int = 10) -> List[dict]:
    """Return top N users from leaderboard with scores."""
    # ZREVRANGE for highest score first
    raw = await redis.zrevrange(LEADERBOARD_KEY, 0, n - 1, withscores=True)
    # raw: List[Tuple[member, score]]
    leaderboard = []
    for rank, (user_id_str, score) in enumerate(raw, start=1):
        leaderboard.append({
            "rank": rank,
            "user_id": int(user_id_str),
            "score": int(score),
        })
    return leaderboard

async def get_user_rank_and_score(user_id: int) -> dict:
    """Return a user's rank and score."""
    score = await redis.zscore(LEADERBOARD_KEY, str(user_id))
    if score is None:
        return {"user_id": user_id, "score": 0, "rank": None}

    # ZREVRANK gives index in descending order
    rank = await redis.zrevrank(LEADERBOARD_KEY, str(user_id))
    return {
        "user_id": user_id,
        "score": int(score),
        "rank": int(rank) + 1,  # convert 0-based to 1-based
    }
