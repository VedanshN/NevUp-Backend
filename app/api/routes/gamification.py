# app/api/routes/gamification.py
from fastapi import APIRouter, Depends
from app.api.deps import get_async_db, get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from app.gamification.leaderboard import (
    award_points, EventType, get_top_n, get_user_rank_and_score
)

router = APIRouter()

@router.post("/events/followed_nudge", tags=["gamification"])
async def event_followed_nudge(
    db: AsyncSession = Depends(get_async_db),
    user = Depends(get_current_user),
):
    points = await award_points(db, user.id, EventType.FOLLOWED_NUDGE)
    return {"message": "Points awarded for following nudge.", "points_awarded": points}

@router.post("/events/avoided_impulsive", tags=["gamification"])
async def event_avoided_impulsive(
    db: AsyncSession = Depends(get_async_db),
    user = Depends(get_current_user),
):
    points = await award_points(db, user.id, EventType.AVOIDED_IMPULSIVE_TRADE)
    return {"message": "Points awarded for avoiding impulsive trade.", "points_awarded": points}

@router.get("/leaderboard", tags=["gamification"])
async def get_leaderboard(
    n: int = 10,
):
    lb = await get_top_n(n)
    return {"leaderboard": lb}

@router.get("/me/points", tags=["gamification"])
async def get_my_points(
    user = Depends(get_current_user),
):
    data = await get_user_rank_and_score(user.id)
    return data
