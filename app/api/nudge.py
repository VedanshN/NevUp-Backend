from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_current_user, get_async_db
from app.db.models import User as DBUser
from app.schemas import NudgeCreate
from app.api.ml import modeloutput
from sqlalchemy.future import select
from app.db.models import Nudge as DBNudge
from app.schemas import User
router = APIRouter()

@router.get("/nudgedata", response_model=NudgeCreate, tags=["nudge"])
async def nudgedata(
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    result = await db.execute(
        select(DBNudge).where(DBNudge.user_id == current_user.id)
    )
    nudge_data = result.scalars().first()
    if not nudge_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nudge data not found for the user.")
    return nudge_data