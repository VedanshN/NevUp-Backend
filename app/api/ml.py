from api.onboarding import get_onboarding_data
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api.deps import get_current_user, get_async_db
from app.db.models import User as DBUser, Onboarding as DBOnboarding
from app.schemas import OnboardingCreate, User
import torch
import torch.nn as nn


router = APIRouter()

@router.get("/classifierdata", response_model=OnboardingCreate, tags=["ml"])
async def get_classifier_data(
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    onboarding_data = await get_onboarding_data(current_user.id, db)
    if not onboarding_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Onboarding data not found for the user.")
    return onboarding_data   


@router.get("/classifierusers", response_model=list[User], tags=["ml"])
async def classifier_users():
    return None
    
