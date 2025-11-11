'''anter the onboarding questions for a user'''
from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.deps import get_current_user, get_async_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password, get_password_hash
from app.db.models import User as DBUser, Onboarding as DBOnboarding
from app.schemas import UserCreate, User, Token, PasswordChange, UserUpdate, OnboardingCreate

router = APIRouter()

@router.post("/onboarding", response_model=User, tags=["onboarding"])
async def complete_onboarding(
    onboarding_data: OnboardingCreate,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    if current_user.onboarding_completed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Onboarding already completed")

    # Check if onboarding data already exists for the user
    existing_onboarding = await db.execute(
        select(DBOnboarding).where(DBOnboarding.user_id == current_user.id)
    )
    if existing_onboarding.scalars().first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Onboarding data already exists. Use PUT to update.")

    # Create new onboarding entry
    db_onboarding = DBOnboarding(
        user_id=current_user.id,
        trade_frequency=onboarding_data.trade_frequency,
        trade_amount=onboarding_data.trade_amount,
        trade_type=onboarding_data.trade_type,
        trade_time=onboarding_data.trade_time,
        trade_profit=onboarding_data.trade_profit
    )
    db.add(db_onboarding)
    
    # Update user's onboarding status
    current_user.onboarding_completed = True
    db.add(current_user)

    await db.commit()
    await db.refresh(db_onboarding)
    await db.refresh(current_user)

    return current_user

@router.put("/onboarding", response_model=User, tags=["onboarding"])
async def update_onboarding(
    onboarding_data: OnboardingCreate,
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Onboarding not yet completed. Use POST to create.")

    existing_onboarding = await db.execute(
        select(DBOnboarding).where(DBOnboarding.user_id == current_user.id)
    )
    db_onboarding = existing_onboarding.scalars().first()

    if not db_onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Onboarding data not found.")

    db_onboarding.trade_frequency = onboarding_data.trade_frequency
    db_onboarding.trade_amount = onboarding_data.trade_amount
    db_onboarding.trade_type = onboarding_data.trade_type
    db_onboarding.trade_time = onboarding_data.trade_time
    db_onboarding.trade_profit = onboarding_data.trade_profit
    
    db.add(db_onboarding)
    await db.commit()
    await db.refresh(db_onboarding)

    return current_user

@router.get("/onboarding", response_model=Onboarding, tags=["onboarding"])
async def get_onboarding_data(
    current_user: DBUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    if not current_user.onboarding_completed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Onboarding not yet completed.")
    
    onboarding_data = await db.execute(
        select(DBOnboarding).where(DBOnboarding.user_id == current_user.id)
    )
    db_onboarding = onboarding_data.scalars().first()

    if not db_onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Onboarding data not found.")

    return db_onboarding