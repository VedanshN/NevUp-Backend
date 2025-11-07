
from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.deps import get_current_user, get_async_db
from app.core.config import settings
from app.core.security import create_access_token, verify_password, get_password_hash
from app.db.models import User as DBUser # Renamed to DBUser to avoid conflict
from app.schemas import UserCreate, User, Token, PasswordChange, UserUpdate

router = APIRouter()


@router.post("/signup", response_model=User, tags=["auth"])
async def signup(*, db: AsyncSession = Depends(get_async_db), user_in: UserCreate):
    result = await db.execute(select(DBUser).where(DBUser.email == user_in.email)) # Use DBUser
    user = result.scalars().first()
    if user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    result = await db.execute(select(DBUser).where(DBUser.username == user_in.username)) # Use DBUser
    user = result.scalars().first()
    if user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
    
    hashed_password = get_password_hash(user_in.password)
    user_id = str(uuid4())
    db_user = DBUser(id=user_id, username=user_in.username, email=user_in.email, DateofBirth=user_in.DateofBirth, password=hashed_password) # Use DBUser
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user # Return DBUser, Pydantic will convert it to User schema


@router.post("/signin", response_model=User, tags=["auth"])
async def authenticate_user(db: AsyncSession = Depends(get_async_db), username: str = Form(...), password: str = Form(...)):
    result = await db.execute(select(DBUser).where(DBUser.username == username)) # Use DBUser
    db_user = result.scalars().first()
    if not db_user or not verify_password(password, db_user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    return db_user # Return DBUser


@router.post("/token", response_model=Token, tags=["auth"])
async def login_for_access_token(db: AsyncSession = Depends(get_async_db), form_data: OAuth2PasswordRequestForm = Depends()):
    result = await db.execute(select(DBUser).where(DBUser.username == form_data.username)) # Use DBUser
    db_user = result.scalars().first()
    if not db_user or not verify_password(form_data.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=str(db_user.id), expires_delta=access_token_expires # Use db_user.id
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User, tags=["auth"])
async def read_users_me(current_user: DBUser = Depends(get_current_user)): # current_user is now DBUser
    return current_user


@router.put("/me", response_model=User, tags=["auth"])
async def update_user_me(
    *,
    db: AsyncSession = Depends(get_async_db),
    user_update: UserUpdate,
    current_user: DBUser = Depends(get_current_user), # current_user is now DBUser
):
    if user_update.username is not None and user_update.username != current_user.username:
        result = await db.execute(select(DBUser).where(DBUser.username == user_update.username)) # Use DBUser
        existing_user = result.scalars().first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    if user_update.email is not None and user_update.email != current_user.email:
        result = await db.execute(select(DBUser).where(DBUser.email == user_update.email)) # Use DBUser
        existing_user = result.scalars().first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

    current_user.username = user_update.username if user_update.username is not None else current_user.username
    current_user.email = user_update.email if user_update.email is not None else current_user.email
    current_user.DateofBirth = user_update.DateofBirth if user_update.DateofBirth is not None else current_user.DateofBirth
    if user_update.password:
        current_user.password = get_password_hash(user_update.password)

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.post("/change-password", tags=["auth"])
async def change_password(
    *,
    db: AsyncSession = Depends(get_async_db),
    password_change: PasswordChange,
    current_user: DBUser = Depends(get_current_user) # current_user is now DBUser
):
    if not verify_password(password_change.old_password, current_user.password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    
    current_user.password = get_password_hash(password_change.new_password)
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return {"message": "Password updated successfully"}


