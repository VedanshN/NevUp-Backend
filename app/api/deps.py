
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal, get_async_db
from app.db.models import User as DBUser # Renamed to DBUser to avoid conflict
from app.schemas import TokenPayload

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def get_db() -> Generator:
    try:
        db = AsyncSessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
    db: AsyncSession = Depends(get_async_db),
    token: str = Depends(oauth2_scheme)
) -> DBUser: # Return type is now DBUser
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await db.execute(select(DBUser).where(DBUser.id == token_data.sub)) # Use DBUser
    user = user.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
