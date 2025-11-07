
from typing import Optional
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    username: str
    email: EmailStr
    DateofBirth: str


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    DateofBirth: Optional[str] = None
    password: Optional[str] = None


class UserInDBBase(UserBase):
    id: Optional[str] = None

    class Config:
        from_attributes = True


class User(UserInDBBase):
    pass


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[str] = None


class PasswordChange(BaseModel):
    old_password: str
    new_password: str

# UserPortfolio Schemas
class UserPortfolioBase(BaseModel):
    user_id: str
    asset_name: str
    quantity: int
    asset_value: int


class UserPortfolioCreate(UserPortfolioBase):
    pass


class UserPortfolioUpdate(UserPortfolioBase):
    user_id: Optional[str] = None
    asset_name: Optional[str] = None
    quantity: Optional[int] = None
    asset_value: Optional[int] = None


class UserPortfolioInDBBase(UserPortfolioBase):
    id: Optional[int] = None

    class Config:
        from_attributes = True


class UserPortfolio(UserPortfolioInDBBase):
    pass


# Binance Credentials Schemas
class UserBinanceCredentialsBase(BaseModel):
    api_key: str


class UserBinanceCredentialsCreate(UserBinanceCredentialsBase):
    secret_key: str


class UserBinanceCredentialsInDBBase(UserBinanceCredentialsBase):
    id: Optional[int] = None
    user_id: str

    class Config:
        from_attributes = True


class UserBinanceCredentials(UserBinanceCredentialsInDBBase):
    pass

