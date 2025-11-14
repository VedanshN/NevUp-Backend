
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class User(Base):
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    DateofBirth = Column(String, nullable=False)
    password = Column(String, nullable=False)
    onboarding_completed = Column(Boolean, default=False)

    # Relationship to Binance credentials
    binance_credentials = relationship("UserBinanceCredentials", back_populates="user", uselist=False)
    onboarding_data = relationship("Onboarding", back_populates="user", uselist=False)

class Onboarding(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user.id"), unique=True, nullable=False)
    trade_frequency = Column(String, nullable=False)
    trade_amount = Column(Integer, nullable=False)
    trade_type = Column(String, nullable=False)
    trade_time = Column(String, nullable=False)
    trade_profit = Column(Integer, nullable=False)

    user = relationship("User", back_populates="onboarding_data")

class nudges(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user.id"), unique=True, nullable=False)
    nudge_type = Column(String, nullable=False)
    nudge_message = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    user = relationship("User")

class UserBinanceCredentials(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user.id"), unique=True, nullable=False)
    api_key = Column(String, nullable=False) # In a real app, this should be encrypted
    secret_key = Column(String, nullable=False) # In a real app, this should be encrypted

    user = relationship("User", back_populates="binance_credentials")


# User portfolio data model
class UserPortfolio(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    asset_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    asset_value = Column(Integer, nullable=False)


class tradeHistory(Base):
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    trade_type = Column(String, nullable=False)
    asset_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    timestamp = Column(String, nullable=False)


