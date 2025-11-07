'''Under Construction: FastAPI routes for interacting with Binance exchange using ccxt.'''





from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import ccxt.async_support as ccxt  # Use async version of ccxt

from app.api.deps import get_current_user, get_async_db
from app.db.models import UserBinanceCredentials, User
from app.schemas import UserBinanceCredentialsCreate, UserBinanceCredentials as UserBinanceCredentialsSchema

router = APIRouter()

async def get_exchange(user_id: str, db: AsyncSession) -> ccxt.binance:
    result = await db.execute(select(UserBinanceCredentials).where(UserBinanceCredentials.user_id == user_id))
    credentials = result.scalars().first()

    if not credentials:
        raise HTTPException(status_code=404, detail="Binance credentials not found for this user.")

    exchange = ccxt.binance({
        'apiKey': credentials.api_key,
        'secret': credentials.secret_key,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future', # or 'spot' or 'margin'
        }
    })
    return exchange


@router.post("/binance-credentials", response_model=UserBinanceCredentialsSchema, tags=["markets"])
async def create_binance_credentials(
    *,
    db: AsyncSession = Depends(get_async_db),
    credentials_in: UserBinanceCredentialsCreate,
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(select(UserBinanceCredentials).where(UserBinanceCredentials.user_id == current_user.id))
    existing_credentials = result.scalars().first()

    if existing_credentials:
        raise HTTPException(status_code=400, detail="Binance credentials already exist for this user. Use PUT to update.")

    db_credentials = UserBinanceCredentials(
        user_id=current_user.id,
        api_key=credentials_in.api_key,
        secret_key=credentials_in.secret_key
    )
    db.add(db_credentials)
    await db.commit()
    await db.refresh(db_credentials)
    return db_credentials


@router.put("/binance-credentials", response_model=UserBinanceCredentialsSchema, tags=["markets"])
async def update_binance_credentials(
    *,
    db: AsyncSession = Depends(get_async_db),
    credentials_in: UserBinanceCredentialsCreate,
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(select(UserBinanceCredentials).where(UserBinanceCredentials.user_id == current_user.id))
    existing_credentials = result.scalars().first()

    if not existing_credentials:
        raise HTTPException(status_code=404, detail="Binance credentials not found for this user. Use POST to create.")

    existing_credentials.api_key = credentials_in.api_key
    existing_credentials.secret_key = credentials_in.secret_key

    db.add(existing_credentials)
    await db.commit()
    await db.refresh(existing_credentials)
    return existing_credentials


@router.get("/time", tags=["markets"])
async def get_binance_server_time():
    """Fetches the current server time from Binance using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        time = await exchange.fetch_time()
        await exchange.close()
        return {"serverTime": time}
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/exchange-info", tags=["markets"])
async def get_binance_exchange_info():
    """Fetches exchange information, including symbols, filters, and trading rules using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        market_info = await exchange.fetch_markets()
        await exchange.close()
        return market_info
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/balance", tags=["markets"])
async def get_user_binance_balance(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Fetches the current user's Binance account balance."""
    exchange = await get_exchange(current_user.id, db)
    try:
        balance = await exchange.fetch_balance()
        await exchange.close()
        return balance
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/tickers/{symbol:path}", tags=["markets"])
async def get_binance_ticker(symbol: str):
    """Fetches ticker information for a specific symbol from Binance using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        ticker = await exchange.fetch_ticker(symbol.upper())
        await exchange.close()
        return ticker
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
