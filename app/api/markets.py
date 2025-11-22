'''Under Construction: FastAPI routes for interacting with Binance exchange using ccxt.'''

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import ccxt.async_support as ccxt  # Use async version of ccxt
from typing import Optional

from app.api.deps import get_current_user, get_async_db
from app.db.models import UserBinanceCredentials, User
from app.schemas import UserBinanceCredentialsCreate, UserBinanceCredentialsInDBBase as UserBinanceCredentialsSchema, UserBinanceCredentialsResponse

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
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
        }
    })
    return exchange


@router.post("/binance-credentials", response_model=UserBinanceCredentialsResponse, tags=["markets"])
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


@router.put("/binance-credentials", response_model=UserBinanceCredentialsResponse, tags=["markets"])
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


@router.delete("/binance-credentials", status_code=status.HTTP_204_NO_CONTENT, tags=["markets"])
async def delete_binance_credentials(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Deletes the current user's Binance account credentials."""
    result = await db.execute(select(UserBinanceCredentials).where(UserBinanceCredentials.user_id == current_user.id))
    credentials = result.scalars().first()

    if not credentials:
        raise HTTPException(status_code=404, detail="Binance credentials not found for this user.")

    await db.delete(credentials)
    await db.commit()
    return UserBinanceCredentialsResponse(id=credentials.id, user_id=credentials.user_id, api_key=credentials.api_key)


@router.get("/binance-credentials", response_model=UserBinanceCredentialsResponse, tags=["markets"])
async def get_binance_credentials(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Retrieves the current user's Binance account credentials (excluding the secret key)."""
    result = await db.execute(select(UserBinanceCredentials).where(UserBinanceCredentials.user_id == current_user.id))
    credentials = result.scalars().first()

    if not credentials:
        raise HTTPException(status_code=404, detail="Binance credentials not found for this user.")
    
    # Return a schema that excludes the secret key
    return UserBinanceCredentialsResponse(id=credentials.id, user_id=credentials.user_id, api_key=credentials.api_key)


@router.get("/time", tags=["markets"])
async def get_binance_server_time():
    """Fetches the current server time from Binance using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        time = await exchange.fetch_time()
        return {"serverTime": time}
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance server time: {str(e)}")
    finally:
        await exchange.close()


@router.get("/exchange-info", tags=["markets"])
async def get_binance_exchange_info():
    """Fetches exchange information, including symbols, filters, and trading rules using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        market_info = await exchange.fetch_markets()
        return market_info
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance exchange information: {str(e)}")
    finally:
        await exchange.close()


@router.get("/balance", tags=["markets"])
async def get_user_binance_balance(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Fetches the current user's Binance account balance."""
    exchange = await get_exchange(current_user.id, db)
    try:
        balance = await exchange.fetch_balance()
        # Manually process raw balances to ensure small amounts are included
        processed_balances = {
            'info': balance['info'],
            'free': {},
            'used': {},
            'total': {}
        }

        for item in balance['info']['balances']:
            asset = item['asset']
            free = float(item['free'])
            locked = float(item['locked'])
            
            if free > 0 or locked > 0:
                processed_balances['free'][asset] = free
                processed_balances['used'][asset] = locked
                processed_balances['total'][asset] = free + locked
        

        for key, value in balance.items():
            if key not in ['info', 'free', 'used', 'total']:
                processed_balances[key] = value

        return processed_balances
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance balance: {str(e)}")
    finally:
        await exchange.close()


@router.get("/user-trades", tags=["markets"])
async def get_user_binance_trade_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    symbol: Optional[str] = Query(None, description="Filter trades by symbol (e.g., 'BTC/USDT')"),
    since: Optional[int] = Query(None, description="Return trades since this datetime (Unix timestamp in ms)"),
    limit: Optional[int] = Query(None, description="Maximum number of trades to retrieve")
):
    """Fetches the current user's personal Binance trade history."""
    exchange = await get_exchange(current_user.id, db)
    try:
        params = {}
        if symbol:
            symbol = symbol.upper() # Ensure symbol is uppercase
        trades = await exchange.fetch_my_trades(symbol=symbol, since=since, limit=limit, params=params)
        return trades
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user Binance trade history: {str(e)}")
    finally:
        await exchange.close()


@router.get("/user-deposits", tags=["markets"])
async def get_user_binance_deposit_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    asset: Optional[str] = Query(None, description="Filter deposits by asset (e.g., 'USDT')"),
    since: Optional[int] = Query(None, description="Return deposits since this datetime (Unix timestamp in ms)"),
    limit: Optional[int] = Query(None, description="Maximum number of deposits to retrieve")
):
    """Fetches the current user's Binance deposit history."""
    exchange = await get_exchange(current_user.id, db)
    try:
        deposits = await exchange.fetch_deposits(code=asset, since=since, limit=limit)
        return deposits
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user Binance deposit history: {str(e)}")
    finally:
        await exchange.close()


@router.get("/user-withdrawals", tags=["markets"])
async def get_user_binance_withdrawal_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    asset: Optional[str] = Query(None, description="Filter withdrawals by asset (e.g., 'USDT')"),
    since: Optional[int] = Query(None, description="Return withdrawals since this datetime (Unix timestamp in ms)"),
    limit: Optional[int] = Query(None, description="Maximum number of withdrawals to retrieve")
):
    """Fetches the current user's Binance withdrawal history."""
    exchange = await get_exchange(current_user.id, db)
    try:
        withdrawals = await exchange.fetch_withdrawals(code=asset, since=since, limit=limit)
        return withdrawals
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user Binance withdrawal history: {str(e)}")
    finally:
        await exchange.close()


@router.get("/asset-details/{asset_code}", tags=["markets"])
async def get_binance_asset_details(
    asset_code: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Fetches detailed information for a specific asset from Binance, including network, fees, etc."""
    exchange = await get_exchange(current_user.id, db)
    try:
        # Fetch all currency details and then filter for the specific asset
        currencies = await exchange.fetch_currencies()
        if asset_code.upper() in currencies:
            return currencies[asset_code.upper()]
        else:
            raise HTTPException(status_code=404, detail=f"Asset '{asset_code}' not found.")
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}. Please check your API keys.")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance asset details: {str(e)}")
    finally:
        await exchange.close()


@router.get("/trades/{symbol:path}", tags=["markets"])
async def get_binance_trades(symbol: str):
    """Fetches recent trades for a specific symbol from Binance using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        trades = await exchange.fetch_trades(symbol.upper())
        if trades is None:
            trades = []
        return trades
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance trades: {str(e)}")
    finally:
        await exchange.close()


@router.get("/tickers/{symbol:path}", tags=["markets"])
async def get_binance_ticker(symbol: str):
    """Fetches ticker information for a specific symbol from Binance using ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        ticker = await exchange.fetch_ticker(symbol.upper())
        return ticker
    except ccxt.NetworkError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise HTTPException(status_code=500, detail=f"Exchange error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Binance ticker: {str(e)}")
    finally:
        await exchange.close()

