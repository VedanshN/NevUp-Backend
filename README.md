# NevUp Backend

This is a FastAPI backend for the NevUp application, featuring user authentication, PostgreSQL database integration, Redis caching, and an API connection to Binance for market data and portfolio management.

## Table of Contents

- [Setup](#setup)
- [Running the Server](#running-the-server)
- [Database Migrations](#database-migrations)
- [API Endpoints](#api-endpoints)

## Setup

### Prerequisites

*   Docker and Docker Compose (for running the database, Redis, and the FastAPI application).

### 1. Clone the repository

```bash
git clone <repository_url>
cd NevUp-backend
```

### 2. Create and configure the `.env` file

Create a file named `.env` in the project root directory with the following content. **Replace placeholder values** with your actual secrets and desired credentials.

```env
# FastAPI Application Settings
PROJECT_NAME="NevUp Backend"
VERSION="1.0.0"

# Database Settings
DATABASE_URL="postgresql+asyncpg://user:password@db:5432/nevup_db"
REDIS_URL="redis://redis:6379/0"

# Authentication Settings
SECRET_KEY="your_super_secret_key_here"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# PostgreSQL Settings (These should match the user/password in DATABASE_URL)
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=nevup_db
```

**Important:** Make sure the `user` and `password` in `DATABASE_URL` match `POSTGRES_USER` and `POSTGRES_PASSWORD`.

### 3. (Optional) Create a Python Virtual Environment

While Docker Compose is used for running the application, you might want a virtual environment for local development tools (like running Alembic commands directly if not via `docker-compose run`).

```bash
python -m venv venv
./venv/Scripts/activate # On Windows
source venv/bin/activate # On macOS/Linux
pip install -r requirements.txt
```

## Running the Server

To run the entire stack (FastAPI, PostgreSQL, Redis) using Docker Compose:

1.  **Build and start the services:**
    ```bash
    docker-compose up --build
    ```
    The `--build` flag is important for the first run or after changing `requirements.txt` or `Dockerfile`.

2.  The FastAPI application will be accessible at `http://localhost:8000`.
3.  The interactive API documentation (Swagger UI) will be at `http://localhost:8000/docs`.

## Database Migrations (Using Alembic)

After making changes to your SQLAlchemy models (`app/db/models.py`), you need to generate and apply database migrations.

### 1. Rebuild the `web` service Docker image

This ensures Alembic and other dependencies are installed:

```bash
docker-compose build web
```

### 2. Initialize Alembic (if not already done)

This creates the `migrations` directory and configuration files. If the directory already exists, you can skip this.

```bash
docker-compose run --rm web alembic init -t async migrations
```

### 3. Configure `migrations/env.py`

Edit `migrations/env.py` to import your SQLAlchemy `Base` and set up the asynchronous engine. **Replace the entire content** with the version provided in the previous conversation (which correctly imports `from app.db.base import Base`).

### 4. Configure `alembic.ini`

Edit `alembic.ini` to set your database connection URL. Find the `sqlalchemy.url` line under the `[alembic]` section and update it to match your `DATABASE_URL` from the `.env` file.

```ini
sqlalchemy.url = postgresql+asyncpg://user:password@db:5432/nevup_db
```

### 5. Generate a new migration script

This command will detect changes in your models and create a migration file:

```bash
docker-compose run --rm web alembic revision --autogenerate -m "Descriptive message about your changes"
```

### 6. Review the generated migration script

Open the new Python file in `migrations/versions/` (e.g., `xxxxxxxxxxxx_descriptive_message.py`) and review the `upgrade()` and `downgrade()` functions to ensure they accurately reflect your intended database changes.

### 7. Apply the migration

Once reviewed, apply the changes to your database:

```bash
docker-compose run --rm web alembic upgrade head
```

## API Endpoints

Here are the main API endpoints provided by this backend:

### Authentication Endpoints (`/auth`)

*   **`POST /auth/signup`**
    *   **Description:** Register a new user.
    *   **Request Body:** `UserCreate` (username, email, DateofBirth, password)
    *   **Response:** `User` object (excluding password hash)
*   **`POST /auth/signin`**
    *   **Description:** Authenticate a user and return the user object.
    *   **Request Body:** form-data with `username` and `password`
    *   **Response:** `User` object
*   **`POST /auth/token`**
    *   **Description:** Authenticate a user and obtain an access token.
    *   **Request Body:** form-data with `username` and `password`
    *   **Response:** `Token` object (access_token, token_type)
*   **`GET /auth/me`**
    *   **Description:** Get the current authenticated user's profile.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `User` object
*   **`PUT /auth/me`**
    *   **Description:** Update the current authenticated user's profile.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `UserUpdate` (optional username, email, DateofBirth, password)
    *   **Response:** Updated `User` object
*   **`POST /auth/change-password`**
    *   **Description:** Change the current authenticated user's password.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `PasswordChange` (old_password, new_password)
    *   **Response:** `{"message": "Password updated successfully"}`

### Markets Endpoints (`/markets`)

*   **`POST /markets/binance-credentials`**
    *   **Description:** Connect a user's Binance portfolio by storing API key and secret.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `UserBinanceCredentialsCreate` (api_key, secret_key)
    *   **Response:** `UserBinanceCredentials` object (excluding secret_key)
*   **`PUT /markets/binance-credentials`**
    *   **Description:** Update a user's Binance API key and secret.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `UserBinanceCredentialsCreate` (api_key, secret_key)
    *   **Response:** Updated `UserBinanceCredentials` object (excluding secret_key)
*   **`DELETE /markets/binance-credentials`**
    *   **Description:** Deletes the current user's Binance account credentials.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `UserBinanceCredentialsResponse` object (id, user_id, api_key)
*   **`GET /markets/binance-credentials`**
    *   **Description:** Retrieves the current user's Binance account credentials (excluding the secret key).
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `UserBinanceCredentialsResponse` object (id, user_id, api_key)
*   **`GET /markets/time`**
    *   **Description:** Fetches the current server time from Binance using `ccxt`.
    *   **Response:** `{"serverTime": <timestamp>}`
*   **`GET /markets/exchange-info`**
    *   **Description:** Fetches exchange information, including symbols, filters, and trading rules using `ccxt`.
    *   **Response:** JSON object containing Binance exchange information.
*   **`GET /markets/balance`**
    *   **Description:** Fetches the current authenticated user's Binance account balance.
    *   **Requires:** Bearer Token in `Authorization` header and stored Binance API credentials.
    *   **Response:** JSON object containing the user's balance.
*   **`GET /markets/user-trades`**
    *   **Description:** Fetches the current user's personal Binance trade history.
    *   **Requires:** Bearer Token in `Authorization` header and stored Binance API credentials.
    *   **Query Parameters:** `symbol` (optional, e.g., `BTC/USDT`), `since` (optional, Unix timestamp in ms), `limit` (optional, max number of trades).
    *   **Response:** JSON array of trade objects.
*   **`GET /markets/user-deposits`**
    *   **Description:** Fetches the current user's Binance deposit history.
    *   **Requires:** Bearer Token in `Authorization` header and stored Binance API credentials.
    *   **Query Parameters:** `asset` (optional, e.g., `USDT`), `since` (optional, Unix timestamp in ms), `limit` (optional, max number of deposits).
    *   **Response:** JSON array of deposit objects.
*   **`GET /markets/user-withdrawals`**
    *   **Description:** Fetches the current user's Binance withdrawal history.
    *   **Requires:** Bearer Token in `Authorization` header and stored Binance API credentials.
    *   **Query Parameters:** `asset` (optional, e.g., `USDT`), `since` (optional, Unix timestamp in ms), `limit` (optional, max number of withdrawals).
    *   **Response:** JSON array of withdrawal objects.
*   **`GET /markets/asset-details/{asset_code}`**
    *   **Description:** Fetches detailed information for a specific asset from Binance, including network, fees, etc.
    *   **Requires:** Bearer Token in `Authorization` header and stored Binance API credentials.
    *   **Path Parameter:** `asset_code` (e.g., `BTC`).
    *   **Response:** JSON object containing asset details.
*   **`GET /markets/trades/{symbol}`**
    *   **Description:** Fetches recent trades for a specific symbol from Binance using `ccxt`.
    *   **Path Parameter:** `symbol` (e.g., `BTC/USDT`).
    *   **Response:** JSON array of trade objects.
*   **`GET /markets/tickers/{symbol}`**
    *   **Description:** Fetches ticker information for a specific symbol from Binance using `ccxt`.
    *   **Path Parameter:** `symbol` (e.g., `BTC/USDT`).
    *   **Response:** JSON object containing ticker information for the specified symbol.

### Machine Learning Endpoints (`/ml`)

*   **`GET /ml/classifierdata`**
    *   **Description:** Retrieves synthetic trade sequences data for the classifier model.
    *   **Response:** JSON array of trade data objects.
*   **`GET /ml/classifierusers`**
    *   **Description:** Returns a list of users for classifier related tasks.
    *   **Response:** `list[User]`
*   **`POST /ml/classifytrades`**
    *   **Description:** Classifies a sequence of trades into an archetype (e.g., "revenge", "fomo").
    *   **Request Body:** `list[TradeData]` (list of trade objects for one trading session)
    *   **Response:** `{"sequence_length": int, "prediction_id": int, "prediction_label": str}`
*   **`POST /ml/get_nudge`**
    *   **Description:** Accepts a list of trades (one sequence/session) and returns a single nudge chosen by the RL policy.
    *   **Request Body:** `list[TradeData]`
    *   **Response:** `{"nudge": str, "action_id": int, "planned_max_trades": int, "actual_trades": int, "baseline_excess_pnl": float, "message": str}`

### Nudge Endpoints (`/nudge`)

*   **`GET /nudge/nudgedata`**
    *   **Description:** Retrieves nudge data for the current authenticated user.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `Nudge` object

### Onboarding Endpoints (`/onboarding`)

*   **`POST /onboarding`**
    *   **Description:** Completes the onboarding process for the current authenticated user.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `OnboardingCreate` (trade_frequency, trade_amount, trade_type, trade_time, trade_profit)
    *   **Response:** `User` object (with `onboarding_completed` set to True)
*   **`PUT /onboarding`**
    *   **Description:** Updates the onboarding data for the current authenticated user.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Request Body:** `OnboardingCreate`
    *   **Response:** `User` object
*   **`GET /onboarding`**
    *   **Description:** Retrieves the onboarding data for the current authenticated user.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `Onboarding` object

### Gamification Endpoints (`/gamification`)

*   **`POST /gamification/events/followed_nudge`**
    *   **Description:** Awards points to the user for following a nudge.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `{"message": "Points awarded for following nudge.", "points_awarded": int}`
*   **`POST /gamification/events/avoided_impulsive`**
    *   **Description:** Awards points to the user for avoiding an impulsive trade.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `{"message": "Points awarded for avoiding impulsive trade.", "points_awarded": int}`
*   **`GET /gamification/leaderboard`**
    *   **Description:** Retrieves the top N users on the leaderboard.
    *   **Query Parameter:** `n` (optional, default: 10)
    *   **Response:** `{"leaderboard": list[dict]}`
*   **`GET /gamification/me/points`**
    *   **Description:** Retrieves the current authenticated user's rank and score on the leaderboard.
    *   **Requires:** Bearer Token in `Authorization` header.
    *   **Response:** `{"rank": int, "score": int}`
