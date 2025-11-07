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
    *   **Description:** Authenticate a user and return the user object (your added endpoint).
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
*   **`GET /markets/tickers/{symbol}`**
    *   **Description:** Fetches ticker information for a specific symbol (e.g., BTC/USDT) from Binance using `ccxt`.
    *   **Path Parameter:** `symbol` (e.g., `BTC/USDT`)
    *   **Response:** JSON object containing ticker information for the specified symbol.
