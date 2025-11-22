
from fastapi import FastAPI
from app.api import auth
from app.api import markets
from app.db.redis import close_redis_client, get_redis_client
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Added for static files

app = FastAPI()

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8000", # Allow requests from the FastAPI server itself
    "http://localhost:8080", # Example frontend development server
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    # Add other origins as needed, e.g., your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth")
app.include_router(markets.router, prefix="/markets")

@app.on_event("startup")
async def startup_event():
    await get_redis_client()

@app.on_event("shutdown")
async def shutdown_event():
    await close_redis_client()

