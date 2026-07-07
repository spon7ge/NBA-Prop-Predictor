from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import games, health, players, predictions, slates
from app.core.config import CORS_ORIGINS

app = FastAPI(
    title="HoopVista API",
    version="0.2.0",
    description=(
        "NBA prop prediction backend. "
        "All endpoints read from Supabase (gold/silver dbt tables). "
        "No NBA or Odds API calls are made here."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Existing routes ────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(slates.router, prefix="/api")

# ── New DB-backed routes ───────────────────────────────────────────────────
app.include_router(predictions.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(games.router, prefix="/api")
