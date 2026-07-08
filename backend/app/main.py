from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import accuracy, features, games, health, matchups, players, predictions, props, slates
from app.core.config import CORS_ORIGINS

app = FastAPI(
    title="HoopVista API",
    version="0.3.0",
    description=(
        "NBA prop prediction backend. All endpoints read from Supabase "
        "(silver / gold / ml schemas). No NBA or Odds API calls are made here."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core Phase 9 routes ────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(games.router, prefix="/api")

# ── Additional DB-backed routes ────────────────────────────────────────────
app.include_router(accuracy.router, prefix="/api")
app.include_router(props.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(matchups.router, prefix="/api")
app.include_router(slates.router, prefix="/api")
