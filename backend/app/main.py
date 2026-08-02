from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    accuracy,
    features,
    games,
    health,
    live_props,
    live_slates,
    matchups,
    mlb_scoreboard,
    performance,
    players,
    predictions,
    props,
    slates,
    wnba_game_detail,
    wnba_leaders,
    wnba_odds,
    wnba_player,
    wnba_props,
    wnba_scoreboard,
    wnba_standings,
    wnba_futures,
)
from app.core.config import CORS_ORIGINS

app = FastAPI(
    title="HoopVista API",
    version="0.3.0",
    description=(
        "NBA prop prediction backend. Most endpoints read from Supabase "
        "(silver / gold / ml schemas) and make no NBA or Odds API calls. "
        "The exceptions are /api/wnba/scoreboard/today, /api/mlb/scoreboard/today, "
        "/api/wnba/leaders, /api/wnba/player/{player_id}, /api/wnba/standings, "
        "/api/wnba/futures, /api/wnba/odds/today, /api/wnba/props/today, and "
        "WNBA game detail routes, which call ESPN, stats.wnba.com, MLB Stats API, "
        "or SharpAPI for live league data."
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
app.include_router(live_props.router, prefix="/api")
app.include_router(live_slates.router, prefix="/api")
app.include_router(performance.router, prefix="/api")
app.include_router(accuracy.router, prefix="/api")
app.include_router(props.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(matchups.router, prefix="/api")
app.include_router(slates.router, prefix="/api")

# ── Direct upstream (non-DB) routes ────────────────────────────────────────
app.include_router(wnba_scoreboard.router, prefix="/api")
app.include_router(mlb_scoreboard.router, prefix="/api")
app.include_router(wnba_leaders.router, prefix="/api")
app.include_router(wnba_player.router, prefix="/api")
app.include_router(wnba_standings.router, prefix="/api")
app.include_router(wnba_futures.router, prefix="/api")
app.include_router(wnba_odds.router, prefix="/api")
app.include_router(wnba_props.router, prefix="/api")
app.include_router(wnba_game_detail.router, prefix="/api")
