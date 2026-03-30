from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import select

from app.database import Base, engine
from app.models import Compatibility, Ingredient
from app.routers import admin, chat, orders


app = FastAPI(title="Cocktail AI", version="0.1.0")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "ws_endpoint": "/ws/{session_id}",
        "admin_endpoint": "/admin/",
    }


app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(orders.router)


@app.on_event("startup")
async def on_startup() -> None:
    # 1) Создаём таблицы (если их ещё нет)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2) Сидим тестовые данные (иногда нужно, чтобы прототип сразу заработал)
    async with engine.begin() as conn:
        ing_exists = (await conn.execute(select(Ingredient.id).limit(1))).first() is not None
        comp_exists = (await conn.execute(select(Compatibility.ing1_id).limit(1))).first() is not None

    if not ing_exists or not comp_exists:
        # seed_data использует AsyncSessionLocal и модели.
        from seed_data import seed as seed_fn  # импорт по месту, чтобы не создавать цикл

        await seed_fn()

