import asyncio
from app.database import engine, Base
from app import models  # импортируем все модели, чтобы Base их увидел

async def init():
    async with engine.begin() as conn:
        # Удаляем все таблицы (осторожно! стирает данные)
        await conn.run_sync(Base.metadata.drop_all)
        # Создаём таблицы заново
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized")

if __name__ == "__main__":
    asyncio.run(init())