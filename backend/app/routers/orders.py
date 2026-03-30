from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Order


router = APIRouter(prefix="/orders", tags=["orders"])


@router.get("/")
async def list_orders(db: AsyncSession = Depends(get_db)):
    orders = (await db.execute(select(Order))).scalars().all()
    return {
        "orders": [
            {
                "id": o.id,
                "session_id": o.session_id,
                "cocktail_id": o.cocktail_id,
                "status": o.status,
                "confirmed_at": o.confirmed_at.isoformat() if o.confirmed_at else None,
            }
            for o in orders
        ]
    }

