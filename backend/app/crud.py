from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DialogueMessage, GeneratedCocktail, Order, Feedback, Ingredient


async def save_dialogue_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
) -> None:
    msg = DialogueMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    await db.commit()


async def create_order(db: AsyncSession, session_id: str, cocktail_id: int) -> Order:
    order = Order(session_id=session_id, cocktail_id=cocktail_id, status="pending")
    db.add(order)
    await db.commit()
    await db.refresh(order)
    return order


async def confirm_order(
    db: AsyncSession,
    order_id: int,
    recipe: Dict[str, float],
) -> bool:
    """
    Атомарно списывает ингредиенты и помечает заказ как подтверждённый.

    recipe: {ingredient_id: qty} (ключи из JSON часто приходят строками)
    """
    try:
        # Важно: сессия может уже быть в транзакции (после предыдущих SELECT),
        # поэтому не используем db.begin(), чтобы не ловить "transaction already begun".
        # Блокируем заказ
        order_res = await db.execute(select(Order).where(Order.id == order_id).with_for_update())
        order = order_res.scalar_one_or_none()
        if not order:
            return False
        if order.status == "confirmed":
            return True

        # Превращаем recipe keys в int
        normalized_recipe: Dict[int, float] = {int(k): float(v) for k, v in recipe.items()}
        if not normalized_recipe:
            return False

        ingredient_ids = list(normalized_recipe.keys())
        ing_res = await db.execute(
            select(Ingredient).where(Ingredient.id.in_(ingredient_ids)).with_for_update()
        )
        ing_map: Dict[int, Ingredient] = {ing.id: ing for ing in ing_res.scalars().all()}

        # Проверяем остатки
        for ing_id, needed_qty in normalized_recipe.items():
            ing = ing_map.get(ing_id)
            if not ing:
                return False
            if float(ing.quantity) < float(needed_qty):
                return False

        # Списываем
        for ing_id, needed_qty in normalized_recipe.items():
            ing = ing_map[ing_id]
            ing.quantity = float(ing.quantity) - float(needed_qty)

        # Обновляем заказ
        order.status = "confirmed"
        order.confirmed_at = datetime.now(timezone.utc)
        await db.commit()
        return True
    except SQLAlchemyError:
        # Ошибка транзакции — значит подтверждение не удалось.
        await db.rollback()
        return False

