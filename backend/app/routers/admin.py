from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Feedback, Ingredient, Order, GeneratedCocktail


router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def admin_home(request: Request, db: AsyncSession = Depends(get_db)):
    ingredients = (await db.execute(select(Ingredient))).scalars().all()
    # Загружаем заказы со статусом 'confirmed' и связанные коктейли для отображения состава
    orders_query = select(Order).where(Order.status == 'confirmed').order_by(Order.confirmed_at.desc())
    active_orders = (await db.execute(orders_query)).scalars().all()
    
    # Также загружаем все заказы для общей таблицы (как было)
    all_orders = (await db.execute(select(Order).order_by(Order.created_at.desc()))).scalars().all()
    
    feedbacks = (await db.execute(select(Feedback))).scalars().all()
    
    # Подгружаем данные коктейлей для активных заказов
    active_orders_data = []
    for order in active_orders:
        cocktail = await db.get(GeneratedCocktail, order.cocktail_id)
        recipe_display = []
        if cocktail and cocktail.recipe:
            # Преобразуем ID ингредиентов в названия
            for ing_id_str, qty in cocktail.recipe.items():
                ing = await db.get(Ingredient, int(ing_id_str))
                if ing:
                    recipe_display.append(f"{ing.name}: {qty} {ing.unit}")
        
        active_orders_data.append({
            "order": order,
            "cocktail": cocktail,
            "recipe_display": recipe_display
        })

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "ingredients": ingredients,
            "active_orders": active_orders_data,
            "orders": all_orders,
            "feedbacks": feedbacks,
        },
    )


@router.post("/order/{order_id}/complete")
async def complete_order(order_id: int, db: AsyncSession = Depends(get_db)):
    order = await db.get(Order, order_id)
    if order:
        order.status = "completed"
        await db.commit()
    return RedirectResponse(url="/admin/", status_code=303)


@router.post("/order/{order_id}/cancel")
async def cancel_order(order_id: int, db: AsyncSession = Depends(get_db)):
    order = await db.get(Order, order_id)
    if order:
        order.status = "cancelled"
        await db.commit()
    return RedirectResponse(url="/admin/", status_code=303)


@router.post("/ingredient/add")
async def add_ingredient(
    request: Request,
    name: str = Form(...),
    category: str = Form(...),
    unit: str = Form("ml"),
    quantity: float = Form(...),
    db: AsyncSession = Depends(get_db),
):
    # Важно: Ingredient.name уникален, поэтому если такое имя уже есть — это приведёт к ошибке.
    # Для дипломного прототипа оставим явное поведение; можно будет расширить позже (upsert).
    ing = Ingredient(name=name, category=category, unit=unit, quantity=quantity)
    db.add(ing)
    await db.commit()
    return RedirectResponse(url="/admin/", status_code=303)


@router.post("/ingredient/update/{ing_id}")
async def update_ingredient(
    ing_id: int,
    quantity: float = Form(...),
    db: AsyncSession = Depends(get_db),
):
    ing = await db.get(Ingredient, ing_id)
    if ing:
        ing.quantity = quantity
        await db.commit()
    return RedirectResponse(url="/admin/", status_code=303)

