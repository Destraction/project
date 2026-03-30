from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Feedback, Ingredient, Order


router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def admin_home(request: Request, db: AsyncSession = Depends(get_db)):
    ingredients = (await db.execute(select(Ingredient))).scalars().all()
    orders = (await db.execute(select(Order))).scalars().all()
    feedbacks = (await db.execute(select(Feedback))).scalars().all()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "ingredients": ingredients,
            "orders": orders,
            "feedbacks": feedbacks,
        },
    )


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

