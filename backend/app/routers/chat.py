from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.free_llm_client import FreeLLMBartender
from app.crud import confirm_order, create_order, save_dialogue_message
from app.database import get_db
from app.models import GeneratedCocktail, Ingredient
from app.websocket_manager import manager


router = APIRouter()

SESSION_STATE: Dict[str, Dict[str, Any]] = {}


def _new_state() -> Dict[str, Any]:
    return {
        "phase": "discovery",  # discovery -> draft_ready -> awaiting_rating
        "prefs": {
            "sweetness": 0.5,
            "sourness": 0.3,
            "fruitiness": 0.5,
            "citrus": 0.3,
            "mint": 0.1,
            "spice": 0.2,
            "base": "вода",
            "ice": True,
            "desired_terms": [],
            "avoid_terms": [],
        },
        "profile": {
            "mood": None,
            "avoid": [],
            "notes": [],
            "confirmed_terms": [],
            "requested_ingredients": [],
            "composition_examples": [],
        },
        "collected": {
            "taste": False,
            "mood": False,
            "base": False,
            "extras": False,
            "composition": False,
        },
        "asked_facets": [],
        "answered_facets": [],
        "recent_assistant_questions": [],
        "last_asked_facet": None,
        "questions_asked_count": 0,
        "draft": None,
        "pending_order_id": None,
        "pending_cocktail_id": None,
    }


def _remember_user_message(state: Dict[str, Any], user_msg: str) -> None:
    text = (user_msg or "").strip()
    if not text:
        return
    notes = state.get("profile", {}).setdefault("notes", [])
    if isinstance(notes, list):
        notes.append(f"user: {text}")
        state["profile"]["notes"] = notes[-40:]


def _remember_assistant_message(state: Dict[str, Any], assistant_msg: str) -> None:
    text = (assistant_msg or "").strip()
    if not text:
        return
    notes = state.get("profile", {}).setdefault("notes", [])
    if isinstance(notes, list):
        notes.append(f"assistant: {text}")
        state["profile"]["notes"] = notes[-40:]


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    await manager.connect(session_id, websocket)
    llm = FreeLLMBartender()
    
    # Инициализируем состояние, если его еще нет
    if session_id not in SESSION_STATE:
        SESSION_STATE[session_id] = _new_state()
    
    state_obj = SESSION_STATE[session_id]

    async def process_llm_response(user_msg: str, state_obj: Dict[str, Any]) -> None:
        try:
            raw_response = await asyncio.wait_for(
                llm.reply(
                    db=db,
                    user_message=user_msg,
                    state=state_obj,
                    fallback_text="Я немного задумался, повтори, пожалуйста.",
                    strict_json=True,
                ),
                timeout=60.0,
            )
            
            # Пытаемся найти JSON в ответе, если модель добавила лишний текст
            try:
                data = json.loads(raw_response)
            except:
                # Поиск JSON структуры внутри строки
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                    except:
                        # Если даже после извлечения не JSON - используем как текст (Fallback)
                        data = {"reply": raw_response, "action": "chat"}
                else:
                    # Если JSON вообще не найден - используем сырой ответ как текст (Fallback)
                    data = {"reply": raw_response, "action": "chat"}

            reply_text = data.get("reply", "")
            action = data.get("action", "chat")
            recipe_data = data.get("recipe")

            if action == "propose_recipe" and recipe_data:
                all_ings_res = await db.execute(select(Ingredient))
                all_ings = list(all_ings_res.scalars().all())
                by_name = {ing.name.lower(): ing for ing in all_ings}
                
                recipe_dict = {}
                for item in recipe_data.get("ingredients", []):
                    ing = by_name.get(item["name"].lower())
                    if ing:
                        recipe_dict[str(ing.id)] = float(item["qty"])
                
                if recipe_dict:
                    details = await llm.build_recipe_details(db, recipe_dict)
                    full_draft = {
                        "name": recipe_data.get("name", "Коктейль"),
                        "description": recipe_data.get("description", ""),
                        "recipe": recipe_dict,
                        "details": details,
                        "totals": llm.calculate_totals(details),
                        "taste_profile": llm.calculate_taste_profile(state_obj.get("prefs", {}), details)
                    }
                    state_obj["draft"] = full_draft
                    state_obj["phase"] = "draft_ready"
                    
                    # Отправляем ТОЛЬКО ОДНО сообщение типа recommendation.
                    # В message передаем весь текст от ИИ (с карточкой), а в draft - данные для фронтенда.
                    # Фронтенд увидит это как одно сообщение с кнопками/структурой.
                    await manager.send_message(session_id, json.dumps({
                        "type": "recommendation", 
                        "message": reply_text,
                        "draft": full_draft
                    }))
                    await save_dialogue_message(db, session_id, "assistant", f"{reply_text}\n(Рецепт: {full_draft['name']})")
                    return

            if action == "confirm_order" and state_obj.get("draft"):
                draft = state_obj["draft"]
                new_cocktail = GeneratedCocktail(
                    session_id=session_id,
                    name=draft["name"],
                    description=draft["description"],
                    recipe=draft["recipe"],
                )
                db.add(new_cocktail)
                await db.commit()
                await db.refresh(new_cocktail)
                order = await create_order(db, session_id, new_cocktail.id)
                success = await confirm_order(db, order.id, draft["recipe"])
                
                if success:
                    state_obj["phase"] = "awaiting_rating"
                    state_obj["pending_order_id"] = order.id
                    await manager.send_message(session_id, json.dumps({"type": "success", "content": reply_text}))
                    await save_dialogue_message(db, session_id, "assistant", reply_text)
                else:
                    await manager.send_message(session_id, json.dumps({"type": "error", "content": "Извини, пока мы обсуждали, какой-то ингредиент закончился. Давай поправим состав?"}))
                return

            await manager.send_message(session_id, json.dumps({"type": "assistant", "content": reply_text}))
            _remember_assistant_message(state_obj, reply_text)
            await save_dialogue_message(db, session_id, "assistant", reply_text)

        except Exception as e:
            print(f"Error in process_llm_response: {e}")
            await manager.send_message(session_id, json.dumps({"type": "assistant", "content": "Прости, я немного отвлекся. Что ты говорил?"}))

    # Приветствие только если сессия новая
    if not state_obj.get("profile", {}).get("notes"):
        await process_llm_response("Начни диалог как бармен.", state_obj)

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            user_msg = str(data.get("message", "")).strip()
            if not user_msg:
                continue

            await save_dialogue_message(db, session_id, "user", user_msg)
            _remember_user_message(state_obj, user_msg)
            
            if state_obj["phase"] == "awaiting_rating":
                rating_raw = user_msg.strip()
                # Пытаемся найти цифру в ответе
                rating_match = re.search(r'[1-5]', rating_raw)
                rating_val = int(rating_match.group()) if rating_match else None
                
                from app.models import Feedback
                feedback = Feedback(order_id=state_obj["pending_order_id"], rating=rating_val, comment=user_msg)
                db.add(feedback)
                await db.commit()
                
                state_obj["phase"] = "discovery"
                state_obj["draft"] = None
                
                # Прямой ответ без вызова LLM для этапа оценки, чтобы избежать ошибок "Я немного задумался"
                thanks_text = "Спасибо за вашу оценку! Рад был помочь. Если захотите попробовать что-то еще — я всегда здесь, у барной стойки!"
                await manager.send_message(session_id, json.dumps({"type": "assistant", "content": thanks_text}))
                await save_dialogue_message(db, session_id, "assistant", thanks_text)
                continue

            await process_llm_response(user_msg, state_obj)

    except WebSocketDisconnect:
        pass
    finally:
        # Не удаляем сессию сразу, чтобы она сохранялась при переподключении
        # SESSION_STATE.pop(session_id, None) 
        manager.disconnect(session_id)
