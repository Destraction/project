from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversation_agent import RussianBartenderAgent
from app.cocktail_generator import CocktailGenerator
from app.crud import confirm_order, create_order, save_dialogue_message
from app.database import get_db
from app.models import GeneratedCocktail
from app.nlp_parser import PreferenceParser
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
        },
        "profile": {
            "mood": None,
            "avoid": [],
            "notes": [],
        },
        "collected": {
            "taste": False,
            "mood": False,
            "base": False,
            "extras": False,
        },
        "draft": None,
        "pending_order_id": None,
        "pending_cocktail_id": None,
    }


def _merge_prefs(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key in ("sweetness", "sourness", "fruitiness", "citrus", "mint", "spice"):
        if key in src:
            dst[key] = max(0.0, min(1.0, float(src[key])))
    if src.get("base"):
        dst["base"] = src["base"]
    if "desired_terms" in src:
        old = dst.get("desired_terms", [])
        dst["desired_terms"] = list(dict.fromkeys([*old, *src["desired_terms"]]))[-30:]
    if "avoid_terms" in src:
        old = dst.get("avoid_terms", [])
        dst["avoid_terms"] = list(dict.fromkeys([*old, *src["avoid_terms"]]))[-30:]


def _parse_meta_preferences(text: str, state: Dict[str, Any]) -> None:
    t = text.lower()
    profile = state["profile"]
    prefs = state["prefs"]

    # Настроение
    mood_map = {
        "спокой": "relaxed",
        "бодр": "energized",
        "романт": "romantic",
        "вечерин": "party",
        "освеж": "fresh",
    }
    for k, mood in mood_map.items():
        if k in t:
            profile["mood"] = mood
            state["collected"]["mood"] = True
            break

    # Ограничения / исключения
    if "без " in t:
        # простое извлечение: всё после "без "
        idx = t.find("без ")
        if idx >= 0:
            chunk = t[idx + 4:].strip()
            if chunk:
                profile["avoid"].append(chunk)
                prefs.setdefault("avoid_terms", [])
                prefs["avoid_terms"].append(chunk.split(" ")[0])

    if "без льда" in t:
        prefs["ice"] = False
        state["collected"]["extras"] = True
    if "со льдом" in t or "с льдом" in t:
        prefs["ice"] = True
        state["collected"]["extras"] = True

    if "газирован" in t or "сода" in t:
        prefs["base"] = "газированная вода"
        state["collected"]["base"] = True
    if "негазир" in t:
        prefs["base"] = "вода"
        state["collected"]["base"] = True

    # Маркеры что вкусовые предпочтения указаны
    if any(x in t for x in ["слад", "кисл", "мят", "цитрус", "фрукт", "прян", "лайм", "лимон"]):
        state["collected"]["taste"] = True


def _need_more_questions(state: Dict[str, Any]) -> bool:
    c = state["collected"]
    return not (c["taste"] and c["mood"] and c["base"])


def _next_question(state: Dict[str, Any]) -> str:
    c = state["collected"]
    if not c["mood"]:
        return "Какое у вас настроение для напитка: спокойное, бодрящее, романтичное или для вечеринки?"
    if not c["base"]:
        return "Предпочитаете основу на воде или на газированной воде?"
    if not c["taste"]:
        return "По вкусу что ближе: сладкий, кислый, более фруктовый, цитрусовый, мятный или пряный профиль?"
    return "Хотите добавить ограничения: например, 'без имбиря', 'без льда', 'меньше сладости'?"


def _is_start_generation_message(text: str) -> bool:
    t = text.lower()
    triggers = ["предлож", "собери", "готово", "давай вариант", "сделай коктейль", "подбери"]
    return any(x in t for x in triggers)


def _is_confirm_message(text: str) -> bool:
    t = text.lower().strip()
    return t in {"да", "подтвердить", "ок", "подтверждаю", "оформить", "принимаю"}


def _is_adjustment_message(text: str) -> bool:
    t = text.lower().strip()
    markers = ["добавь", "убери", "удали", "слаще", "менее слад", "меньше слад", "объем", "объём", "без льда", "добавь лёд", "добавь лед"]
    return any(m in t for m in markers)


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    await manager.connect(session_id, websocket)
    parser = PreferenceParser()
    generator = CocktailGenerator(db)
    convo = RussianBartenderAgent()
    SESSION_STATE[session_id] = _new_state()

    # Приветствие
    await manager.send_message(
        session_id,
        json.dumps(
            {
                "type": "system",
                "content": convo.greet(),
            },
        ),
    )
    await manager.send_message(
        session_id,
        json.dumps({"type": "assistant", "content": convo.next_question(SESSION_STATE[session_id])}),
    )

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            user_msg = str(data.get("message", "")).strip()
            if not user_msg:
                continue

            await save_dialogue_message(db, session_id, "user", user_msg)
            state = SESSION_STATE.get(session_id) or _new_state()

            # 1) Если ждём оценку — сохраняем feedback
            if state["phase"] == "awaiting_rating":
                rating_raw = user_msg.strip()
                rating_val = int(rating_raw) if rating_raw.isdigit() else None
                from app.models import Feedback

                if state.get("pending_order_id") is not None:
                    feedback = Feedback(order_id=state["pending_order_id"], rating=rating_val, comment=rating_raw or None)
                    db.add(feedback)
                    await db.commit()
                await manager.send_message(
                    session_id,
                    json.dumps({"type": "success", "content": "Спасибо! Оценка сохранена. Могу предложить следующий напиток под новое настроение."}),
                )
                state["phase"] = "discovery"
                state["draft"] = None
                state["pending_order_id"] = None
                state["pending_cocktail_id"] = None
                state["collected"] = {"taste": False, "mood": False, "base": False, "extras": False}
                await manager.send_message(
                    session_id,
                    json.dumps({"type": "assistant", "content": _next_question(state)}),
                )
                SESSION_STATE[session_id] = state
                continue

            # 2) Режим редактирования готового черновика
            if state["phase"] == "draft_ready" and state.get("draft"):
                if _is_confirm_message(user_msg):
                    draft = state["draft"]
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
                    if not success:
                        await manager.send_message(
                            session_id,
                            json.dumps({"type": "error", "content": "Не удалось подтвердить заказ (недостаточно ингредиентов). Давайте скорректируем рецепт."}),
                        )
                        continue
                    state["pending_order_id"] = order.id
                    state["pending_cocktail_id"] = new_cocktail.id
                    state["phase"] = "awaiting_rating"
                    await manager.send_message(
                        session_id,
                        json.dumps(
                            {
                                "type": "success",
                                "content": (
                                    "Заказ подтверждён и отправлен в приготовление.\n"
                                    "После дегустации оцените напиток от 1 до 5."
                                ),
                            }
                        ),
                    )
                    SESSION_STATE[session_id] = state
                    continue

                if _is_adjustment_message(user_msg):
                    state["draft"] = await generator.adjust_recipe(state["draft"], user_msg, state["prefs"])
                    card = generator.format_recipe_card(state["draft"])
                    await manager.send_message(session_id, json.dumps({"type": "recommendation", "message": card}))
                    await save_dialogue_message(db, session_id, "assistant", card)
                    SESSION_STATE[session_id] = state
                    continue

                await manager.send_message(
                    session_id,
                    json.dumps(
                        {
                            "type": "assistant",
                            "content": (
                                "Если хотите изменения — напишите, например, 'добавь мяту', 'убери сироп', "
                                "'сделай менее сладким', 'без льда' или 'объем 350 мл'. "
                                "Если всё устраивает — напишите 'подтвердить'."
                            ),
                        }
                    ),
                )
                continue

            # 3) Discovery: собираем предпочтения и задаём наводящие вопросы
            parsed = parser.parse(user_msg)
            _merge_prefs(state["prefs"], parsed)
            _parse_meta_preferences(user_msg, state)

            if _need_more_questions(state) and not _is_start_generation_message(user_msg):
                await manager.send_message(
                    session_id,
                    json.dumps(
                        {
                            "type": "assistant",
                            "content": f"{convo.ack()} {convo.next_question(state)}",
                        }
                    ),
                )
                SESSION_STATE[session_id] = state
                continue

            cocktail_data = await generator.generate(state["prefs"])
            if cocktail_data is None:
                await manager.send_message(
                    session_id,
                    json.dumps(
                        {
                            "type": "error",
                            "content": "Сейчас не смог собрать сбалансированный вариант из доступных ингредиентов. Давайте уточним пожелания: можно менее кислый или без ограничений?",
                        }
                    ),
                )
                continue

            card = generator.format_recipe_card(cocktail_data)
            state["draft"] = cocktail_data
            state["phase"] = "draft_ready"
            await manager.send_message(
                session_id,
                json.dumps({"type": "assistant", "content": f"{convo.ack()} {convo.draft_intro()}"}),
            )
            await manager.send_message(session_id, json.dumps({"type": "recommendation", "message": card}))
            await manager.send_message(
                session_id,
                json.dumps({"type": "assistant", "content": convo.adjustment_hint()}),
            )
            await save_dialogue_message(db, session_id, "assistant", card)
            SESSION_STATE[session_id] = state

    except WebSocketDisconnect:
        # Клиент разорвал соединение — просто завершаем обработчик.
        pass
    finally:
        SESSION_STATE.pop(session_id, None)
        manager.disconnect(session_id)

