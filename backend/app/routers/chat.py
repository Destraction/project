from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.cocktail_generator import CocktailGenerator
from app.free_llm_client import FreeLLMBartender
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
            "desired_terms": [],
            "avoid_terms": [],
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
    explicit = set(src.get("_explicit_fields", []))
    for key in ("sweetness", "sourness", "fruitiness", "citrus", "mint", "spice"):
        if key in src and key in explicit:
            dst[key] = max(0.0, min(1.0, float(src[key])))
    if src.get("base") and "base" in explicit:
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
        "жив": "energized",
        "легк": "relaxed",
        "мягк": "relaxed",
        "ярк": "party",
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
    if "газиров" in t or "пузыр" in t:
        prefs["base"] = "газированная вода"
        state["collected"]["base"] = True
    if "негазир" in t:
        prefs["base"] = "вода"
        state["collected"]["base"] = True
    if "обычн" in t and "вода" in t:
        prefs["base"] = "вода"
        state["collected"]["base"] = True

    # Маркеры что вкусовые предпочтения указаны
    if any(x in t for x in ["слад", "кисл", "мят", "цитрус", "фрукт", "прян", "лайм", "лимон"]):
        state["collected"]["taste"] = True

    # Если настроение не распознано по словарю, но ответ осмысленный — всё равно фиксируем.
    if not state["collected"]["mood"] and len(t.strip()) >= 4:
        profile["mood"] = t.strip()
        state["collected"]["mood"] = True


def _need_more_questions(state: Dict[str, Any]) -> bool:
    c = state["collected"]
    return not (c["taste"] and c["mood"] and c["base"])


def _next_question(state: Dict[str, Any]) -> str:
    c = state["collected"]
    if not c["mood"]:
        return "Какое настроение у напитка?"
    if not c["base"]:
        return "Основа: вода или газированная?"
    if not c["taste"]:
        return "По вкусу куда смещаемся?"
    return "Есть ограничения?"


def _is_start_generation_message(text: str) -> bool:
    t = text.lower()
    triggers = ["предлож", "собери", "готово", "давай вариант", "сделай коктейль", "подбери"]
    return any(x in t for x in triggers)


def _is_confirm_message(text: str) -> bool:
    t = text.lower().strip()
    return t in {"да", "подтвердить", "ок", "подтверждаю", "оформить", "принимаю"}


def _is_adjustment_message(text: str) -> bool:
    t = text.lower().strip()
    markers = ["добавь", "убери", "удали", "слаще", "менее слад", "меньше слад", "объем", "объём", "без льда", "добавь лёд", "добавь лед", "/10"]
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
    llm = FreeLLMBartender()
    SESSION_STATE[session_id] = _new_state()

    async def ask_llm(prompt: str, state_obj: Dict[str, Any]) -> str:
        try:
            return await asyncio.wait_for(
                llm.reply(
                    db=db,
                    user_message=prompt,
                    state=state_obj,
                    fallback_text="",
                    strict_json=False,
                ),
                timeout=20.0,
            )
        except Exception:
            return "Я на связи, но модель отвечает медленно. Напиши ещё раз через пару секунд."

    # Приветствие
    await manager.send_message(
        session_id,
        json.dumps(
            {
                "type": "system",
                "content": await ask_llm("Начало диалога с гостем. Коротко поздоровайся и спроси, как прошел день.", SESSION_STATE[session_id]),
            },
        ),
    )
    await manager.send_message(
        session_id,
        json.dumps(
            {
                "type": "assistant",
                "content": await ask_llm("Задай первый уточняющий вопрос по напитку.", SESSION_STATE[session_id]),
            }
        ),
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
                    json.dumps(
                        {
                            "type": "success",
                            "content": await ask_llm("Гость оставил оценку. Ответь тепло и предложи следующий напиток.", state),
                        }
                    ),
                )
                state["phase"] = "discovery"
                state["draft"] = None
                state["pending_order_id"] = None
                state["pending_cocktail_id"] = None
                state["collected"] = {"taste": False, "mood": False, "base": False, "extras": False}
                await manager.send_message(
                    session_id,
                    json.dumps(
                        {
                            "type": "assistant",
                            "content": await ask_llm("Продолжи диалог и собери новые пожелания для следующего напитка.", state),
                        }
                    ),
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
                            json.dumps(
                                {
                                    "type": "error",
                                    "content": await ask_llm("Заказ не подтвердился из-за остатков. Коротко предложи поправить рецепт.", state),
                                }
                            ),
                        )
                        continue
                    state["pending_order_id"] = order.id
                    state["pending_cocktail_id"] = new_cocktail.id
                    state["phase"] = "awaiting_rating"
                    success_text = await ask_llm("Сообщи, что заказ подтверждён, и попроси оценку 1-5.", state)
                    await manager.send_message(
                        session_id,
                        json.dumps({"type": "success", "content": success_text}),
                    )
                    SESSION_STATE[session_id] = state
                    continue

                if _is_adjustment_message(user_msg):
                    parsed_adj = parser.parse(user_msg)
                    _merge_prefs(state["prefs"], parsed_adj)
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
                            "content": await ask_llm(user_msg, state),
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
                            "content": await ask_llm(user_msg, state),
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
                            "content": await ask_llm(
                                "Не удалось собрать коктейль из остатков. Задай несколько уточняющих вопросов и предложи другой вариант.",
                                state,
                            ),
                        }
                    ),
                )
                continue

            card = generator.format_recipe_card(cocktail_data)
            state["draft"] = cocktail_data
            state["phase"] = "draft_ready"
            await manager.send_message(
                session_id,
                json.dumps(
                    {
                        "type": "assistant",
                        "content": await ask_llm("Представь вариант напитка по-дружески и коротко.", state),
                    }
                ),
            )
            await manager.send_message(session_id, json.dumps({"type": "recommendation", "message": card}))
            await manager.send_message(
                session_id,
                json.dumps(
                    {
                        "type": "assistant",
                        "content": await ask_llm("Спроси, хочет ли гость что-то изменить перед подтверждением.", state),
                    }
                ),
            )
            await save_dialogue_message(db, session_id, "assistant", card)
            SESSION_STATE[session_id] = state

    except WebSocketDisconnect:
        # Клиент разорвал соединение — просто завершаем обработчик.
        pass
    finally:
        SESSION_STATE.pop(session_id, None)
        manager.disconnect(session_id)

