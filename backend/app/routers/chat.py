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


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").lower().replace("ё", "е")).strip()


def _stem_token(token: str) -> str:
    t = _normalize_text(token)
    endings = ("иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "ая", "ое", "ые", "ую", "ий", "ый", "ой", "ов", "ев", "ам", "ям", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь")
    for ending in endings:
        if t.endswith(ending) and len(t) - len(ending) >= 4:
            return t[: -len(ending)]
    return t


def _build_ingredient_matchers(ingredient_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    normalized_full = sorted({_normalize_text(x) for x in ingredient_names if x}, key=len, reverse=True)
    stop_tokens = {"вода", "газированная", "сироп", "сок", "лед", "лёд", "основа", "напиток", "коктейль"}
    stem_map: Dict[str, str] = {}
    for name in normalized_full:
        for token in re.findall(r"[а-яa-z0-9-]+", name):
            stem = _stem_token(token)
            if len(stem) < 4 or stem in stop_tokens:
                continue
            stem_map.setdefault(stem, name)
    return normalized_full, stem_map


def _extract_requested_ingredients(text: str, full_names: List[str], stem_map: Dict[str, str]) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    found: List[str] = []
    for name in full_names:
        if len(name) >= 4 and name in normalized:
            found.append(name)
    for token in re.findall(r"[а-яa-z0-9-]+", normalized):
        stem = _stem_token(token)
        matched = stem_map.get(stem)
        if matched:
            found.append(matched)
    return list(dict.fromkeys(found))[:8]


def _register_requested_ingredients(state: Dict[str, Any], ingredients: List[str]) -> None:
    if not ingredients:
        return
    profile = state.setdefault("profile", {})
    current = list(profile.get("requested_ingredients", []) or [])
    merged = list(dict.fromkeys([*current, *ingredients]))[-30:]
    profile["requested_ingredients"] = merged
    state["collected"]["composition"] = True


def _composition_question(state: Dict[str, Any]) -> str:
    examples = list(state.get("profile", {}).get("composition_examples", []) or [])[:5]
    if examples:
        sample = ", ".join(examples)
        return f"Отлично, направление зафиксировали. Теперь назови 1-3 конкретных ингредиента, которые хочешь в составе. Например: {sample}."
    return "Отлично, направление зафиксировали. Теперь назови 1-3 конкретных ингредиента, которые хочешь видеть в составе."


def _normalize_avoid_term(raw_term: str) -> str:
    term = (raw_term or "").strip().lower()
    term = re.split(r"[,.!?;:]", term)[0].strip()
    if not term:
        return ""
    if "цитрус" in term:
        return "цитрус"
    if "орех" in term:
        return "орех"
    if "молок" in term:
        return "молоко"
    if "имбир" in term:
        return "имбир"
    if "мят" in term:
        return "мята"
    words = [x for x in re.findall(r"[а-яa-zё0-9\-]+", term) if len(x) >= 3]
    return words[0] if words else ""


def _extract_avoid_terms_from_text(text: str) -> list[str]:
    t = (text or "").lower()
    patterns = [
        r"(?:без|исключи|исключить|не добавляй|не добавлять|не клади|не класть|обойти)\s+([а-яa-zё0-9\-\s]{2,60})",
        r"(?:аллерг\w*|аллегр\w*|не переношу|нельзя)\s*(?:на|к)?\s*([а-яa-zё0-9\-\s]{2,60})",
    ]
    extracted: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, t):
            chunk = str(match).strip()
            if not chunk:
                continue
            piece = re.split(r"(?:\s+но\s+|\s+а\s+|\s+и\s+)", chunk)[0].strip()
            normalized = _normalize_avoid_term(piece)
            if normalized:
                extracted.append(normalized)
    return list(dict.fromkeys(extracted))


def _parse_meta_preferences(text: str, state: Dict[str, Any]) -> None:
    t = text.lower()
    profile = state["profile"]
    prefs = state["prefs"]

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
        "фрукт": "fruity",
        "ягод": "berry",
        "прян": "spicy",
        "цитрус": "citrus",
        "троп": "tropical",
        "косточ": "stone_fruit",
    }
    for k, mood in mood_map.items():
        if k in t:
            profile["mood"] = mood
            state["collected"]["mood"] = True
            break

    avoid_terms = _extract_avoid_terms_from_text(t)
    if avoid_terms:
        profile.setdefault("avoid", [])
        prefs.setdefault("avoid_terms", [])
        for term in avoid_terms:
            profile["avoid"].append(term)
            prefs["avoid_terms"].append(term)
        profile["avoid"] = list(dict.fromkeys(profile["avoid"]))[-30:]
        prefs["avoid_terms"] = list(dict.fromkeys(prefs["avoid_terms"]))[-30:]
        state["collected"]["extras"] = True

    if any(x in t for x in ["нет огранич", "без огранич", "ограничений нет", "не огранич"]):
        state["collected"]["extras"] = True
    if any(x in t for x in ["всё подходит", "все подходит", "всё остальное подходит", "все остальное подходит", "больше ничего", "других ограничений нет"]):
        state["collected"]["extras"] = True

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

    if any(x in t for x in ["слад", "кисл", "мят", "цитрус", "фрукт", "прян", "лайм", "лимон", "легк", "освеж", "плотн", "насыщ"]):
        state["collected"]["taste"] = True

    if any(x in t for x in ["огранич", "исключ", "аллер", "не использ", "не добав"]):
        state["collected"]["extras"] = True


def _need_more_questions(state: Dict[str, Any]) -> bool:
    c = state["collected"]
    return not (c["taste"] and c["mood"] and c["base"] and c["extras"] and c["composition"])


def _next_question(state: Dict[str, Any]) -> str:
    c = state["collected"]
    if not c["mood"]:
        return "Какое настроение у напитка?"
    if not c["base"]:
        return "Основа: вода или газированная?"
    if not c["taste"]:
        return "По вкусу куда смещаемся?"
    return "Есть ограничения?"


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


def _extract_facets(text: str) -> list[str]:
    t = (text or "").lower()
    facets: list[str] = []
    rules = {
        "base": ["основа", "вода", "газирован", "газиров", "пузыр", "сода"],
        "style": ["фрукт", "ягод", "прян", "троп", "косточ", "цитрус", "травян"],
        "composition": ["состав", "ингредиент", "добав", "конкретн"],
        "sweetness": ["слад", "притор", "сахар", "сироп"],
        "sourness": ["кисл", "лайм", "лимон", "маракуй", "грейпфрут", "юдзу", "каламанси"],
        "flavor_profile": ["фрукт", "цитрус", "ягод", "троп", "манго", "ананас", "кокос"],
        "texture": ["лёгк", "легк", "плотн", "десерт", "насыщ"],
        "carbonation": ["газирован", "с пузыр", "пузыр", "шипуч", "без газа"],
        "ice_volume": ["лед", "лёд", "объем", "объём", "мл"],
        "constraints": ["без ", "исключ", "не добав", "аллер", "нельзя"],
    }
    for facet, markers in rules.items():
        if any(m in t for m in markers):
            facets.append(facet)
    return list(dict.fromkeys(facets))


def _is_question_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if "?" in t:
        return True
    return bool(re.search(r"\b(какой|какая|какие|нужен|нужна|нужно|предпочитаете|хотите|выберите|куда)\b", t))


def _is_negative_preference_response(text: str) -> bool:
    cleaned = _normalize_text(re.sub(r"[,.!?;:]+", " ", text))
    if not cleaned:
        return False
    negative_starts = (
        "нет",
        "неа",
        "ничего",
        "никаких",
        "никакого",
        "никакая",
        "без ограничений",
        "без исключений",
        "запретов нет",
        "нет запретов",
    )
    if any(cleaned.startswith(x) for x in negative_starts):
        return True
    return cleaned in {"все подходит", "всё подходит", "такого нет", "ничего такого"}


def _select_primary_facet_for_question(facets: list[str]) -> str | None:
    if not facets:
        return None
    priority = ["constraints", "composition", "base", "carbonation", "sweetness", "sourness", "texture", "style", "flavor_profile", "ice_volume"]
    for item in priority:
        if item in facets:
            return item
    return facets[0]


def _mark_answered_from_user(state: Dict[str, Any], user_msg: str) -> None:
    facets = _extract_facets(user_msg)
    answered = state.get("answered_facets", [])
    lowered = (user_msg or "").strip().lower()
    if not facets and (lowered in {"да", "ага", "ок", "хорошо"} or _is_negative_preference_response(lowered)):
        last_facet = state.get("last_asked_facet")
        if isinstance(last_facet, str) and last_facet:
            facets = [last_facet]
    if not facets and _is_negative_preference_response(lowered):
        last_question = ""
        recent_questions = state.get("recent_assistant_questions", []) or []
        if recent_questions:
            last_question = str(recent_questions[-1])
        last_question_facets = _extract_facets(last_question)
        if "constraints" in last_question_facets:
            facets = ["constraints"]
    for facet in facets:
        if facet not in answered:
            answered.append(facet)
    state["answered_facets"] = answered[-20:]
    facet_to_collected = {
        "constraints": "extras",
        "base": "base",
        "carbonation": "base",
        "style": "mood",
        "composition": "composition",
        "sweetness": "taste",
        "sourness": "taste",
        "flavor_profile": "taste",
        "texture": "taste",
        "ice_volume": "extras",
    }
    collected = state.get("collected", {})
    for facet in facets:
        key = facet_to_collected.get(facet)
        if key:
            collected[key] = True
    state["collected"] = collected


def _mark_asked_from_assistant(state: Dict[str, Any], assistant_msg: str) -> None:
    if not _is_question_text(assistant_msg):
        return
    facets = _extract_facets(assistant_msg)
    asked = state.get("asked_facets", [])
    for facet in facets:
        if facet not in asked:
            asked.append(facet)
    state["asked_facets"] = asked[-20:]
    selected = _select_primary_facet_for_question(facets)
    state["last_asked_facet"] = selected if selected else state.get("last_asked_facet")
    state["questions_asked_count"] = int(state.get("questions_asked_count", 0)) + 1
    questions = state.get("recent_assistant_questions", [])
    normalized = re.sub(r"\s+", " ", assistant_msg.lower()).strip()
    questions.append(normalized)
    state["recent_assistant_questions"] = questions[-12:]


def _question_repeats_recent(state: Dict[str, Any], assistant_msg: str) -> bool:
    if not _is_question_text(assistant_msg):
        return False
    normalized = re.sub(r"\s+", " ", (assistant_msg or "").lower()).strip()
    recent = state.get("recent_assistant_questions", []) or []
    return normalized in recent


def _unresolved_facets(state: Dict[str, Any]) -> list[str]:
    collected = state.get("collected", {})
    mapping = {
        "mood": "направление вкуса",
        "base": "тип основы",
        "composition": "конкретные ингредиенты состава",
        "taste": "баланс вкуса",
        "extras": "ограничения и исключения",
    }
    unresolved: list[str] = []
    for key, title in mapping.items():
        if not bool(collected.get(key, False)):
            unresolved.append(title)
    return unresolved


def _refresh_confirmed_terms(state: Dict[str, Any]) -> None:
    prefs = state.get("prefs", {})
    desired_terms = list(prefs.get("desired_terms", []) or [])
    requested = list(state.get("profile", {}).get("requested_ingredients", []) or [])
    generic_markers = (
        "напит",
        "коктей",
        "вкус",
        "освеж",
        "легк",
        "плот",
        "слад",
        "кисл",
        "прян",
        "баланс",
        "основа",
        "фрукт",
        "ягод",
        "цитрус",
        "троп",
        "косточ",
    )
    terms: list[str] = [_normalize_text(x) for x in requested if _normalize_text(x)]
    for term in desired_terms:
        t = str(term).strip().lower()
        if len(t) < 4:
            continue
        if any(t.startswith(marker) for marker in generic_markers):
            continue
        terms.append(t)
    state["profile"]["confirmed_terms"] = list(dict.fromkeys(terms))[-30:]


def _question_repeats_answered_facet(state: Dict[str, Any], assistant_msg: str) -> bool:
    if not _is_question_text(assistant_msg):
        return False
    asked_now = set(_extract_facets(assistant_msg))
    if not asked_now:
        return False
    answered = set(state.get("answered_facets", []) or [])
    return bool(asked_now & answered)


def _is_start_generation_message(text: str) -> bool:
    t = text.lower()
    triggers = ["предлож", "собери", "готово", "давай вариант", "сделай коктейль", "подбери"]
    return any(x in t for x in triggers)


def _is_confirm_message(text: str) -> bool:
    t = text.lower().strip()
    return t in {"да", "подтвердить", "ок", "подтверждаю", "оформить", "принимаю"}


def _is_adjustment_message(text: str) -> bool:
    t = text.lower().strip()
    markers = [
        "добав",
        "убер",
        "удал",
        "замен",
        "измени",
        "скоррект",
        "слаще",
        "менее слад",
        "меньше слад",
        "кислее",
        "менее кисл",
        "меньше кисл",
        "прян",
        "мят",
        "цитрус",
        "фрукт",
        "больше",
        "меньше",
        "усиль",
        "ослаб",
        "объем",
        "объём",
        "мл",
        "без льда",
        "со льдом",
        "с льдом",
        "/10",
    ]
    return any(m in t for m in markers)


def _should_adjust_draft(text: str, state: Dict[str, Any], parsed: Dict[str, Any]) -> bool:
    if _is_adjustment_message(text):
        return True
    explicit_fields = parsed.get("_explicit_fields", [])
    if explicit_fields:
        return True
    if parsed.get("desired_terms") or parsed.get("avoid_terms"):
        return True
    draft = state.get("draft") or {}
    draft_details = draft.get("details") or []
    t = text.lower().strip()
    for item in draft_details:
        ingredient_name = str(item.get("name", "")).lower()
        if ingredient_name and ingredient_name in t:
            return True
    return False


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    await manager.connect(session_id, websocket)
    parser = PreferenceParser()
    llm = FreeLLMBartender()
    SESSION_STATE[session_id] = _new_state()
    ingredients_res = await db.execute(select(Ingredient))
    all_ingredients = list(ingredients_res.scalars().all())
    flavor_ingredients = [
        ing for ing in all_ingredients if float(ing.quantity or 0.0) > 0.0 and str(ing.category or "").lower() not in {"base", "ice"}
    ]
    ingredient_names = [str(ing.name) for ing in flavor_ingredients]
    ingredient_full_names, ingredient_stem_map = _build_ingredient_matchers(ingredient_names)
    composition_examples = [str(ing.name) for ing in flavor_ingredients[:10]]
    SESSION_STATE[session_id]["profile"]["composition_examples"] = composition_examples

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

    async def build_draft_with_llm(state_obj: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            raw = await asyncio.wait_for(
                llm.propose_cocktail_from_db(db=db, state=state_obj),
                timeout=30.0,
            )
        except Exception:
            raw = None
        if not raw:
            return None
        recipe = raw.get("recipe", {})
        if not isinstance(recipe, dict) or not recipe:
            return None
        details = await llm.build_recipe_details(db, recipe)
        if len(details) < 2:
            return None
        totals = llm.calculate_totals(details)
        taste_profile = llm.calculate_taste_profile(state_obj.get("prefs", {}), details)
        raw["details"] = details
        raw["totals"] = totals
        raw["taste_profile"] = taste_profile
        return raw

    greeting_text = await ask_llm(
        "Начало диалога. Ответь в тоне бармен-гость одной короткой дружелюбной репликой и подготовь к выбору формата вкуса.",
        SESSION_STATE[session_id],
    )
    _remember_assistant_message(SESSION_STATE[session_id], greeting_text)
    await manager.send_message(
        session_id,
        json.dumps(
            {
                "type": "system",
                "content": greeting_text,
            },
        ),
    )
    first_question = await ask_llm(
        "Задай первый вопрос в формате бармена с вариантами направлений вкуса. Пример структуры: фруктовая, ягодная или пряная база.",
        SESSION_STATE[session_id],
    )
    _remember_assistant_message(SESSION_STATE[session_id], first_question)
    _mark_asked_from_assistant(SESSION_STATE[session_id], first_question)
    await manager.send_message(
        session_id,
        json.dumps(
            {
                "type": "assistant",
                "content": first_question,
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
            _remember_user_message(state, user_msg)
            _mark_answered_from_user(state, user_msg)

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
                state["collected"]["composition"] = False
                state["asked_facets"] = []
                state["answered_facets"] = []
                state["recent_assistant_questions"] = []
                state["last_asked_facet"] = None
                state["questions_asked_count"] = 0
                state["profile"]["notes"] = []
                state["profile"]["confirmed_terms"] = []
                state["profile"]["requested_ingredients"] = []
                state["profile"]["composition_examples"] = composition_examples
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

                parsed_adj = parser.parse(user_msg)
                if _should_adjust_draft(user_msg, state, parsed_adj):
                    requested_now = _extract_requested_ingredients(user_msg, ingredient_full_names, ingredient_stem_map)
                    _register_requested_ingredients(state, requested_now)
                    _merge_prefs(state["prefs"], parsed_adj)
                    _parse_meta_preferences(user_msg, state)
                    _refresh_confirmed_terms(state)
                    updated = await llm.propose_cocktail_from_db(
                        db=db,
                        state=state,
                        user_request=user_msg,
                        current_draft=state["draft"],
                    )
                    if not updated:
                        await manager.send_message(
                            session_id,
                            json.dumps(
                                {
                                    "type": "error",
                                    "content": await ask_llm("Не удалось применить правку. Предложи новую корректировку по вкусу.", state),
                                }
                            ),
                        )
                        continue
                    details = await llm.build_recipe_details(db, updated["recipe"])
                    updated["details"] = details
                    updated["totals"] = llm.calculate_totals(details)
                    updated["taste_profile"] = llm.calculate_taste_profile(state["prefs"], details)
                    state["draft"] = updated
                    card = llm.format_recipe_card(state["draft"])
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
            requested_now = _extract_requested_ingredients(user_msg, ingredient_full_names, ingredient_stem_map)
            _register_requested_ingredients(state, requested_now)
            _merge_prefs(state["prefs"], parsed)
            _parse_meta_preferences(user_msg, state)
            _refresh_confirmed_terms(state)

            if _need_more_questions(state) and not _is_start_generation_message(user_msg):
                if state["collected"].get("mood") and not state["collected"].get("composition"):
                    assistant_text = _composition_question(state)
                else:
                    assistant_text = await ask_llm(user_msg, state)
                if _question_repeats_answered_facet(state, assistant_text) or _question_repeats_recent(state, assistant_text):
                    unresolved = _unresolved_facets(state)
                    unresolved_text = ", ".join(unresolved) if unresolved else "новый аспект вкуса"
                    if state["collected"].get("mood") and not state["collected"].get("composition"):
                        assistant_text = _composition_question(state)
                    else:
                        assistant_text = await ask_llm(
                            "Ты повторил уже закрытую тему. Задай только один новый вопрос по незакрытым темам: "
                            f"{unresolved_text}. Не повторяй прошлые формулировки дословно.",
                            state,
                        )
                _remember_assistant_message(state, assistant_text)
                _mark_asked_from_assistant(state, assistant_text)
                await manager.send_message(
                    session_id,
                    json.dumps(
                        {
                            "type": "assistant",
                            "content": assistant_text,
                        }
                    ),
                )
                SESSION_STATE[session_id] = state
                continue

            if not state["collected"].get("composition"):
                assistant_text = _composition_question(state)
                _remember_assistant_message(state, assistant_text)
                _mark_asked_from_assistant(state, assistant_text)
                await manager.send_message(
                    session_id,
                    json.dumps({"type": "assistant", "content": assistant_text}),
                )
                SESSION_STATE[session_id] = state
                continue

            cocktail_data = await build_draft_with_llm(state)
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

            card = llm.format_recipe_card(cocktail_data)
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
