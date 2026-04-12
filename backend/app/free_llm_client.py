from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Compatibility, Ingredient


class FreeLLMBartender:
    """
    Клиент для общения с моделью бармена.
    """

    SYSTEM_PROMPT = (
        "Ты — опытный бармен-нутрициолог в безалкогольном баре. Твоя цель — провести гостя от приветствия до заказа идеального коктейля.\n"
        "ВСЁ ОБЩЕНИЕ И ЛОГИКА ОПРЕДЕЛЯЮТСЯ ТОБОЙ ЧЕРЕЗ JSON.\n\n"
        "ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА (СТРОГО JSON):\n"
        "{\n"
        "  \"reply\": \"Твой текст для гостя (без markdown, приятельский тон)\",\n"
        "  \"action\": \"chat | propose_recipe | confirm_order\",\n"
        "  \"recipe\": {\n"
        "    \"name\": \"Название\",\n"
        "    \"description\": \"Красивое описание\",\n"
        "    \"ingredients\": [{\"name\": \"ингредиент из списка\", \"qty\": 50}],\n"
        "    \"serving_ml\": 300\n"
        "  }\n"
        "}\n\n"
        "ПРАВИЛА:\n"
        "1. Используй ТОЛЬКО ингредиенты из available_ingredients. Соблюдай остатки (quantity).\n"
        "2. В режиме 'chat' (action='chat') — узнавай предпочтения. Задавай ОДИН вопрос за раз.\n"
        "3. Когда предпочтения ясны (есть направление вкуса и 1-3 конкретных ингредиента), переходи к 'propose_recipe'.\n"
        "4. В 'propose_recipe' — сформируй полный рецепт (2-7 ингредиентов) и красиво опиши его в 'reply'.\n"
        "5. Если гость просит правки (добавить, убрать, изменить сладость) — снова используй 'propose_recipe' с обновленным составом.\n"
        "6. Если гость подтверждает (пишет 'да', 'подтверждаю', 'оформить') — используй action='confirm_order'.\n"
        "7. Если данных не хватает — продолжай 'chat'.\n"
        "8. Учитывай prefs (лед, сладость и т.д.) и profile (историю).\n"
    )

    def __init__(self) -> None:
        self.enabled_chad = bool(getattr(settings, "ENABLE_CHAD", False)) and bool(getattr(settings, "CHAD_API_KEY", ""))
        self.chad_api_key = getattr(settings, "CHAD_API_KEY", "")
        self.chad_model = getattr(settings, "CHAD_MODEL", "gpt-5.4-mini")
        self.chad_endpoint = f"https://ask.chadgpt.ru/api/public/{self.chad_model}"

        self.min_required_by_unit = {"ml": 30.0, "g": 10.0, "piece": 1.0}

    async def build_bar_context(self, db: AsyncSession) -> Dict[str, Any]:
        res = await db.execute(select(Ingredient))
        ingredients = list(res.scalars().all())
        compat_rows = (await db.execute(select(Compatibility))).scalars().all()

        by_category: Dict[str, List[str]] = {}
        available_stock: Dict[str, float] = {}
        for ing in ingredients:
            min_need = self.min_required_by_unit.get((ing.unit or "").lower(), 1.0)
            if float(ing.quantity or 0.0) < min_need:
                continue
            by_category.setdefault(ing.category, []).append(ing.name)
            available_stock[ing.name.lower()] = float(ing.quantity or 0.0)

        compat_pairs: List[Tuple[str, str, int]] = []
        id_to_name = {x.id: x.name for x in ingredients}
        for row in compat_rows:
            n1 = id_to_name.get(row.ing1_id)
            n2 = id_to_name.get(row.ing2_id)
            if not n1 or not n2:
                continue
            compat_pairs.append((n1, n2, int(row.score or 1)))

        trimmed = {k: sorted(v)[:20] for k, v in by_category.items()}
        return {
            "available_count": len(ingredients),
            "categories": sorted(by_category.keys()),
            "examples_by_category": trimmed,
            "available_stock": available_stock,
            "compatibility_examples": compat_pairs[:120],
        }

    def _extract_allowed_set(self, context: Dict[str, Any]) -> Set[str]:
        stock = context.get("available_stock", {})
        return set(stock.keys())

    def _validate_json_payload(self, payload: Dict[str, Any], allowed_set: Set[str]) -> bool:
        if not isinstance(payload, dict):
            return False
        if "reply" not in payload:
            return False
        mentioned = payload.get("mentioned_ingredients", [])
        if not isinstance(mentioned, list):
            return False
        for item in mentioned:
            if str(item).lower() not in allowed_set:
                return False
        return True

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _draft_for_prompt(
        self,
        current_draft: Optional[Dict[str, Any]],
        all_ingredients: List[Ingredient],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(current_draft, dict):
            return None
        recipe = current_draft.get("recipe", {})
        if not isinstance(recipe, dict):
            return None
        by_id = {int(ing.id): ing for ing in all_ingredients}
        ingredients_for_prompt: List[Dict[str, Any]] = []
        for ing_id_str, qty in recipe.items():
            try:
                ing_id = int(ing_id_str)
                qty_val = float(qty)
            except Exception:
                continue
            ing = by_id.get(ing_id)
            if not ing:
                continue
            ingredients_for_prompt.append(
                {
                    "id": ing_id,
                    "name": str(ing.name),
                    "category": str(ing.category),
                    "unit": str(ing.unit or "ml"),
                    "qty": qty_val,
                }
            )
        return {
            "name": str(current_draft.get("name", "")),
            "description": str(current_draft.get("description", "")),
            "ingredients": ingredients_for_prompt,
            "taste_profile": current_draft.get("taste_profile", {}),
            "totals": current_draft.get("totals", {}),
        }

    async def propose_cocktail_from_db(
        self,
        db: AsyncSession,
        state: Dict[str, Any],
        user_request: str = "",
        current_draft: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled_chad:
            return None

        res = await db.execute(select(Ingredient))
        all_ingredients = list(res.scalars().all())
        context = await self.build_bar_context(db)

        available_ingredients = [
            {
                "id": int(ing.id),
                "name": str(ing.name),
                "category": str(ing.category),
                "unit": str(ing.unit or "ml"),
                "quantity": float(ing.quantity or 0.0),
            }
            for ing in all_ingredients
            if float(ing.quantity or 0.0) > 0.0
        ]
        if not available_ingredients:
            return None

        prefs = state.get("prefs", {})
        profile = state.get("profile", {})
        is_adjustment = bool(current_draft) and bool((user_request or "").strip())

        user_payload = {
            "task": "compose_cocktail_recipe",
            "prefs": prefs,
            "profile": profile,
            "user_request": user_request,
            "current_draft": self._draft_for_prompt(current_draft, all_ingredients),
            "bar_context": context,
            "available_ingredients": available_ingredients,
            "is_adjustment": is_adjustment,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                req_json: Dict[str, Any] = {
                    "message": json.dumps(user_payload, ensure_ascii=False),
                    "api_key": self.chad_api_key,
                    "history": [{"role": "system", "content": self.SYSTEM_PROMPT}],
                }
                resp = await client.post(url=self.chad_endpoint, json=req_json)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("is_success", False):
                    return None

                payload = self._extract_json_object(str(data.get("response", "")))
                if not payload:
                    return None

                name = str(payload.get("name", "")).strip()
                description = str(payload.get("description", "")).strip()
                ingredients = payload.get("ingredients", [])
                if not name or not isinstance(ingredients, list):
                    return None

                by_name = {str(ing.name).lower(): ing for ing in all_ingredients}
                recipe: Dict[str, float] = {}
                for item in ingredients:
                    if not isinstance(item, dict):
                        continue
                    ing_name = str(item.get("name", "")).strip().lower()
                    qty_raw = item.get("qty", 0)
                    try:
                        qty = float(qty_raw)
                    except Exception:
                        continue
                    ing = by_name.get(ing_name)
                    if not ing:
                        continue
                    key = str(int(ing.id))
                    recipe[key] = round(float(recipe.get(key, 0.0)) + float(qty), 1)

                if len(recipe) < 2:
                    return None

                ingredients_details = [{"id": int(k), "qty": float(v)} for k, v in recipe.items()]
                return {
                    "name": name,
                    "description": description,
                    "recipe": recipe,
                    "ingredients_details": ingredients_details,
                }
        except Exception:
            return None

    async def build_recipe_details(self, db: AsyncSession, recipe: Dict[str, float]) -> List[Dict[str, Any]]:
        if not recipe:
            return []
        ids = [int(x) for x in recipe.keys()]
        res = await db.execute(select(Ingredient).where(Ingredient.id.in_(ids)))
        ing_map = {int(ing.id): ing for ing in res.scalars().all()}
        details: List[Dict[str, Any]] = []
        for ing_id_str, qty in recipe.items():
            ing = ing_map.get(int(ing_id_str))
            if not ing:
                continue
            details.append(
                {
                    "id": int(ing.id),
                    "name": str(ing.name),
                    "category": str(ing.category),
                    "unit": str(ing.unit or "ml"),
                    "qty": float(qty),
                }
            )
        return details

    def calculate_totals(self, details: List[Dict[str, Any]]) -> Dict[str, float]:
        nutrition_reference = {
            "base": {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0},
            "ice": {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0},
            "juice": {"kcal": 44.0, "protein": 0.4, "fat": 0.1, "carbs": 10.0},
            "syrup": {"kcal": 260.0, "protein": 0.0, "fat": 0.0, "carbs": 65.0},
            "additive": {"kcal": 130.0, "protein": 2.0, "fat": 2.0, "carbs": 18.0},
            "fruit": {"kcal": 45.0, "protein": 0.6, "fat": 0.2, "carbs": 10.0},
        }

        volume_without_ice_ml = 0.0
        ice_volume_ml = 0.0
        kcal = protein = fat = carbs = 0.0

        for item in details:
            qty = float(item.get("qty", 0.0))
            unit = str(item.get("unit") or "")
            category = str(item.get("category") or "")
            base = nutrition_reference.get(category, nutrition_reference["additive"])
            factor = qty if unit == "piece" else qty / 100.0
            factor = max(0.0, factor)
            kcal += base["kcal"] * factor
            protein += base["protein"] * factor
            fat += base["fat"] * factor
            carbs += base["carbs"] * factor

            if unit == "ml":
                if category == "ice":
                    ice_volume_ml += qty
                else:
                    volume_without_ice_ml += qty
            elif unit == "g":
                if category == "ice":
                    ice_volume_ml += qty
                else:
                    volume_without_ice_ml += qty * 0.7
            elif unit == "piece":
                volume_without_ice_ml += qty * 25.0

        return {
            "volume_without_ice_ml": round(volume_without_ice_ml, 1),
            "volume_with_ice_ml": round(volume_without_ice_ml + ice_volume_ml, 1),
            "kcal": round(kcal, 1),
            "protein": round(protein, 1),
            "fat": round(fat, 1),
            "carbs": round(carbs, 1),
        }

    def calculate_taste_profile(self, prefs: Dict[str, Any], details: List[Dict[str, Any]]) -> Dict[str, int]:
        def to_ml(item: Dict[str, Any]) -> float:
            qty = float(item.get("qty", 0.0) or 0.0)
            unit = str(item.get("unit") or "")
            if unit == "ml":
                return qty
            if unit == "g":
                return qty * 0.7
            if unit == "piece":
                return qty * 25.0
            return qty

        citrus_keywords = ["лимон", "лайм", "апельсин", "мандар", "грейпфрут", "цедра", "юдзу", "каламанси", "помело"]
        tart_keywords = ["клюкв", "гранат", "смородин", "облепих", "барбарис", "ревен", "уксус", "томат"]
        sweet_keywords = ["арбуз", "дын", "банан", "манго", "персик", "груш", "виноград", "яблок", "ананас", "маракуй"]
        mint_keywords = ["мят"]
        spice_keywords = ["имбир", "перец", "кориц", "кардамон", "гвозд", "чили", "куркум", "бадьян", "мускат"]

        flavor_volume_ml = 0.0
        total_volume_ml = 0.0
        sweet_points = 0.0
        sour_points = 0.0
        fresh_points = 0.0
        spice_points = 0.0
        body_points = 0.0

        for item in details:
            name = str(item.get("name", "")).lower()
            cat = str(item.get("category", "")).lower()
            ml = to_ml(item)
            total_volume_ml += ml

            if cat == "ice":
                continue
            if cat != "base":
                flavor_volume_ml += ml

            sweet_coef = 0.05
            sour_coef = 0.02
            fresh_coef = 0.03
            spice_coef = 0.0
            body_coef = 0.04

            if cat == "syrup":
                sweet_coef, sour_coef, fresh_coef, body_coef = 1.0, 0.02, 0.02, 0.55
            elif cat == "juice":
                sweet_coef, sour_coef, fresh_coef, body_coef = 0.22, 0.18, 0.07, 0.10
            elif cat == "fruit":
                sweet_coef, sour_coef, fresh_coef, body_coef = 0.18, 0.12, 0.08, 0.28
            elif cat == "additive":
                sweet_coef, sour_coef, fresh_coef, body_coef = 0.08, 0.05, 0.10, 0.18
            elif cat == "base":
                sweet_coef, sour_coef, fresh_coef, body_coef = 0.0, 0.0, 0.02, 0.0

            if any(k in name for k in citrus_keywords):
                sour_coef += 0.75
                fresh_coef += 0.35
                sweet_coef = max(0.0, sweet_coef - 0.08)
            if any(k in name for k in tart_keywords):
                sour_coef += 0.45
                sweet_coef = max(0.0, sweet_coef - 0.05)
            if any(k in name for k in sweet_keywords):
                sweet_coef += 0.25
                sour_coef = max(0.0, sour_coef - 0.08)
            if any(k in name for k in mint_keywords):
                fresh_coef += 0.90
            if any(k in name for k in spice_keywords):
                spice_coef += 1.05
                body_coef += 0.08
            if "газирован" in name or "сода" in name or "тоник" in name:
                fresh_coef += 0.25

            sweet_points += ml * sweet_coef
            sour_points += ml * sour_coef
            fresh_points += ml * fresh_coef
            spice_points += ml * spice_coef
            body_points += ml * body_coef

        base_den = max(1.0, flavor_volume_ml)
        total_den = max(1.0, total_volume_ml)
        sweet_proxy = min(1.0, sweet_points / (base_den * 0.85))
        sour_proxy = min(1.0, sour_points / (base_den * 0.95))
        fresh_proxy = min(1.0, fresh_points / (total_den * 0.45))
        spice_proxy = min(1.0, spice_points / (base_den * 0.60))
        body_proxy = min(1.0, body_points / (total_den * 0.35))

        if flavor_volume_ml <= 0.01:
            sweet_proxy = max(sweet_proxy, max(0.0, min(1.0, float(prefs.get("sweetness", 0.5)))))
            sour_proxy = max(sour_proxy, max(0.0, min(1.0, float(prefs.get("sourness", 0.3)))))
            fresh_proxy = max(fresh_proxy, 0.2)

        return {
            "sweetness": min(10, max(1, int(round(sweet_proxy * 10)))),
            "sourness": min(10, max(1, int(round(sour_proxy * 10)))),
            "freshness": min(10, max(1, int(round(fresh_proxy * 10)))),
            "spice": min(10, max(1, int(round(spice_proxy * 10)))),
            "body": min(10, max(1, int(round(body_proxy * 10)))),
        }

    async def reply(
        self,
        db: AsyncSession,
        user_message: str,
        state: Dict[str, Any],
        fallback_text: str,
        strict_json: bool = False,
    ) -> str:
        technical_fallback = "Модель Chad недоступна (не задан `CHAD_API_KEY` или выключен `ENABLE_CHAD`)."
        if not self.enabled_chad:
            return fallback_text or technical_fallback

        context = await self.build_bar_context(db)
        prefs = state.get("prefs", {})
        profile = state.get("profile", {})
        phase = state.get("phase", "discovery")
        allowed_set = self._extract_allowed_set(context)

        user_payload = {
            "phase": phase,
            "user_message": user_message,
            "prefs": prefs,
            "profile": profile,
            "bar_context": context,
            "strict_json": strict_json,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                req_json: Dict[str, Any] = {
                    "message": json.dumps(
                        {
                            "user_payload": user_payload,
                            "user_message": user_message,
                            "phase": phase,
                            "strict_json": strict_json,
                        },
                        ensure_ascii=False,
                    ),
                    "api_key": self.chad_api_key,
                    "history": [{"role": "system", "content": self.SYSTEM_PROMPT}],
                }
                resp = await client.post(url=self.chad_endpoint, json=req_json)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("is_success", False):
                    return fallback_text or str(data.get("error_message") or technical_fallback)

                content = str(data.get("response", "")).strip()
                if not content:
                    return fallback_text or technical_fallback

                # Мы больше не санируем текст жестко, даем ИИ свободу.
                # Но если это strict_json, проверяем его.
                if strict_json:
                    try:
                        parsed = json.loads(content)
                        if not self._validate_json_payload(parsed, allowed_set):
                            return fallback_text or technical_fallback
                        return content # Возвращаем сырой JSON для дальнейшей обработки в chat.py
                    except Exception:
                        return fallback_text or technical_fallback

                return content
        except Exception:
            return fallback_text or technical_fallback
