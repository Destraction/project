from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Ingredient, Compatibility


class CocktailGenerator:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        # Упрощённая нутриционная база (на 100 ml / 100 g / 1 piece)
        self.nutrition_reference = {
            "base": {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0},
            "ice": {"kcal": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0},
            "juice": {"kcal": 44.0, "protein": 0.4, "fat": 0.1, "carbs": 10.0},
            "syrup": {"kcal": 260.0, "protein": 0.0, "fat": 0.0, "carbs": 65.0},
            "additive": {"kcal": 130.0, "protein": 2.0, "fat": 2.0, "carbs": 18.0},
            "fruit": {"kcal": 45.0, "protein": 0.6, "fat": 0.2, "carbs": 10.0},
        }

    def _qty_for_unit(self, unit: str, role: str) -> float:
        # Простые фиксированные нормы под seed_data (ml/piece/g).
        if unit == "ml":
            return 200.0 if role == "base" else 30.0
        if unit == "piece":
            return 2.0 if role == "main" else 1.0
        if unit == "g":
            return 20.0 if role == "main" else 10.0
        # Неизвестная единица: на всякий случай чуть меньше
        return 10.0

    async def _get_ingredients_by_category(self, category: str) -> List[Ingredient]:
        res = await self.db.execute(
            select(Ingredient).where(and_(Ingredient.category == category))
        )
        return list(res.scalars().all())

    async def _find_ingredient_with_enough(self, ingredients: List[Ingredient], role: str, qty_needed: float) -> Optional[Ingredient]:
        # Выбираем первый ингредиент, у которого хватает остатка.
        for ing in ingredients:
            if ing.quantity is not None and float(ing.quantity) >= float(qty_needed):
                return ing
        return None

    async def _compat_score(self, ing_a_id: int, ing_b_id: int) -> Optional[int]:
        res = await self.db.execute(
            select(Compatibility).where(
                and_(
                    (
                        (Compatibility.ing1_id == ing_a_id) & (Compatibility.ing2_id == ing_b_id)
                    )
                    | (
                        (Compatibility.ing1_id == ing_b_id) & (Compatibility.ing2_id == ing_a_id)
                    )
                )
            ).order_by(Compatibility.score.desc()).limit(1)
        )
        comp = res.scalars().first()
        return int(comp.score) if comp else None

    async def generate(self, prefs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1) База: ищем ингредиент по названию, потом fallback на любой base.
        base_name = str(prefs.get("base", "вода"))
        base = await self._get_ingredient_by_name(base_name)
        if base and self._is_avoided(base.name, prefs):
            base = None
        if not base:
            base_candidates = await self._get_ingredients_by_category("base")
            base_candidates = [x for x in base_candidates if not self._is_avoided(x.name, prefs)]
            base = random.choice(base_candidates) if base_candidates else None

        if not base:
            return None

        base_qty = self._qty_for_unit(base.unit or "ml", "base")
        if float(base.quantity) < base_qty:
            # Ищем другую base с достаточным остатком
            base_candidates = await self._get_ingredients_by_category("base")
            base2 = self._find_ingredient_with_enough(base_candidates, "base", base_qty)
            if not base2:
                return None
            base = base2

        # 2) Основной вкус: выбираем категорию по prefs.
        main_category = await self._choose_main_category(prefs)
        main = await self._choose_main_ingredient(main_category, prefs)
        if not main:
            return None

        main_qty = self._qty_for_unit(main.unit or "ml", "main")
        if float(main.quantity) < main_qty:
            # Пробуем выбрать другой main с достаточным остатком
            main_candidates = await self._get_ingredients_by_category(main_category)
            main2 = self._find_ingredient_with_enough(main_candidates, "main", main_qty)
            if not main2:
                return None
            main = main2

        # 3) Дополнения: берём 0..2 ингредиента, которые совместимы с main.
        extras = await self._get_extras(main, prefs)

        # 4) Формируем рецепт. JSON: ключи сделаем строками.
        recipe: Dict[str, float] = {
            str(base.id): float(base_qty),
            str(main.id): float(main_qty),
        }
        for ing, qty in extras:
            recipe[str(ing.id)] = float(qty)

        # 5) Проверяем, что остатков достаточно для всех ингредиентов.
        for ing_id_str, qty in recipe.items():
            ing_id = int(ing_id_str)
            ing = await self._get_ingredient_by_id(ing_id)
            if not ing or float(ing.quantity) < float(qty):
                return None

        # 6) Добавляем лёд, если пользователь не попросил без льда.
        if bool(prefs.get("ice", True)):
            ice = await self._find_first_available_by_category("ice")
            if ice:
                ice_qty = 120.0
                if float(ice.quantity) >= ice_qty:
                    recipe[str(ice.id)] = ice_qty

        name = self._generate_name(main, extras)
        description = self._generate_description(prefs, main)
        details = await self.build_recipe_details(recipe)
        totals = self.calculate_totals(details)
        taste_profile = self.calculate_taste_profile(prefs, details)

        ingredients_details = [
            {"id": int(k), "qty": v} for k, v in recipe.items()
        ]

        return {
            "name": name,
            "description": description,
            "recipe": recipe,
            "ingredients_details": ingredients_details,
            "details": details,
            "totals": totals,
            "taste_profile": taste_profile,
        }

    async def _choose_main_category(self, prefs: Dict[str, Any]) -> str:
        citrus = float(prefs.get("citrus", 0.0))
        fruitiness = float(prefs.get("fruitiness", 0.0))
        sweetness = float(prefs.get("sweetness", 0.0))
        sourness = float(prefs.get("sourness", 0.0))

        if citrus >= 0.6 or sourness >= 0.6:
            return "juice"
        if fruitiness >= 0.6:
            return "fruit"
        if sweetness >= 0.75:
            return "syrup"
        return "juice"

    def _is_avoided(self, ingredient_name: str, prefs: Dict[str, Any]) -> bool:
        avoid_terms = [str(x).lower() for x in prefs.get("avoid_terms", [])]
        name = ingredient_name.lower()
        return any(term and term in name for term in avoid_terms)

    def _preference_score(self, ingredient: Ingredient, prefs: Dict[str, Any]) -> int:
        """
        Чем выше score, тем больше ингредиент соответствует явным пожеланиям клиента.
        """
        score = 0
        name = (ingredient.name or "").lower()
        desired_terms = [str(x).lower() for x in prefs.get("desired_terms", [])]
        if desired_terms:
            for t in desired_terms:
                if t and t in name:
                    score += 5

        # Вкусовой профиль -> категории
        sweetness = float(prefs.get("sweetness", 0.5))
        sourness = float(prefs.get("sourness", 0.3))
        fruitiness = float(prefs.get("fruitiness", 0.5))
        citrus = float(prefs.get("citrus", 0.2))
        spice = float(prefs.get("spice", 0.2))
        mint = float(prefs.get("mint", 0.0))

        if ingredient.category == "syrup":
            score += int(sweetness * 6)
        if ingredient.category == "juice":
            score += int((sourness + citrus + fruitiness) * 3)
        if ingredient.category == "fruit":
            score += int((fruitiness + citrus) * 4)
        if ingredient.category == "additive":
            score += int((spice + mint) * 4)

        # Точечные предпочтения
        if "мят" in name:
            score += int(mint * 8)
        if any(x in name for x in ["лимон", "лайм", "апельсин", "грейпфрут"]):
            score += int(citrus * 8)
        if any(x in name for x in ["имбир", "перец", "кориц", "кардамон"]):
            score += int(spice * 8)
        return score

    async def _choose_main_ingredient(self, category: str, prefs: Dict[str, Any]) -> Optional[Ingredient]:
        candidates = await self._get_ingredients_by_category(category)
        if not candidates:
            return None
        candidates = [
            c for c in candidates
            if float(c.quantity or 0) > 0 and not self._is_avoided(c.name, prefs)
        ]
        if not candidates:
            return None
        scored = sorted(candidates, key=lambda c: self._preference_score(c, prefs), reverse=True)
        top = scored[:5]
        return random.choice(top)

    async def _choose_extras_candidates(self) -> List[Ingredient]:
        # Дополнительные категории для безалкогольного бара.
        candidates_categories = ["syrup", "additive", "fruit", "juice"]
        res = await self.db.execute(
            select(Ingredient).where(Ingredient.category.in_(candidates_categories))
        )
        return list(res.scalars().all())

    def _category_compat_score(self, main_category: str, extra_category: str) -> int:
        # fallback-правила, если пары нет в таблице compatibility
        matrix = {
            ("juice", "syrup"): 2,
            ("juice", "fruit"): 2,
            ("juice", "additive"): 1,
            ("fruit", "syrup"): 2,
            ("fruit", "additive"): 1,
            ("syrup", "juice"): 2,
            ("syrup", "fruit"): 1,
            ("syrup", "additive"): 1,
        }
        return matrix.get((main_category, extra_category), 0)

    async def _get_extras(self, main_ing: Ingredient, prefs: Dict[str, Any]) -> List[Any]:
        candidates = await self._choose_extras_candidates()

        scored: List[Any] = []
        # Перемешаем, чтобы каждый диалог давал разный результат.
        random.shuffle(candidates)

        for ing in candidates:
            if ing.id == main_ing.id:
                continue
            if self._is_avoided(ing.name, prefs):
                continue
            score = await self._compat_score(main_ing.id, ing.id)
            if score is None:
                score = self._category_compat_score(main_ing.category, ing.category)
            if score <= 0:
                continue

            qty = self._qty_for_unit(ing.unit or "ml", role="extra")
            if float(ing.quantity) < float(qty):
                continue
            pref_bonus = self._preference_score(ing, prefs)
            scored.append((score * 10 + pref_bonus, ing, qty))

        # Сначала берём наиболее совместимые (score=2 лучше).
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(ing, qty) for _, ing, qty in scored[:2]]

    async def _find_first_available_by_category(self, category: str) -> Optional[Ingredient]:
        candidates = await self._get_ingredients_by_category(category)
        available = [x for x in candidates if float(x.quantity or 0.0) > 0]
        return available[0] if available else None

    async def _get_base_ingredient(self, base_type: str) -> Optional[Ingredient]:
        return await self._get_ingredient_by_name(base_type)

    async def _get_ingredient_by_name(self, name: str) -> Optional[Ingredient]:
        # name может совпасть точно (обычно из base_map), поэтому используем ilike для мягкости.
        res = await self.db.execute(
            select(Ingredient).where(Ingredient.name.ilike(name))
        )
        return res.scalar_one_or_none()

    async def _get_ingredient_by_id(self, ing_id: int) -> Optional[Ingredient]:
        res = await self.db.execute(select(Ingredient).where(Ingredient.id == ing_id))
        return res.scalar_one_or_none()

    async def find_ingredient_by_name_like(self, query: str) -> Optional[Ingredient]:
        q = query.strip().lower()
        if not q:
            return None
        res = await self.db.execute(
            select(Ingredient).where(Ingredient.name.ilike(f"%{q}%")).limit(1)
        )
        return res.scalar_one_or_none()

    async def build_recipe_details(self, recipe: Dict[str, float]) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for ing_id_str, qty in recipe.items():
            ing = await self._get_ingredient_by_id(int(ing_id_str))
            if not ing:
                continue
            details.append(
                {
                    "id": ing.id,
                    "name": ing.name,
                    "category": ing.category,
                    "unit": ing.unit,
                    "qty": float(qty),
                }
            )
        return details

    def _nutrition_for_item(self, category: str, qty: float, unit: str) -> Dict[str, float]:
        base = self.nutrition_reference.get(category, self.nutrition_reference["additive"])
        if unit == "piece":
            factor = max(0.0, qty)
        else:
            factor = max(0.0, qty) / 100.0
        return {
            "kcal": round(base["kcal"] * factor, 2),
            "protein": round(base["protein"] * factor, 2),
            "fat": round(base["fat"] * factor, 2),
            "carbs": round(base["carbs"] * factor, 2),
        }

    def calculate_totals(self, details: List[Dict[str, Any]]) -> Dict[str, Any]:
        volume_without_ice_ml = 0.0
        ice_volume_ml = 0.0
        kcal = protein = fat = carbs = 0.0

        for item in details:
            qty = float(item["qty"])
            unit = item["unit"]
            category = item["category"]
            nutr = self._nutrition_for_item(category, qty, unit)
            kcal += nutr["kcal"]
            protein += nutr["protein"]
            fat += nutr["fat"]
            carbs += nutr["carbs"]

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

        volume_with_ice_ml = volume_without_ice_ml + ice_volume_ml
        return {
            "volume_without_ice_ml": round(volume_without_ice_ml, 1),
            "volume_with_ice_ml": round(volume_with_ice_ml, 1),
            "kcal": round(kcal, 1),
            "protein": round(protein, 1),
            "fat": round(fat, 1),
            "carbs": round(carbs, 1),
        }

    def calculate_taste_profile(self, prefs: Dict[str, Any], details: List[Dict[str, Any]]) -> Dict[str, int]:
        sweetness = min(10, max(1, int(round(float(prefs.get("sweetness", 0.5)) * 10))))
        sourness = min(10, max(1, int(round(float(prefs.get("sourness", 0.3)) * 10))))
        freshness = min(10, max(1, int(round((float(prefs.get("citrus", 0.2)) + float(prefs.get("mint", 0.0))) * 5 + 2))))
        spice = 2
        body = 4
        for item in details:
            name = item["name"].lower()
            cat = item["category"]
            if "имбир" in name or "перец" in name or "кориц" in name:
                spice += 2
            if cat == "syrup":
                body += 1
            if cat == "fruit":
                body += 1
            if "газирован" in name:
                freshness += 1
        return {
            "sweetness": min(10, max(1, sweetness)),
            "sourness": min(10, max(1, sourness)),
            "freshness": min(10, max(1, freshness)),
            "spice": min(10, max(1, spice)),
            "body": min(10, max(1, body)),
        }

    def format_recipe_card(self, cocktail_data: Dict[str, Any]) -> str:
        name = cocktail_data.get("name", "Авторский напиток")
        description = cocktail_data.get("description", "")
        details = cocktail_data.get("details", [])
        totals = cocktail_data.get("totals", {})
        taste = cocktail_data.get("taste_profile", {})

        lines = [
            f"🍹 {name}",
            description,
            "",
            "Профиль напитка (1-10):",
            f"- сладость: {taste.get('sweetness', 0)}",
            f"- кислотность: {taste.get('sourness', 0)}",
            f"- свежесть: {taste.get('freshness', 0)}",
            f"- пряность: {taste.get('spice', 0)}",
            f"- плотность вкуса: {taste.get('body', 0)}",
            "",
            "Объём 1 порции:",
            f"- без льда: {totals.get('volume_without_ice_ml', 0)} мл",
            f"- со льдом: {totals.get('volume_with_ice_ml', 0)} мл",
            "",
            "КБЖУ (на порцию):",
            f"- К: {totals.get('kcal', 0)} ккал",
            f"- Б: {totals.get('protein', 0)} г",
            f"- Ж: {totals.get('fat', 0)} г",
            f"- У: {totals.get('carbs', 0)} г",
            "",
            "Состав:",
        ]
        for item in details:
            lines.append(f"- {item['name']}: {item['qty']} {item['unit']}")

        lines.extend(
            [
                "",
                "Если хотите, я изменю состав: напишите, например:",
                "- 'добавь мяту'",
                "- 'убери сироп'",
                "- 'сделай менее сладким'",
                "- 'без льда'",
                "- 'объем 350 мл'",
                "Либо напишите 'подтвердить', чтобы оформить заказ.",
            ]
        )
        return "\n".join(lines)

    async def adjust_recipe(
        self,
        cocktail_data: Dict[str, Any],
        user_message: str,
        prefs: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = user_message.lower().strip()
        recipe = dict(cocktail_data.get("recipe", {}))

        # Без льда / добавить лёд
        if "без льда" in text:
            for item in list(cocktail_data.get("details", [])):
                if item.get("category") == "ice":
                    recipe.pop(str(item["id"]), None)
        elif "добавь лед" in text or "добавь лёд" in text:
            ice = await self._find_first_available_by_category("ice")
            if ice and float(ice.quantity) >= 80.0:
                recipe[str(ice.id)] = 120.0

        # Добавить ингредиент
        if text.startswith("добавь "):
            name = text.replace("добавь ", "", 1).strip()
            ing = await self.find_ingredient_by_name_like(name)
            if ing:
                qty = self._qty_for_unit(ing.unit or "ml", "extra")
                recipe[str(ing.id)] = float(recipe.get(str(ing.id), 0.0)) + qty

        # Убрать ингредиент
        if text.startswith("убери ") or text.startswith("удали "):
            marker = "убери " if text.startswith("убери ") else "удали "
            name = text.replace(marker, "", 1).strip()
            ing = await self.find_ingredient_by_name_like(name)
            if ing:
                recipe.pop(str(ing.id), None)

        # Управление сладостью (через сиропы)
        if "менее слад" in text or "меньше слад" in text:
            for item in list(recipe.keys()):
                ing = await self._get_ingredient_by_id(int(item))
                if ing and ing.category == "syrup":
                    recipe[item] = max(0.0, float(recipe[item]) - 10.0)
                    if recipe[item] <= 0.0:
                        recipe.pop(item, None)
            prefs["sweetness"] = max(0.1, float(prefs.get("sweetness", 0.5)) - 0.2)
        if "слаще" in text or "больше слад" in text:
            prefs["sweetness"] = min(1.0, float(prefs.get("sweetness", 0.5)) + 0.2)
            syrup = await self._find_first_available_by_category("syrup")
            if syrup:
                recipe[str(syrup.id)] = float(recipe.get(str(syrup.id), 0.0)) + 10.0

        # Масштабирование объема
        if "объем" in text or "объём" in text:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                target_volume = float(digits)
                details_now = await self.build_recipe_details(recipe)
                totals_now = self.calculate_totals(details_now)
                current = float(totals_now.get("volume_without_ice_ml", 0.0))
                if current > 0 and target_volume > 0:
                    factor = max(0.5, min(1.8, target_volume / current))
                    for key in list(recipe.keys()):
                        recipe[key] = round(float(recipe[key]) * factor, 1)

        # Проверка остатков
        for ing_id_str, qty in recipe.items():
            ing = await self._get_ingredient_by_id(int(ing_id_str))
            if not ing or float(ing.quantity) < float(qty):
                continue

        details = await self.build_recipe_details(recipe)
        totals = self.calculate_totals(details)
        taste_profile = self.calculate_taste_profile(prefs, details)
        cocktail_data["recipe"] = recipe
        cocktail_data["details"] = details
        cocktail_data["totals"] = totals
        cocktail_data["taste_profile"] = taste_profile
        return cocktail_data

    def _generate_name(self, main_ing: Ingredient, extras: List[Any]) -> str:
        # Формируем название из main + 1 доп. ингредиента (если есть).
        parts = [main_ing.name]
        if extras:
            parts.append(extras[0][0].name)
        return " ".join(parts) + " (AI)"

    def _generate_description(self, prefs: Dict[str, Any], main_ing: Ingredient) -> str:
        sweetness = float(prefs.get("sweetness", 0.0))
        sourness = float(prefs.get("sourness", 0.0))
        citrus = float(prefs.get("citrus", 0.0))
        mint = float(prefs.get("mint", 0.0))

        desc = f"Освежающий коктейль на основе {main_ing.name}."
        if sweetness >= 0.7:
            desc += " Сладкая нота."
        if sourness >= 0.6:
            desc += " Присутствует приятная кислинка."
        if citrus >= 0.6:
            desc += " Цитрусовая свежесть."
        if mint >= 0.7:
            desc += " Мятный оттенок."
        return desc

