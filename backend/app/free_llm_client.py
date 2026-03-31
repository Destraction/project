from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Compatibility, Ingredient


class FreeLLMBartender:
    """
    Бесплатная локальная модель через Ollama.
    """

    def __init__(self) -> None:
        self.enabled = bool(settings.ENABLE_OLLAMA)
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
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

    async def reply(
        self,
        db: AsyncSession,
        user_message: str,
        state: Dict[str, Any],
        fallback_text: str,
        strict_json: bool = False,
    ) -> str:
        technical_fallback = (
            "Локальная модель недоступна. Запусти Ollama и скачай модель, затем повтори сообщение."
        )
        if not self.enabled:
            return fallback_text or technical_fallback

        context = await self.build_bar_context(db)
        prefs = state.get("prefs", {})
        profile = state.get("profile", {})
        phase = state.get("phase", "discovery")
        allowed_set = self._extract_allowed_set(context)

        system_prompt = (
            "Ты бармен в безалкогольном баре. "
            "Говори по-русски, тепло и естественно, как приятель. "
            "Учитывай контекст доступных ингредиентов и сочетаемость. "
            "Не предлагай ингредиенты, которых нет в остатках. "
            "Задавай уточняющие вопросы по вкусу и настроению."
        )
        if strict_json:
            system_prompt += (
                " Ответь СТРОГО JSON: "
                "{\"reply\": \"...\", \"mentioned_ingredients\": [\"...\"]}."
            )

        user_payload = {
            "phase": phase,
            "user_message": user_message,
            "prefs": prefs,
            "profile": profile,
            "bar_context": context,
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                        ],
                        "options": {"temperature": 0.7},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = str(data.get("message", {}).get("content", "")).strip()
                if not content:
                    return fallback_text or technical_fallback
                if strict_json:
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        return fallback_text or technical_fallback
                    if not self._validate_json_payload(parsed, allowed_set):
                        return fallback_text or technical_fallback
                    return str(parsed.get("reply") or fallback_text)
                return content
        except Exception:
            return fallback_text or technical_fallback

