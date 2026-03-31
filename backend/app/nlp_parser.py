from __future__ import annotations

import re
from typing import Dict, Any, List


class PreferenceParser:
    """
    Простейший rule-based парсер предпочтений.

    Важно: база в БД задаётся названиями ингредиентов (например, "вода", "лимонад"),
    поэтому prefs["base"] тоже будет содержать именно такие значения.
    """

    def __init__(self) -> None:
        self.stopwords = {
            "и", "в", "во", "на", "с", "со", "к", "по", "из", "для", "а", "но", "или", "же",
            "не", "ни", "я", "мы", "мне", "мои", "мой", "хочу", "хотел", "хотелось", "будет",
            "можно", "дайте", "нужно", "пожалуйста",
        }
        self.common_suffixes = [
            "иями", "ями", "ами", "иями", "ого", "ему", "ому", "ими", "ыми", "ее", "ие", "ые",
            "ая", "яя", "ой", "ый", "ий", "ам", "ям", "ах", "ях", "ом", "ем", "у", "ю", "а",
            "я", "ы", "и", "е", "о",
        ]

        # Ключевые слова и их вес (чем больше — тем сильнее предпочтение)
        self.keywords: Dict[str, Dict[str, float]] = {
            "sweetness": {
                "сладкий": 0.9,
                "сладко": 0.8,
                "сахар": 0.7,
                "сироп": 0.6,
                "медовый": 0.6,
            },
            "sourness": {
                "кислый": 0.9,
                "кислин": 0.8,
                "кислинка": 0.8,
                "кислот": 0.7,
                "лимонный": 0.7,
                "лимон": 0.7,
                "лайм": 0.7,
            },
            "fruitiness": {
                "фруктовый": 0.9,
                "ягодный": 0.8,
                "яблоко": 0.7,
                "яблочный": 0.7,
                "яблочный сок": 0.9,
                "клубнич": 0.85,
                "персик": 0.7,
                "манго": 0.7,
            },
            "citrus": {
                "цитрус": 0.8,
                "лимон": 0.9,
                "лайм": 0.9,
                "апельсин": 0.8,
                "грейпфрут": 0.7,
            },
            "mint": {
                "мята": 0.9,
                "мятный": 0.9,
                "перечная мята": 0.8,
            },
        }

        # Явные основы (названия ингредиентов из seed_data.py)
        self.base_map: Dict[str, str] = {
            "вода": "вода",
            "газирован": "газированная вода",
            "сод": "газированная вода",
            "лед": "вода",
            "сок": "яблочный сок",  # "сок" -> типовой сок для примера
        }

        self.default_prefs: Dict[str, Any] = {
            "sweetness": 0.5,
            "sourness": 0.3,
            "fruitiness": 0.5,
            "citrus": 0.2,
            "mint": 0.0,
            "spice": 0.2,
            "base": "вода",
            "desired_terms": [],
            "avoid_terms": [],
            "_explicit_fields": [],
        }

    def _normalize_token(self, token: str) -> str:
        t = token.lower().strip()
        if len(t) <= 3:
            return t
        for suf in self.common_suffixes:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t

    def _tokenize(self, text: str) -> List[str]:
        raw = re.findall(r"[а-яa-z0-9ё]+", text.lower())
        out: List[str] = []
        for token in raw:
            if token in self.stopwords:
                continue
            out.append(self._normalize_token(token))
        return out

    def _contains_meaning(self, normalized_tokens: List[str], phrase: str) -> bool:
        parts = [self._normalize_token(x) for x in re.findall(r"[а-яa-z0-9ё]+", phrase.lower())]
        if not parts:
            return False
        token_set = set(normalized_tokens)
        return all(p in token_set for p in parts)

    def parse(self, text: str) -> Dict[str, Any]:
        text_norm = text.lower()
        prefs = dict(self.default_prefs)
        normalized_tokens = self._tokenize(text_norm)
        prefs["desired_terms"] = []
        prefs["avoid_terms"] = []
        prefs["_explicit_fields"] = []

        # База: ищем по ключевым словам/подстрокам
        for needle, base_value in self.base_map.items():
            if self._contains_meaning(normalized_tokens, needle) or needle in text_norm:
                prefs["base"] = base_value
                prefs["_explicit_fields"].append("base")
                break

        # Весовые признаки по ключевым словам (используем подстрочный поиск)
        for feature, words in self.keywords.items():
            best = prefs.get(feature, 0.0)
            for word, weight in words.items():
                if self._contains_meaning(normalized_tokens, word) or word in text_norm:
                    best = max(best, weight)
            prefs[feature] = best
            if best > self.default_prefs.get(feature, 0.0):
                prefs["_explicit_fields"].append(feature)

        # Пряность
        spice_markers = ["пряный", "прян", "имбир", "перец", "кориц", "кардамон"]
        spice_hit = any(self._contains_meaning(normalized_tokens, m) or m in text_norm for m in spice_markers)
        if spice_hit:
            prefs["spice"] = max(float(prefs.get("spice", 0.2)), 0.75)
            prefs["_explicit_fields"].append("spice")

        # Явные шкалы 1-10: "сладость 7/10", "кислотность 3/10", "7/10"
        def clamp10(v: int) -> float:
            return max(0.0, min(1.0, v / 10.0))

        # Пары формата "сладость 7/10" / "кислотность 4/10"
        sweet_match = re.search(r"(слад\w*)\D{0,8}(\d{1,2})\s*/\s*10", text_norm)
        sour_match = re.search(r"(кисл\w*|кислот\w*)\D{0,8}(\d{1,2})\s*/\s*10", text_norm)
        if sweet_match:
            prefs["sweetness"] = clamp10(int(sweet_match.group(2)))
            prefs["_explicit_fields"].append("sweetness")
        if sour_match:
            prefs["sourness"] = clamp10(int(sour_match.group(2)))
            prefs["_explicit_fields"].append("sourness")

        # Если просто "7/10" без уточнения — считаем это общей интенсивностью:
        plain_scale = re.findall(r"(\d{1,2})\s*/\s*10", text_norm)
        if plain_scale:
            val = clamp10(int(plain_scale[0]))
            if "sweetness" not in prefs["_explicit_fields"]:
                prefs["sweetness"] = val
                prefs["_explicit_fields"].append("sweetness")
            if "sourness" not in prefs["_explicit_fields"]:
                prefs["sourness"] = val
                prefs["_explicit_fields"].append("sourness")

        # Извлекаем явные ограничения "без X"
        for m in re.findall(r"без\s+([а-яa-zё0-9\s\-]{2,30})", text_norm):
            part = m.strip().split(",")[0].split(".")[0].strip()
            if part:
                prefs["avoid_terms"].append(self._normalize_token(part))

        # Термины предпочтений для прямого матчинга по названиям ингредиентов
        generic = {
            "сладк", "кисл", "фрукт", "цитрус", "прян", "настроен", "напит", "коктейл",
            "вкус", "легк", "сильн", "меньш", "больш", "хоч", "хотел", "пожалуйст",
        }
        for tok in normalized_tokens:
            if len(tok) < 4:
                continue
            if tok in generic:
                continue
            prefs["desired_terms"].append(tok)

        # Дедуп
        prefs["desired_terms"] = list(dict.fromkeys(prefs["desired_terms"]))
        prefs["avoid_terms"] = list(dict.fromkeys(prefs["avoid_terms"]))
        prefs["_explicit_fields"] = list(dict.fromkeys(prefs["_explicit_fields"]))

        # Небольшая нормализация
        prefs["sweetness"] = float(prefs["sweetness"])
        prefs["sourness"] = float(prefs["sourness"])
        prefs["fruitiness"] = float(prefs["fruitiness"])
        prefs["citrus"] = float(prefs["citrus"])
        prefs["mint"] = float(prefs["mint"])
        prefs["spice"] = float(prefs["spice"])

        return prefs
