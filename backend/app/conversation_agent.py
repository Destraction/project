from __future__ import annotations

import random
from typing import Any, Dict, List


class RussianBartenderAgent:
    """
    Небольшой диалоговый движок в стиле "бармен-гость":
    - живые формулировки
    - вариативные вопросы
    - мягкие подводки к уточнению предпочтений
    """

    def __init__(self) -> None:
        self.greetings = [
            "Добро пожаловать! Я ваш AI-бармен. Давайте соберем напиток именно под ваш вкус.",
            "Рад видеть вас у барной стойки. Я помогу собрать идеальный безалкогольный коктейль.",
            "Здравствуйте! Я виртуальный бармен. Расскажите, какой напиток хочется именно сейчас?",
        ]
        self.ack_templates = [
            "Отлично, понял вас.",
            "Хороший ориентир, беру в работу.",
            "Принял, уже подстраиваю рецепт.",
            "Супер, это помогает точнее попасть во вкус.",
        ]
        self.discovery_questions: Dict[str, List[str]] = {
            "mood": [
                "Какое сегодня настроение у напитка: расслабляющее, бодрящее или ярко-праздничное?",
                "Хотите что-то спокойное и мягкое или более бодрое и акцентное?",
                "Под какой момент подбираем: неспешный вечер, работа, встреча с друзьями?",
            ],
            "base": [
                "Основа ближе на обычной воде или на газированной?",
                "Любите легкое шипение в напитке или предпочитаете негазированный вариант?",
            ],
            "taste": [
                "По балансу куда смещаемся: больше сладости, кислинки, свежести или пряности?",
                "Если представить шкалу 1-10, какую сладость и кислотность хотите на выходе?",
                "Хотите более фруктовый профиль или травяной/цитрусовый характер?",
            ],
            "constraints": [
                "Есть ингредиенты, которые точно исключаем? Например: имбирь, мята, цитрус, сиропы.",
                "Нужен ли лед и какой объем порции вам комфортен: 250, 300 или 350 мл?",
                "Есть пожелания по калорийности: полегче или можно насыщеннее по вкусу?",
            ],
        }
        self.adjust_prompts = [
            "Могу сразу подстроить рецепт под вас: добавить/убрать ингредиенты, изменить сладость, лед и объем.",
            "Если хотите, доведем напиток до идеала: скажите, что усилить или ослабить.",
            "Перед подтверждением можем точечно подправить состав под ваш вкус.",
        ]

    def greet(self) -> str:
        return random.choice(self.greetings)

    def ack(self) -> str:
        return random.choice(self.ack_templates)

    def next_question(self, state: Dict[str, Any]) -> str:
        collected = state.get("collected", {})
        if not collected.get("mood", False):
            return random.choice(self.discovery_questions["mood"])
        if not collected.get("base", False):
            return random.choice(self.discovery_questions["base"])
        if not collected.get("taste", False):
            return random.choice(self.discovery_questions["taste"])
        return random.choice(self.discovery_questions["constraints"])

    def draft_intro(self) -> str:
        return (
            "Я собрал первый вариант. Посмотрите профиль и состав — и скажите, что подправить перед подтверждением."
        )

    def adjustment_hint(self) -> str:
        return random.choice(self.adjust_prompts)

