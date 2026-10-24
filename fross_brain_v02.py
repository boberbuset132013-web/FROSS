#!/usr/bin/env python3
"""FROSS Local Brain 0.2 — Python reference implementation.

The deployed Cloudflare Worker runs the matching JavaScript implementation
because Cloudflare Workers do not execute ordinary Python modules. This
engine uses explicit rules, a safe arithmetic parser, and search-snippet
ranking. It does not call a hosted language-model API or use eval/exec.
"""
from __future__ import annotations
import re
from typing import Any

VERSION = "FROSS Local Brain 0.2"

DEFINITIONS: list[tuple[tuple[str, ...], str]] = [
    (("html", "хтмл"), "HTML описывает структуру веб-страницы. CSS отвечает за оформление, а JavaScript — за поведение."),
    (("css",), "CSS — язык оформления веб-страниц: цвета, шрифты, отступы, сетки, адаптивность и анимации."),
    (("javascript", "джаваскрипт"), "JavaScript — язык программирования для интерактивности страниц, обработки событий и сетевых запросов."),
    (("алгоритм",), "Алгоритм — последовательность шагов для решения задачи. Хороший алгоритм определяет входные данные, порядок действий и ожидаемый результат."),
    (("переменная",), "Переменная — именованное значение, которое программа может читать и изменять."),
    (("цикл", "циклы"), "Цикл повторяет действия, пока выполняется условие или пока не закончится последовательность."),
    (("функция",), "Функция — отдельный блок кода для конкретной задачи. Она помогает повторно использовать логику и делить программу на части."),
    (("массив",), 'Массив — упорядоченный набор значений. В Python: names = ["Аня", "Илья"]. Индексы начинаются с нуля.'),
    (("интернет",), "Интернет — глобальная сеть взаимосвязанных сетей. Веб — только одна из служб, использующих интернет."),
    (("искусственный интеллект",), "Искусственный интеллект — общее название методов для решения задач распознавания, поиска закономерностей, планирования и обработки языка. Это может быть нейросеть или система правил."),
    (("нейросеть",), "Нейросеть — вычислительная модель с обучаемыми параметрами, которая находит закономерности на примерах."),
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()

def clean_intent(text: str) -> tuple[str, str]:
    q = re.sub(r"[!?.,;:()\[\]{}]+", " ", normalize(text))
    q = re.sub(r"\s+", " ", q).strip()
    stripped = re.sub(r"(fross|фросс|ботик|бот)", " ", q, flags=re.I)
    return q, re.sub(r"\s+", " ", stripped).strip()

class MathParser:
    """Recursive-descent parser for arithmetic only; never executes Python code."""
    def __init__(self, expression: str) -> None:
        self.tokens = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+|[()+\-*/%^])", expression)
        if not self.tokens or "".join(self.tokens) != re.sub(r"\s+", "", expression):
            raise ValueError("invalid expression")
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        if self.i >= len(self.tokens):
            raise ValueError("unexpected end")
        token = self.tokens[self.i]
        self.i += 1
        return token

    def primary(self) -> float:
        token = self.take()
        if token == "(":
            value = self.add_sub()
            if self.take() != ")":
                raise ValueError("missing parenthesis")
            return value
        if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", token):
            raise ValueError("expected number")
        return float(token)

    def power(self) -> float:
        left = self.primary()
        if self.peek() == "^":
            self.take()
            right = self.unary()
            if abs(right) > 100:
                raise ValueError("exponent too large")
            return left ** right
        return left

    def unary(self) -> float:
        if self.peek() == "+":
            self.take()
            return self.unary()
        if self.peek() == "-":
            self.take()
            return -self.unary()
        return self.power()

    @staticmethod
    def check(value: float) -> float:
        if value != value or abs(value) > 1e100:
            raise ValueError("result outside safe range")
        return value

    def mul_div(self) -> float:
        value = self.unary()
        while self.peek() in ("*", "/", "%"):
            op = self.take()
            right = self.unary()
            if op in ("/", "%") and right == 0:
                raise ZeroDivisionError("division by zero")
            value = value * right if op == "*" else value / right if op == "/" else value % right
            value = self.check(value)
        return value

    def add_sub(self) -> float:
        value = self.mul_div()
        while self.peek() in ("+", "-"):
            op = self.take()
            right = self.mul_div()
            value = value + right if op == "+" else value - right
            value = self.check(value)
        return value

    def parse(self) -> float:
        value = self.add_sub()
        if self.i != len(self.tokens):
            raise ValueError("unexpected token")
        return round(value, 12)

def parse_math(text: str) -> tuple[str, float] | None:
    expr = normalize(text)
    expr = re.sub(r"^(сколько будет|сколько получится|посчитай|вычисли|реши|calculate)\s*", "", expr)
    expr = re.sub(r"умножить на|умножить|×", "*", expr)
    expr = re.sub(r"разделить на|делить на|÷", "/", expr)
    expr = re.sub(r"\bплюс\b", "+", expr)
    expr = re.sub(r"\bминус\b", "-", expr)
    expr = re.sub(r"\bв степени\b", "^", expr)
    expr = re.sub(r"[?!;:=]+\s*$", "", expr).replace(",", ".").strip()
    if not expr or len(expr) > 120 or not re.fullmatch(r"[\d\s().+\-*/%^]+", expr):
        return None
    try:
        return expr, MathParser(expr).parse()
    except ZeroDivisionError:
        return expr, float("nan")
    except (ValueError, OverflowError):
        return None

def quick_reply(message: str, is_creator: bool = False, mode: str = "normal") -> str | None:
    q, stripped = clean_intent(message)
    if re.fullmatch(r"(эй\s+)?(привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый вечер)", stripped):
        return "Привет, создатель FROSS! Что будем разбирать сегодня?" if is_creator else "Привет! Я на связи. Что разберём?"
    if stripped in {"спасибо", "благодарю", "спс", "пасиб"}:
        return "Пожалуйста! Что делаем дальше?"
    if stripped in {"пока", "до свидания", "увидимся"}:
        return "До встречи! Возвращайся, когда понадобится помощь."
    if stripped in {"как дела", "что нового", "ты здесь", "ты тут", "ты на связи", "ау"}:
        return "Я на связи. Могу посчитать выражение, объяснить базовую тему или найти материалы."
    if stripped in {"кто ты", "ты кто", "как тебя зовут", "что ты такое"}:
        return f"Я {VERSION} — самописный движок на правилах, безопасном калькуляторе, шаблонах и поисковой сводке. Это не готовая языковая модель."
    if re.search(r"(кто создал fross|кто создатель fross|кто автор fross|кто тебя создал)", q):
        return "Автор проекта FROSS: https://t.me/Spamton_BigShot_hahaha"
    if re.search(r"(моя почта|мой email|какая у меня почта|назови мой email|мой адрес электронной почты|почт[ауеы].{0,35}(автор|создател|владельц))", q):
        return "Я могу распознать подтверждённый аккаунт создателя, но не вывожу личную почту в сообщениях. Публичный контакт: @Spamton_BigShot_hahaha."
    if re.search(r"(что ты умеешь|твои возможности|помощь|help)[\s?!.,]*$", q):
        return "Я умею считать выражения, объяснять несколько базовых технических понятий, выдавать некоторые шаблоны кода и составлять сводки найденных страниц. Свободные ответы на любую тему пока ограничены."
    if re.search(r"(что такое fross local brain|что за версия мышления|версия твоего мышления|какая у тебя версия)", q):
        return f"Сейчас у меня {VERSION}. Это собственный алгоритмический движок, а не обученная нейросеть."

    math = parse_math(message)
    if math:
        expr, value = math
        if value != value:
            return "Делить на ноль нельзя — результат не определён."
        result = f"{value:g}"
        if mode == "reasoning":
            return f"Результат: {result}\n\nПроверка: локальный парсер учитывает скобки, степени, умножение/деление и затем сложение/вычитание."
        return f"Ответ: {result}. Вычислено локально."

    if re.search(r"(что такое|объясни|расскажи|определение|что означает|что значит)", q):
        for keys, answer in DEFINITIONS:
            if any(key in q for key in keys):
                return answer + ("\n\nЭто базовое определение; можно уточнить пример." if mode == "reasoning" else "")

    if re.search(r"(с чего начать|план для|составь план|помоги спланировать|как разработать|как создать)", q):
        if re.search(r"(сайт|веб|страниц)", q):
            return ("План разработки сайта:\n\n1. Определить назначение и функции.\n"
                    "2. Спроектировать экраны и состояния.\n"
                    "3. Сделать адаптивную разметку и стили.\n"
                    "4. Реализовать логику и обработку ошибок.\n"
                    "5. Проверить разные экраны и опубликовать.")
        return ("План задачи:\n\n1. Сформулировать желаемый результат.\n"
                "2. Определить входные данные и ограничения.\n"
                "3. Разбить работу на небольшие шаги.\n"
                "4. Сделать минимальный прототип.\n"
                "5. Проверить его на примерах и исправить ошибки.")
    return None

def think(message: str, history: list[dict[str, Any]] | None = None,
          results: list[dict[str, Any]] | None = None,
          mode: str = "normal", is_creator: bool = False) -> dict[str, Any]:
    history = history or []
    results = results or []
    direct = quick_reply(message, is_creator=is_creator, mode=mode)
    if direct:
        return {"reply": direct, "kind": "local-rule-v02",
                "trace": ["Нормализовать запрос", "Распознать намерение", "Ответить локальным правилом"], "sources": []}

    words = {w for w in re.findall(r"[a-zа-яё0-9]{2,}", normalize(message))
             if w not in {"что", "как", "где", "когда", "для", "это", "the", "and", "with"}}
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        title = normalize(str(item.get("title", "")))
        text = normalize(title + " " + str(item.get("snippet", "")))
        title_hits = sum(word in title for word in words)
        hits = sum(word in text for word in words)
        overlap = hits / len(words) if words else 0.0
        title_overlap = title_hits / len(words) if words else 0.0
        score = min(1.0, float(item.get("score", 0) or 0) * 0.55 + overlap * 0.35 + title_overlap * 0.2)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    seen: set[str] = set()
    unique: list[tuple[float, dict[str, Any]]] = []
    for score, item in ranked:
        key = str(item.get("url") or item.get("title") or "").rstrip("/").lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append((score, item))
    relevant = [pair for pair in unique if pair[0] >= 0.08]
    selected = (relevant or unique)[:4 if mode == "reasoning" else 3]
    if selected:
        lines = ["Я использовал локальный алгоритм 0.2: сравнил слова запроса с заголовками и фрагментами." if mode == "reasoning"
                 else "Вот наиболее подходящие найденные материалы."]
        for index, (_, item) in enumerate(selected, 1):
            lines.append(f"{index}. **{item.get('title') or 'Без заголовка'}**\n{str(item.get('snippet') or 'Описание отсутствует.')[:480]}")
        lines.append("Результаты ранжированы по словесным совпадениям; это не независимая проверка полных страниц.")
        lines.append("\nИсточники:")
        for _, item in selected:
            url = str(item.get("url", ""))
            if url.startswith(("https://", "http://")):
                lines.append(f"- [{item.get('title') or 'Открыть источник'}]({url})")
        trace = (["Нормализовать запрос", "Выделить ключевые слова", "Сравнить материалы", "Оценить релевантность",
                  "Убрать дубликаты", "Проверить ограничения", "Сформировать ответ"] if mode == "reasoning"
                 else ["Нормализовать запрос", "Оценить релевантность", "Составить сводку", "Добавить источники"])
        return {"reply": "\n".join(lines), "kind": "search-summary-v02", "trace": trace,
                "sources": [item for _, item in selected]}

    last = next((item.get("content", "") for item in reversed(history)
                 if item.get("role") == "assistant" and isinstance(item.get("content"), str)), "")
    if re.search(r"\b(а почему|почему|а как|а если|расскажи подробнее|продолжи)\b", normalize(message)) and last:
        return {"reply": "Похоже, это уточнение к предыдущей теме, но я не понял к какой детали. Укажи фразу, которую нужно разобрать.",
                "kind": "context-fallback-v02", "trace": ["Проверить контекст", "Найти неоднозначность", "Попросить уточнить"], "sources": []}
    return {"reply": "Я не нашёл достаточно надёжных данных и не хочу выдумывать ответ. Мой движок 0.2 основан на локальных правилах и поисковых фрагментах; уточни запрос, и я попробую точнее.",
            "kind": "no-match-v02", "trace": ["Нормализовать запрос", "Проверить правила", "Обнаружить недостаток данных", "Сообщить об ограничении"], "sources": []}


if __name__ == "__main__":
    import unittest

    class BrainTests(unittest.TestCase):
        def test_greeting_with_bot_name(self):
            self.assertIn("Привет!", quick_reply("привет фросс") or "")
        def test_creator_greeting(self):
            self.assertIn("создатель FROSS", quick_reply("Привет, ФРОСС!", is_creator=True) or "")
        def test_math_precedence(self):
            self.assertEqual(parse_math("сколько будет (2+3)*4"), ("(2+3)*4", 20.0))
        def test_power(self):
            self.assertEqual(parse_math("2^8"), ("2^8", 256.0))
        def test_divide_by_zero(self):
            self.assertIn("Делить на ноль", quick_reply("10/0") or "")
        def test_search_summary(self):
            result = think("как изучать Python", results=[{"title":"Python guide","snippet":"Python basics","url":"https://example.com","score":0.2}])
            self.assertIn("Python guide", result["reply"])

    unittest.main()
