#!/usr/bin/env python3
"""FROSS Local Brain 0.6 — Python reference implementation.

The deployed Cloudflare Worker runs the matching JavaScript implementation
because Cloudflare Workers do not execute ordinary Python modules. This
engine uses explicit rules, a safe arithmetic parser, and search-snippet
ranking. It does not call a hosted language-model API or use eval/exec.
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any

VERSION = "FROSS Local Brain 0.6"

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
    (("python", "питон"), "Python — язык программирования с читаемым синтаксисом. Его часто используют для автоматизации, анализа данных, веб-сервисов и обучения машинному обучению."),
    (("c++", "си плюс плюс"), "C++ — компилируемый язык программирования с контролем ресурсов и высокой производительностью. Его используют в системном ПО, игровых движках и приложениях."),
    (("база данных", "базы данных"), "База данных — организованное хранилище, в котором можно создавать, находить и изменять записи. В реляционных базах данные часто представлены таблицами."),
    (("api",), "API — описанный способ взаимодействия программ. Веб-API обычно принимает HTTP-запросы и возвращает данные, например в формате JSON."),
    (("http", "https"), "HTTP — протокол передачи данных в вебе. HTTPS использует TLS для шифрования соединения и проверки подлинности сервера."),
    (("dns",), "DNS сопоставляет доменные имена с сетевой информацией, например IP-адресами, чтобы браузер мог найти сервер."),
    (("процессор", "cpu"), "CPU — центральный процессор, который выполняет инструкции программ. Скорость зависит не только от частоты, но и от архитектуры и характера задачи."),
    (("оперативная память", "ram"), "RAM — оперативная память для временного хранения данных, с которыми активно работает компьютер. Обычно её содержимое очищается при выключении питания."),
    (("видеокарта", "gpu"), "GPU — графический процессор, приспособленный к большому числу параллельных вычислений. Он используется для графики и ряда вычислительных задач."),
    (("firestore",), "Cloud Firestore — облачная документная база данных Firebase. Доступ к документам следует ограничивать правилами, которые проверяют пользователя и владельца данных."),
    (("cloudflare",), "Cloudflare предоставляет сетевые сервисы, CDN, DNS, защиту сайтов и Workers — среду для запуска серверного JavaScript."),
    (("git",), "Git — система контроля версий, которая хранит историю изменений, помогает сравнивать версии файлов и работать в ветках."),
    (("github",), "GitHub — платформа для Git-репозиториев, совместной работы, обсуждений кода и автоматизации разработки."),
    (("пароль", "пароли"), "Пароль помогает защитить аккаунт. Используй уникальные пароли, не публикуй их и включай дополнительную проверку входа, когда она доступна."),
    (("компилятор",), "Компилятор преобразует исходный код в другую форму, например машинный код или промежуточное представление, которое сможет выполнить среда запуска."),
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()


TYPO_ALIASES = {
    "созрдайтель": "создатель", "создатиль": "создатель", "создител": "создатель",
    "создательь": "создатель", "разроботчик": "разработчик", "разработчк": "разработчик",
    "привте": "привет", "привит": "привет", "превет": "привет", "приве": "привет",
    "фрос": "фросс", "фроссс": "фросс", "фросса": "фросс",
    "памаги": "помоги", "помаги": "помоги", "пачему": "почему", "почиму": "почему",
    "сколька": "сколько", "скока": "сколько", "обесни": "объясни", "обьясни": "объясни",
    "програма": "программа", "програмирование": "программирование",
    "интелект": "интеллект", "алгаритм": "алгоритм", "неправельно": "неправильно",
}
TYPO_VOCABULARY = [
    "создатель", "создателя", "создателем", "создатели", "создали", "создал", "автор",
    "разработчик", "разработчика", "владелец", "сделал", "привет", "здравствуй",
    "здравствуйте", "фросс", "почему", "помоги", "объясни", "расскажи", "посчитать",
    "вычисли", "математика", "пример", "программа", "программирование", "переменная",
    "функция", "массив", "алгоритм", "нейросеть", "искусственный", "интеллект",
    "ошибка", "исправить", "неправильно", "сообщение", "приветствие", "вопрос",
    "поиск", "источник", "python", "javascript", "html", "css", "сайт", "чат",
    "удалить", "настройка", "объяснение", "версия", "мышление", "умеешь", "умеет",
]


def correct_word(word: str) -> str:
    if word in TYPO_ALIASES:
        return TYPO_ALIASES[word]
    if len(word) < 6 or len(word) > 24 or not re.fullmatch(r"[a-zа-я]+", word):
        return word
    limit = 2 if len(word) >= 9 else 1
    options: list[tuple[float, str]] = []
    for candidate in TYPO_VOCABULARY:
        if abs(len(candidate) - len(word)) > limit:
            continue
        ratio = SequenceMatcher(None, word, candidate).ratio()
        if ratio >= (0.80 if len(word) >= 9 else 0.86):
            options.append((ratio, candidate))
    options.sort(reverse=True)
    if not options:
        return word
    if len(options) > 1 and options[0][0] - options[1][0] < 0.035:
        return word
    return options[0][1]


def correct_text(text: str) -> str:
    return re.sub(r"[a-zа-я]+", lambda match: correct_word(match.group(0)), normalize(text))


def clean_intent(text: str) -> tuple[str, str]:
    q = re.sub(r"[!?.,;:()\[\]{}]+", " ", correct_text(text))
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

def generate_code_reply(message: str) -> str | None:
    """Deterministic templates for simple code requests; no language model is used."""
    q = normalize(correct_text(message))
    if not re.search(r"(напиши|создай|сделай|сгенерируй|покажи|пример).{0,55}(код|программ)|\bкод на\b|\bпрограмма на\b|пример кода|простейший код", q):
        return None
    lang = "cpp" if re.search(r"(c\+\+|си плюс плюс|\bcpp\b)", q) else \
           "python" if re.search(r"(python|пайтон|питон)", q) else \
           "javascript" if re.search(r"(javascript|java script|\bjs\b|джаваскрипт)", q) else \
           "html" if re.search(r"\bhtml\b", q) else "python"
    task = "calculator" if re.search(r"(калькулятор|посчитать|сумм|сложи два числа|арифметик)", q) else \
           "fibonacci" if re.search(r"(фибоначчи|fibonacci)", q) else \
           "factorial" if re.search(r"(факториал|factorial)", q) else \
           "prime" if re.search(r"(простое число|проверить прост|prime number)", q) else \
           "even" if re.search(r"(четн|нечет|чётн|even|odd)", q) else "hello"
    fence = chr(96) * 3
    if lang == "python":
        examples = {
            "hello": ("Минимальная программа на Python.", 'def main():\n    print("Привет, мир!")\n\nif __name__ == "__main__":\n    main()'),
            "calculator": ("Калькулятор с проверкой деления на ноль.", 'def calculate(a, operator, b):\n    if operator == "+":\n        return a + b\n    if operator == "-":\n        return a - b\n    if operator == "*":\n        return a * b\n    if operator == "/":\n        if b == 0:\n            raise ValueError("На ноль делить нельзя")\n        return a / b\n    raise ValueError("Неизвестная операция")\n\nprint(calculate(8, "+", 4))'),
            "fibonacci": ("Первые 10 чисел Фибоначчи.", 'def fibonacci(n):\n    result = []\n    a, b = 0, 1\n    for _ in range(max(0, n)):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(10))'),
            "factorial": ("Факториал неотрицательного целого числа.", 'def factorial(n):\n    if n < 0:\n        raise ValueError("n должно быть неотрицательным")\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n\nprint(factorial(5))  # 120'),
            "prime": ("Проверка простого числа.", 'def is_prime(n):\n    if n < 2:\n        return False\n    divisor = 2\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 1\n    return True\n\nprint(is_prime(17))  # True'),
            "even": ("Проверка чётности.", 'number = int(input("Введите целое число: "))\nprint("Чётное" if number % 2 == 0 else "Нечётное")')
        }
        desc, code = examples[task]
        return desc + "\n\n" + fence + "python\n" + code + "\n" + fence
    if lang == "cpp":
        examples = {
            "hello": ('Минимальная программа на C++.', '#include <iostream>\n\nint main() {\n    std::cout << "Привет, мир!\\n";\n    return 0;\n}'),
            "calculator": ('Калькулятор с проверкой деления на ноль.', '#include <iostream>\n\nint main() {\n    double a, b;\n    char op;\n    if (!(std::cin >> a >> op >> b)) return 1;\n    if (op == \'+\') std::cout << a + b;\n    else if (op == \'-\') std::cout << a - b;\n    else if (op == \'*\') std::cout << a * b;\n    else if (op == \'/\' && b != 0) std::cout << a / b;\n    else std::cout << "Ошибка операции или деление на ноль";\n    std::cout << std::endl;\n    return 0;\n}'),
            "fibonacci": ('Первые 10 чисел Фибоначчи.', '#include <iostream>\n\nint main() {\n    long long a = 0, b = 1;\n    for (int i = 0; i < 10; ++i) {\n        std::cout << a << (i == 9 ? \'\\n\' : \' \');\n        long long next = a + b;\n        a = b;\n        b = next;\n    }\n    return 0;\n}'),
            "factorial": ('Факториал числа от 0 до 20.', '#include <iostream>\n\nint main() {\n    int n;\n    unsigned long long result = 1;\n    std::cin >> n;\n    if (n < 0 || n > 20) return 1;\n    for (int i = 2; i <= n; ++i) result *= i;\n    std::cout << result << std::endl;\n    return 0;\n}'),
            "prime": ('Проверка простого числа.', '#include <iostream>\n\nint main() {\n    int n;\n    std::cin >> n;\n    bool prime = n >= 2;\n    for (int d = 2; d * d <= n && prime; ++d)\n        if (n % d == 0) prime = false;\n    std::cout << (prime ? "Простое" : "Не простое") << std::endl;\n    return 0;\n}'),
            "even": ('Проверка чётности.', '#include <iostream>\n\nint main() {\n    long long n;\n    std::cin >> n;\n    std::cout << (n % 2 == 0 ? "Чётное" : "Нечётное") << std::endl;\n    return 0;\n}')
        }
        desc, code = examples[task]
        return desc + "\n\n" + fence + "cpp\n" + code + "\n" + fence + "\n\nСохрани как main.cpp и собери компилятором C++17: g++ main.cpp -std=c++17 -o app."
    if lang == "javascript":
        code = 'function main() {\n  console.log("Привет, мир!");\n}\n\nmain();'
        if task == "calculator":
            code = 'function calculate(a, op, b) {\n  if (op === "+") return a + b;\n  if (op === "-") return a - b;\n  if (op === "*") return a * b;\n  if (op === "/") {\n    if (b === 0) throw new Error("Деление на ноль");\n    return a / b;\n  }\n  throw new Error("Неизвестная операция");\n}\nconsole.log(calculate(8, "+", 4));'
        return "Минимальная программа на JavaScript.\n\n" + fence + "javascript\n" + code + "\n" + fence
    code = '<!doctype html>\n<html lang="ru">\n<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Моя страница</title></head>\n<body><h1>Привет, мир!</h1></body>\n</html>'
    return "Минимальная HTML-страница.\n\n" + fence + "html\n" + code + "\n" + fence


def conversation_reply(message: str, history: list[dict[str, Any]] | None = None) -> str | None:
    """Answer short follow-ups from recent chat context without inventing new facts."""
    _, stripped = clean_intent(message)
    turns = [item for item in (history or [])[-12:] if isinstance(item, dict)]
    previous = next((str(item.get("content", "")) for item in reversed(turns)
                     if item.get("role") == "assistant"), "")
    last_user = next((str(item.get("content", "")) for item in reversed(turns)
                      if item.get("role") != "assistant"), "")
    plain = re.sub(r"[\u0060]{3}[\s\S]*?[\u0060]{3}", "пример кода", previous)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"https?://\S+", "", plain)
    plain = re.sub(r"[*_#>\u0060]", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])", plain)
                 if len(part.strip()) >= 18][:3]

    if re.fullmatch(r"(не то|не так|я имел в виду|я имела в виду|я хотел сказать|я хотела сказать|я про другое|не это)", stripped):
        return f"Понял, я неверно истолковал запрос. Буду ориентироваться на твоё уточнение: «{message.strip()[:420]}». Я скорректирую ответ с учётом этого."
    if re.fullmatch(r"(готово|я сделал|я сделала|получилось|заработало|исправил|исправила)", stripped):
        return "Отлично, значит этот шаг сработал. Проверь результат ещё на одном примере, чтобы убедиться, что исправление не затронуло другие функции."
    if re.fullmatch(r"(не работает|не получается|не сработало|у меня ошибка|сломалось|не запускается)", stripped):
        return "Давай найдём причину без догадок. Пришли точный текст ошибки и укажи, после какого действия она появляется."
    if re.fullmatch(r"(да|ага|угу|ок|окей|ладно|хорошо|понятно|договорились)", stripped):
        return "Понял. Продолжим с учётом этого ответа — можешь прислать следующую деталь или показать результат."
    if re.fullmatch(r"(нет|неа|не нужно|не надо|не хочу)", stripped):
        return "Понял, этот вариант не подходит. Можем изменить подход или перейти к другой задаче."
    if re.fullmatch(r"(а\s+)?(почему|а почему|как так|что это значит|поясни|подробнее|расскажи подробнее|объясни подробнее|продолжи|продолжай|а если)", stripped):
        if not previous.strip():
            return "Я не вижу предыдущего ответа в переданном контексте. Повтори тему или фразу, к которой относится вопрос, и я продолжу."
        math = parse_math(last_user)
        if stripped in {"почему", "а почему", "как так"} and math:
            _, value = math
            return (f"Ты спрашиваешь о вычислении «{last_user.strip()[:100]}». Результат — {value:g}. "
                    "Сначала выполняются скобки и степени, затем умножение/деление, потом сложение/вычитание.")
        if stripped in {"почему", "а почему"}:
            cause = next((part for part in sentences if re.search(r"\b(потому что|так как|поскольку|из-за|поэтому)\b", part, re.I)), "")
            if cause:
                return f"В предыдущем ответе причина была сформулирована так: «{cause[:360]}». Если вопрос о другом тезисе, укажи его — разберу именно его."
            return f"Связываю вопрос с предыдущим ответом: «{(sentences[0] if sentences else plain)[:360]}». Причина там раскрыта не полностью, поэтому не буду придумывать детали. Укажи, какой тезис объяснить."
        summary = "\n".join(f"{i}. {part[:260]}" for i, part in enumerate(sentences, 1))
        return "Конечно, продолжу предыдущую мысль. Основные части ответа:\n" + (summary or plain[:500]) + "\n\nНапиши, какой пункт разобрать примером или пошагово."
    if re.fullmatch(r"(круто|класс|супер|отлично|здорово|всё понятно|все понятно)", stripped):
        return "Отлично! Можем перейти к следующему шагу или проверить идею на конкретном примере."
    if re.fullmatch(r"(что думаешь|а ты как думаешь|твоё мнение|твое мнение|как считаешь)", stripped) and previous.strip():
        return "Могу оценить предыдущую идею по трём вещам: пользу, сложность реализации и возможные риски. Лучше начать с маленького рабочего варианта и проверить его."
    return None


def quick_reply(message: str, is_creator: bool = False, mode: str = "normal", history: list[dict[str, Any]] | None = None) -> str | None:
    q, stripped = clean_intent(message)
    conversational = conversation_reply(message, history)
    if conversational:
        return conversational
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
    if re.search(r"(кто|назови|скажи|подскажи).{0,45}(создател|создал|автор|разработчик|владелец|сделал)|(кто сделал тебя|кто тебя создал|кто создал фросс|кто автор фросс)", q):
        return "Автор проекта FROSS: https://t.me/Spamton_BigShot_hahaha"
    if re.search(r"(моя почта|мой email|какая у меня почта|назови мой email|мой адрес электронной почты|почт[ауеы].{0,35}(автор|создател|владельц))", q):
        return "Я могу распознать подтверждённый аккаунт создателя, но не вывожу личную почту в сообщениях. Публичный контакт: @Spamton_BigShot_hahaha."
    if re.search(r"(что ты умеешь|твои возможности|помощь|help)[\s?!.,]*$", q):
        return "Я умею считать выражения, объяснять несколько базовых технических понятий, выдавать некоторые шаблоны кода и составлять сводки найденных страниц. Свободные ответы на любую тему пока ограничены."
    if re.search(r"(что такое fross local brain|что за версия мышления|версия твоего мышления|какая у тебя версия)", q):
        return f"Сейчас у меня {VERSION}. Это собственный алгоритмический движок, а не обученная нейросеть."

    code_reply = generate_code_reply(message)
    if code_reply:
        return code_reply

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
          mode: str = "normal", is_creator: bool = False,
          knowledge: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    history = history or []
    results = results or []
    direct = quick_reply(message, is_creator=is_creator, mode=mode, history=history)
    if direct:
        return {"reply": direct, "kind": "local-rule-v05",
                "trace": ["Нормализовать запрос", "Распознать намерение", "Ответить локальным правилом"], "sources": []}

    knowledge = knowledge or []
    query_words = {w for w in re.findall(r"[a-zа-яё0-9+#]{2,}", normalize(correct_text(message)))
                   if w not in {"что", "как", "где", "когда", "для", "это", "такое", "который", "которая", "которые", "того", "его", "она", "они", "оно", "он", "и", "в", "на", "по", "об", "про", "мне", "меня", "пожалуйста", "расскажи", "объясни", "тему", "темы", "работает", "работают", "работал", "работала", "подробнее", "информация", "информацию", "нужно", "хочу", "узнать", "the", "and", "with", "from", "this", "that"}}
    memory_ranked: list[tuple[float, dict[str, Any]]] = []
    for note in knowledge:
        topic = normalize(correct_text(str(note.get("topic", ""))))
        title = normalize(correct_text(str(note.get("title", ""))))
        snippet = normalize(correct_text(str(note.get("snippet", ""))))
        merged = topic + " " + title + " " + snippet
        if query_words:
            topic_hits = sum(word in topic for word in query_words)
            hits = sum(word in merged for word in query_words)
            score = min(1.0, hits / len(query_words) * 0.65 + topic_hits / len(query_words) * 0.45 + max(0.0, float(note.get("score", 0) or 0)) * 0.1)
            memory_ranked.append((score, note))
    memory_ranked.sort(key=lambda item: item[0], reverse=True)
    memory_matches = [item for score, item in memory_ranked if score >= 0.48][:4 if mode == "reasoning" else 2]
    if memory_matches:
        lines = ["Сначала проверил личную базу знаний FROSS и нашёл подходящие заметки."]
        for index, note in enumerate(memory_matches, 1):
            lines.append(f"{index}. **{note.get('title') or note.get('topic') or 'Заметка'}**\n{str(note.get('snippet', ''))[:850]}")
            url = str(note.get("url", ""))
            if url.startswith(("https://", "http://")):
                lines.append(f"Источник: [{url}]({url})")
        if mode == "reasoning":
            lines.append("Ограничение: это сохранённые поисковые фрагменты, а не обучение параметров нейросети.")
        return {"reply": "\n".join(lines), "kind": "saved-memory-v05",
                "trace": ["Проверить память", "Оценить совпадения", "Сформировать ответ", "Указать источник"],
                "sources": memory_matches}

    words = {w for w in re.findall(r"[a-zа-яё0-9]{2,}", normalize(correct_text(message)))
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
        return {"reply": "\n".join(lines), "kind": "search-summary-v05", "trace": trace,
                "sources": [item for _, item in selected]}

    last = next((item.get("content", "") for item in reversed(history)
                 if item.get("role") == "assistant" and isinstance(item.get("content"), str)), "")
    if re.search(r"\b(а почему|почему|а как|а если|расскажи подробнее|продолжи)\b", normalize(message)) and last:
        return {"reply": "Похоже, это уточнение к предыдущей теме, но я не понял к какой детали. Укажи фразу, которую нужно разобрать.",
                "kind": "context-fallback-v05", "trace": ["Проверить контекст", "Найти неоднозначность", "Попросить уточнить"], "sources": []}
    return {"reply": "Я не нашёл достаточно надёжных данных и не хочу выдумывать ответ. Мой движок 0.2 основан на локальных правилах и поисковых фрагментах; уточни запрос, и я попробую точнее.",
            "kind": "no-match-v05", "trace": ["Нормализовать запрос", "Проверить правила", "Обнаружить недостаток данных", "Сообщить об ограничении"], "sources": []}


if __name__ == "__main__":
    import unittest

    class BrainTests(unittest.TestCase):
        def test_greeting_with_bot_name(self):
            self.assertIn("Привет!", quick_reply("привет фросс") or "")
        def test_creator_typo(self):
            self.assertEqual(correct_text("кто созрдайтель фросса?"), "кто создатель фросс?")
            self.assertIn("Spamton_BigShot_hahaha", quick_reply("кто созрдайтель фросса?") or "")
        def test_typo_greeting(self):
            self.assertEqual(quick_reply("превет фросс"), "Привет! Я на связи. Что разберём?")
        def test_definition_knowledge(self):
            self.assertIn("Cloud Firestore", quick_reply("что такое Firestore") or "")
        def test_creator_greeting(self):
            self.assertIn("создатель FROSS", quick_reply("Привет, ФРОСС!", is_creator=True) or "")
        def test_math_precedence(self):
            self.assertEqual(parse_math("сколько будет (2+3)*4"), ("(2+3)*4", 20.0))
        def test_power(self):
            self.assertEqual(parse_math("2^8"), ("2^8", 256.0))
        def test_divide_by_zero(self):
            self.assertIn("Делить на ноль", quick_reply("10/0") or "")
        def test_generated_python(self):
            self.assertIn("def main()", quick_reply("создай простейший код на Python") or "")
        def test_generated_cpp(self):
            self.assertIn("int main()", quick_reply("напиши программу на C++") or "")
        def test_saved_knowledge(self):
            result = think("объясни тему про Kubernetes", knowledge=[{"topic": "Kubernetes", "title": "Kubernetes intro", "snippet": "Kubernetes manages containers", "url": "https://example.com/k", "score": 0.2}])
            self.assertEqual(result["kind"], "saved-memory-v05")
        def test_search_summary(self):
            result = think("как изучать Python", results=[{"title":"Python guide","snippet":"Python basics","url":"https://example.com","score":0.2}])
            self.assertIn("Python guide", result["reply"])

    unittest.main()
