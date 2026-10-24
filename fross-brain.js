// FROSS Local Brain 0.6 — compositional, rule-based reasoning engine.
// It calls no hosted language-model API. Answers come from explicit rules,
// a safe arithmetic parser, local templates, and summaries of search snippets.

const DEFINITIONS = [
  { keys: ["html", "хтмл"], answer: "HTML — язык разметки, который задаёт структуру веб-страницы: заголовки, абзацы, ссылки, изображения и формы. CSS отвечает за внешний вид, а JavaScript — за поведение." },
  { keys: ["css", "стили css"], answer: "CSS — язык оформления веб-страниц. Он задаёт цвета, шрифты, отступы, сетки, адаптацию под экран и анимации." },
  { keys: ["javascript", "джаваскрипт"], answer: "JavaScript — язык программирования, который добавляет веб-страницам поведение: обработку кнопок, проверку форм, изменение интерфейса и запросы к серверу." },
  { keys: ["алгоритм"], answer: "Алгоритм — точная последовательность шагов для решения задачи. Хороший алгоритм имеет понятные входные данные, правила обработки и ожидаемый результат." },
  { keys: ["переменная"], answer: "Переменная — именованное место, где программа хранит значение. Например, let score = 10 создаёт переменную score со значением 10." },
  { keys: ["цикл", "циклы"], answer: "Цикл повторяет действия. Например, for обычно используют, когда нужно пройти по диапазону или списку, а while — пока некоторое условие истинно." },
  { keys: ["функция"], answer: "Функция — именованный или отдельно описанный блок кода, который можно вызывать для выполнения задачи. Она помогает повторно использовать код и делить большую программу на части." },
  { keys: ["массив"], answer: "Массив — упорядоченный набор значений. В JavaScript можно создать его так: const names = [\"Анна\", \"Илья\"]; и обратиться к элементу по индексу, начиная с нуля." },
  { keys: ["интернет"], answer: "Интернет — глобальная сеть взаимосвязанных устройств и сетей. Веб-сайты — лишь одна из служб интернета; также существуют электронная почта, обмен файлами, звонки и другие сервисы." },
  { keys: ["искусственный интеллект", "ии"], answer: "Искусственный интеллект — общее название методов, которые помогают программам решать задачи вроде распознавания, поиска закономерностей, планирования и обработки языка. Это не обязательно нейросеть: ИИ может быть и набором правил." },
  { keys: ["нейросеть"], answer: "Нейросеть — вычислительная модель из взаимосвязанных узлов и параметров, которая учится находить закономерности на примерах. Она отличается от простого набора правил тем, что многие её правила представлены обученными числовыми параметрами." },
  { keys: ["память", "контекст"], answer: "В этом прототипе я использую переданные сайтом последние сообщения текущего чата. Это не человеческая память: если нужная информация не передана в контексте, я могу её не знать." },
  { keys: ["python", "питон"], answer: "Python — язык программирования с читаемым синтаксисом. Его часто используют для автоматизации, анализа данных, веб-сервисов и обучения машинному обучению. Пример: print(\"Привет!\")." },
  { keys: ["c++", "си плюс плюс", "си++"], answer: "C++ — компилируемый язык программирования, который поддерживает процедурный, объектно-ориентированный и обобщённый подходы. Его используют для игровых движков, системного ПО и приложений, где важны производительность и контроль ресурсов." },
  { keys: ["база данных", "базу данных", "базы данных"], answer: "База данных — организованное хранилище данных, с которым можно работать через запросы. Реляционные базы обычно хранят данные в таблицах, а документные могут хранить записи в виде документов." },
  { keys: ["api"], answer: "API — описанный способ взаимодействия программ. Например, веб-API принимает HTTP-запрос, обрабатывает его на сервере и возвращает ответ, часто в формате JSON." },
  { keys: ["http", "https"], answer: "HTTP — протокол передачи данных, используемый вебом. HTTPS — HTTP поверх TLS: соединение шифруется и позволяет проверять подлинность сервера." },
  { keys: ["dns"], answer: "DNS — система, которая сопоставляет доменные имена с сетевой информацией, например IP-адресами. Она помогает браузеру находить нужный сервер по имени сайта." },
  { keys: ["процессор", "cpu"], answer: "CPU — центральный процессор, который выполняет инструкции программ и координирует вычисления. Его производительность зависит не только от частоты, но и от архитектуры, числа ядер, памяти и характера задачи." },
  { keys: ["оперативная память", "ram"], answer: "RAM — оперативная память для временного хранения данных, которые активно использует система. Обычно её содержимое очищается при выключении питания." },
  { keys: ["видеокарта", "gpu"], answer: "GPU — графический процессор, хорошо приспособленный к большому числу параллельных вычислений. Он используется для графики и некоторых задач обработки данных." },
  { keys: ["firestore"], answer: "Cloud Firestore — облачная документная база данных Firebase. Данные хранятся в коллекциях и документах; доступ следует ограничивать правилами безопасности, проверяющими пользователя и владельца данных." },
  { keys: ["cloudflare"], answer: "Cloudflare предоставляет сетевые и облачные сервисы, включая CDN, защиту сайтов, DNS и Workers — среду для запуска серверного JavaScript и WebAssembly на периферии сети." },
  { keys: ["пароль", "пароли"], answer: "Пароль подтверждает доступ к аккаунту, но его нельзя публиковать или повторно использовать на разных сайтах. Сервис должен хранить не обычный текст пароля, а безопасный криптографический хеш. Для пользователя полезны уникальные пароли и двухфакторная защита." },
  { keys: ["переменная окружения", "секрет"], answer: "Переменная окружения позволяет передать настройку приложению без встраивания значения прямо в исходный код. Приватные ключи и чувствительные значения следует хранить в хранилище секретов, а не в публичном репозитории." },
  { keys: ["компилятор"], answer: "Компилятор преобразует исходный код в другую форму, например машинный код или промежуточное представление. Это отличается от интерпретатора, который обычно выполняет программу через среду исполнения." },
  { keys: ["git", "гит"], answer: "Git — система контроля версий. Она хранит историю изменений файлов, позволяет сравнивать версии, создавать ветки и возвращаться к предыдущим состояниям проекта." },
  { keys: ["github", "гитхаб"], answer: "GitHub — платформа для размещения Git-репозиториев и совместной работы над кодом: pull request, issues, review и автоматизация через Actions." },
  { keys: ["ошибка программирования", "отладка"], answer: "Отладка — поиск причины неправильного поведения программы. Полезно воспроизвести ошибку минимальным примером, посмотреть точное сообщение, проверить входные данные и исправить причину, а не только симптом." },
  { keys: ["переменная"], answer: "Переменная — именованное значение, которое программа может читать и изменять." }
];

function normalize(text) {
  return String(text || "").toLowerCase().replace(/ё/g, "е").trim();
}

const TYPO_ALIASES = Object.freeze({
  "созрдайтель": "создатель", "создотель": "создатель", "создатиль": "создатель", "создател": "создатель", "создител": "создатель", "саздатель": "создатель",
  "создатeль": "создатель", "создательь": "создатель", "создателя": "создателя",
  "разроботчик": "разработчик", "разработчк": "разработчик",
  "привте": "привет", "привит": "привет", "приве": "привет", "превет": "привет", "преветт": "привет",
  "фрос": "фросс", "фроссс": "фросс", "фросса": "фросс",
  "памаги": "помоги", "помаги": "помоги", "обясни": "объясни", "объесни": "объясни", "пачему": "почему", "почиму": "почему",
  "сколька": "сколько", "скока": "сколько", "обесни": "объясни", "обьясни": "объясни",
  "програма": "программа", "програмирование": "программирование",
  "интелект": "интеллект", "нейросит": "нейросеть", "нейросетьь": "нейросеть",
  "функцыя": "функция", "алгаритм": "алгоритм", "алгорит": "алгоритм",
  "неправельно": "неправильно", "неправельные": "неправильные", "неправельный": "неправильный", "созрдайтель": "создатель", "неправильно": "неправильно"
});

const CORRECTION_VOCABULARY = [
  "создатель", "создателя", "создателем", "создатели", "создали", "создал", "автор",
  "разработчик", "разработчика", "владелец", "сделал", "привет", "здравствуй",
  "здравствуйте", "фросс", "почему", "помоги", "объясни", "расскажи", "посчитать",
  "вычисли", "математика", "пример", "программа", "программирование", "переменная",
  "функция", "массив", "алгоритм", "нейросеть", "искусственный", "интеллект",
  "ошибка", "исправить", "неправильно", "сообщение", "приветствие", "вопрос",
  "поиск", "источник", "python", "javascript", "html", "css", "сайт", "чат",
  "удалить", "настройка", "объяснение", "версия", "мышление", "умеешь", "умеет"
];

function editDistance(left, right, stopAt = 3) {
  if (Math.abs(left.length - right.length) > stopAt) return stopAt + 1;
  let prev = Array.from({ length: right.length + 1 }, (_, i) => i);
  for (let i = 1; i <= left.length; i++) {
    const curr = [i];
    let rowMin = i;
    for (let j = 1; j <= right.length; j++) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
      rowMin = Math.min(rowMin, curr[j]);
    }
    if (rowMin > stopAt) return stopAt + 1;
    prev = curr;
  }
  return prev[right.length];
}

function correctToken(token) {
  const exact = TYPO_ALIASES[token];
  if (exact) return exact;
  if (token.length < 6 || token.length > 24 || !/^[a-zа-я]+$/i.test(token)) return token;
  const maxDistance = token.length >= 9 ? 2 : 1;
  let best = token;
  let bestDistance = maxDistance + 1;
  let ambiguous = false;
  for (const candidate of CORRECTION_VOCABULARY) {
    if (Math.abs(candidate.length - token.length) > maxDistance) continue;
    const distance = editDistance(token, candidate, maxDistance);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
      ambiguous = false;
    } else if (distance === bestDistance && candidate !== best) {
      ambiguous = true;
    }
  }
  return bestDistance <= maxDistance && !ambiguous ? best : token;
}

export function frossCorrectText(text) {
  return String(text || "").toLowerCase().replace(/ё/g, "е")
    .replace(/[a-zа-я]+/gi, token => correctToken(token));
}

function parseMath(text) {
  let expr = normalize(text)
    .replace(/^(сколько будет|сколько получится|посчитай|вычисли|реши|calculate)\s*/i, "")
    .replace(/умножить на|умножить|×/g, "*")
    .replace(/разделить на|делить на|÷/g, "/")
    .replace(/\bплюс\b/g, "+")
    .replace(/\bминус\b/g, "-")
    .replace(/\bв степени\b/g, "^")
    .replace(/[?!;:=]+\s*$/g, "")
    .trim();
  if (!expr || expr.length > 120 || !/^[\d\s().,+\-*/%^]+$/.test(expr)) return null;
  expr = expr.replace(/,/g, ".");
  const tokens = expr.match(/(?:\d+(?:\.\d*)?|\.\d+|[()+\-*/%^])/g);
  if (!tokens || tokens.join("") !== expr.replace(/\s+/g, "")) return null;
  let i = 0;
  const peek = () => tokens[i];
  const take = () => tokens[i++];

  function primary() {
    const token = take();
    if (token === "(") {
      const value = addSub();
      if (take() !== ")") throw new Error("Нужна закрывающая скобка");
      return value;
    }
    if (!token || !/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(token)) throw new Error("Ожидалось число");
    const value = Number(token);
    if (!Number.isFinite(value)) throw new Error("Число вне диапазона");
    return value;
  }
  function power() {
    const left = primary();
    if (peek() === "^") {
      take();
      const right = unary();
      if (Math.abs(right) > 100) throw new Error("Слишком большая степень");
      return left ** right;
    }
    return left;
  }
  function unary() {
    if (peek() === "+") { take(); return unary(); }
    if (peek() === "-") { take(); return -unary(); }
    return power();
  }
  function mulDiv() {
    let value = unary();
    while (["*", "/", "%"].includes(peek())) {
      const op = take();
      const right = unary();
      if ((op === "/" || op === "%") && right === 0) throw new Error("Делить на ноль нельзя");
      if (op === "*") value *= right;
      else if (op === "/") value /= right;
      else value %= right;
      if (!Number.isFinite(value) || Math.abs(value) > 1e100) throw new Error("Результат вне безопасного диапазона");
    }
    return value;
  }
  function addSub() {
    let value = mulDiv();
    while (peek() === "+" || peek() === "-") {
      const op = take();
      const right = mulDiv();
      value = op === "+" ? value + right : value - right;
      if (!Number.isFinite(value) || Math.abs(value) > 1e100) throw new Error("Результат вне безопасного диапазона");
    }
    return value;
  }
  try {
    const answer = addSub();
    if (i !== tokens.length) return null;
    return { expression: expr, value: Number(answer.toPrecision(12)) };
  } catch (error) {
    if (error?.message === "Делить на ноль нельзя") return { error: "division-by-zero" };
    return null;
  }
}

function generateCodeReply(message) {
  const q = normalize(frossCorrectText(message));
  const wantsCode = /(напиши|создай|сделай|сгенерируй|покажи|пример).{0,55}(код|программ)|\bкод на\b|\bпрограмма на\b|пример кода|простейший код/.test(q);
  if (!wantsCode) return null;

  const lang = /(c\+\+|си плюс плюс|\bcpp\b)/.test(q) ? "cpp"
    : /(python|пайтон|питон)/.test(q) ? "python"
    : /(javascript|java script|\bjs\b|джаваскрипт)/.test(q) ? "javascript"
    : /\bhtml\b/.test(q) ? "html" : "python";
  const task = /(калькулятор|посчитать|сумм|сложи два числа|арифметик)/.test(q) ? "calculator"
    : /(фибоначчи|fibonacci)/.test(q) ? "fibonacci"
    : /(факториал|factorial)/.test(q) ? "factorial"
    : /(простое число|проверить прост|prime number)/.test(q) ? "prime"
    : /(четн|нечет|чётн|even|odd)/.test(q) ? "even"
    : /(hello world|привет мир|выведи привет|напечатай привет)/.test(q) ? "hello"
    : "hello";
  const fence = String.fromCharCode(96).repeat(3);
  let code = "";
  let description = "";
  if (lang === "python") {
    if (task === "calculator") {
      description = "Небольшой калькулятор с проверкой деления на ноль.";
      code = "def calculate(a, operator, b):\n    if operator == \"+\":\n        return a + b\n    if operator == \"-\":\n        return a - b\n    if operator == \"*\":\n        return a * b\n    if operator == \"/\":\n        if b == 0:\n            raise ValueError(\"На ноль делить нельзя\")\n        return a / b\n    raise ValueError(\"Неизвестная операция\")\n\nprint(calculate(8, \"+\", 4))";
    } else if (task === "fibonacci") {
      description = "Функция возвращает первые n чисел последовательности Фибоначчи.";
      code = "def fibonacci(n):\n    result = []\n    a, b = 0, 1\n    for _ in range(max(0, n)):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(10))";
    } else if (task === "factorial") {
      description = "Вычисляет факториал неотрицательного целого числа.";
      code = "def factorial(n):\n    if n < 0:\n        raise ValueError(\"n должно быть неотрицательным\")\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result\n\nprint(factorial(5))  # 120";
    } else if (task === "prime") {
      description = "Проверяет, является ли целое число простым.";
      code = "def is_prime(n):\n    if n < 2:\n        return False\n    divisor = 2\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 1\n    return True\n\nprint(is_prime(17))  # True";
    } else if (task === "even") {
      description = "Проверяет чётность числа.";
      code = "number = int(input(\"Введите целое число: \"))\nif number % 2 == 0:\n    print(\"Чётное\")\nelse:\n    print(\"Нечётное\")";
    } else {
      description = "Минимальная программа на Python.";
      code = "def main():\n    print(\"Привет, мир!\")\n\nif __name__ == \"__main__\":\n    main()";
    }
  } else if (lang === "cpp") {
    if (task === "calculator") {
      description = "Простой калькулятор для двух чисел с проверкой деления на ноль.";
      code = "#include <iostream>\nusing namespace std;\n\nint main() {\n    double a, b;\n    char op;\n    cout << \"Введите выражение (например 8 + 4): \";\n    cin >> a >> op >> b;\n    if (op == '+') cout << a + b;\n    else if (op == '-') cout << a - b;\n    else if (op == '*') cout << a * b;\n    else if (op == '/' && b != 0) cout << a / b;\n    else cout << \"Ошибка операции или деление на ноль\";\n    cout << endl;\n    return 0;\n}";
    } else if (task === "fibonacci") {
      description = "Выводит первые 10 чисел Фибоначчи.";
      code = "#include <iostream>\nusing namespace std;\n\nint main() {\n    long long a = 0, b = 1;\n    for (int i = 0; i < 10; ++i) {\n        cout << a << (i == 9 ? '\\n' : ' ');\n        long long next = a + b;\n        a = b;\n        b = next;\n    }\n    return 0;\n}";
    } else if (task === "factorial") {
      description = "Считает факториал числа с проверкой входных данных.";
      code = "#include <iostream>\nusing namespace std;\n\nint main() {\n    int n;\n    unsigned long long result = 1;\n    cout << \"Введите n (0..20): \";\n    cin >> n;\n    if (n < 0 || n > 20) {\n        cout << \"Недопустимое значение\" << endl;\n        return 1;\n    }\n    for (int i = 2; i <= n; ++i) result *= i;\n    cout << result << endl;\n    return 0;\n}";
    } else if (task === "prime") {
      description = "Проверяет, является ли число простым.";
      code = "#include <iostream>\nusing namespace std;\n\nint main() {\n    int n;\n    cin >> n;\n    bool prime = n >= 2;\n    for (int d = 2; d * d <= n && prime; ++d)\n        if (n % d == 0) prime = false;\n    cout << (prime ? \"Простое\" : \"Не простое\") << endl;\n    return 0;\n}";
    } else {
      description = "Минимальная программа на C++.";
      code = "#include <iostream>\n\nint main() {\n    std::cout << \"Привет, мир!\\\\n\";\n    return 0;\n}";
    }
  } else if (lang === "javascript") {
    description = task === "calculator" ? "Простой калькулятор двух чисел." : "Минимальная программа на JavaScript.";
    code = task === "calculator"
      ? "function calculate(a, operator, b) {\n  if (operator === \"+\") return a + b;\n  if (operator === \"-\") return a - b;\n  if (operator === \"*\") return a * b;\n  if (operator === \"/\") {\n    if (b === 0) throw new Error(\"Деление на ноль\");\n    return a / b;\n  }\n  throw new Error(\"Неизвестная операция\");\n}\n\nconsole.log(calculate(8, \"+\", 4));"
      : "function main() {\n  console.log(\"Привет, мир!\");\n}\n\nmain();";
  } else {
    description = "Минимальная HTML-страница.";
    code = "<!doctype html>\n<html lang=\"ru\">\n<head>\n  <meta charset=\"utf-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n  <title>Моя страница</title>\n</head>\n<body>\n  <h1>Привет, мир!</h1>\n</body>\n</html>";
  }
  const compileNote = lang === "cpp" ? "\n\nСохрани как main.cpp и собери компилятором C++17, например: g++ main.cpp -std=c++17 -o app." : "";
  return description + "\n\n" + fence + lang + "\n" + code + "\n" + fence + compileNote;
}

const FROSS_STOP_WORDS_LOCAL = new Set(["что", "как", "где", "когда", "для", "это", "такое", "который", "которая", "которые", "это", "того", "его", "она", "они", "оно", "он", "и", "в", "на", "по", "об", "о", "про", "мне", "меня", "пожалуйста", "расскажи", "объясни", "тему", "темы", "работает", "работают", "работал", "работала", "расскажи", "подробнее", "информация", "информацию", "нужно", "хочу", "узнать", "the", "and", "with", "from", "this", "that"]);

function frossComposeExplanation(query, notes, mode = "normal") {
  const stop = FROSS_STOP_WORDS_LOCAL;
  const queryWords = [...new Set((normalize(frossCorrectText(query)).match(/[a-zа-яё0-9+#]{3,}/g) || [])
    .filter(word => !stop.has(word)))];
  const candidates = [];
  for (const note of (Array.isArray(notes) ? notes : []).slice(0, 6)) {
    const fields = [
      ...(Array.isArray(note.keyPoints) ? note.keyPoints : []),
      note.summary || "",
      note.snippet || note.content || ""
    ];
    for (const field of fields) {
      for (const raw of String(field || "").replace(/\s+/g, " ").split(/(?<=[.!?])\s+|\n+|\s+•\s+/)) {
        let sentence = raw.trim().replace(/^[-•\d.)\s]+/, "");
        if (sentence.length < 22 || sentence.length > 420) continue;
        if (!/[.!?]$/.test(sentence)) sentence += ".";
        const normalized = normalize(sentence).replace(/[^\p{L}\p{N}]+/gu, " ").trim();
        if (!normalized || candidates.some(item => item.normalized === normalized)) continue;
        const words = new Set((normalized.match(/[a-zа-яё0-9+#]{3,}/g) || []).filter(word => !stop.has(word)));
        const hits = queryWords.filter(word => words.has(word)).length;
        const relevance = queryWords.length ? hits / queryWords.length : 0;
        candidates.push({ sentence, normalized, relevance, length: sentence.length });
      }
    }
  }
  const selected = candidates
    .sort((a, b) => b.relevance - a.relevance || Math.min(b.length, 240) - Math.min(a.length, 240))
    .slice(0, mode === "reasoning" ? 4 : 3);
  if (!selected.length) return "";
  const parts = selected.map((item, index) => {
    const sentence = item.sentence.charAt(0).toLowerCase() + item.sentence.slice(1);
    if (index === 0) return sentence.charAt(0).toUpperCase() + sentence.slice(1);
    if (/^(это|также|кроме того|например|поэтому|однако|при этом)\b/i.test(sentence)) return sentence.charAt(0).toUpperCase() + sentence.slice(1);
    if (index === 1) return "Кроме того, " + sentence;
    if (index === 2) return "На практике это означает, что " + sentence.replace(/[.!?]+$/, "") + ".";
    return "Ещё один важный момент: " + sentence;
  });
  return parts.join(" ");
}

function findKnowledgeReply(query, knowledge, mode = "normal") {
  if (!Array.isArray(knowledge) || !knowledge.length) return null;
  const terms = value => new Set(
    (normalize(frossCorrectText(String(value || ""))).match(/[a-zа-яё0-9+#]{2,}/g) || [])
      .filter(word => !FROSS_STOP_WORDS_LOCAL.has(word) && word.length > 2)
  );
  const queryWords = [...terms(query)];
  if (!queryWords.length) return null;
  const coverage = (querySet, field) => {
    const words = terms(field);
    return querySet.filter(word => words.has(word)).length / querySet.length;
  };
  const ranked = knowledge.map(note => {
    const topicCoverage = coverage(queryWords, note.topic || "");
    const titleCoverage = coverage(queryWords, note.title || "");
    const keywordCoverage = coverage(queryWords, Array.isArray(note.keywords) ? note.keywords.join(" ") : "");
    const summaryCoverage = coverage(queryWords, [note.summary || "", ...(Array.isArray(note.keyPoints) ? note.keyPoints : [])].join(" "));
    const bodyCoverage = coverage(queryWords, note.snippet || note.content || "");
    const score = Math.min(1, topicCoverage * 0.34 + titleCoverage * 0.22 + keywordCoverage * 0.30 +
      summaryCoverage * 0.18 + bodyCoverage * 0.12 + Math.max(0, Number(note.score) || 0) * 0.06);
    return { note, score };
  }).sort((a, b) => b.score - a.score);
  const matches = ranked.filter(item => item.score >= 0.40).slice(0, mode === "reasoning" ? 4 : 3);
  if (!matches.length) return null;
  const synthesis = frossComposeExplanation(query, matches.map(item => item.note), mode);
  const lines = [mode === "reasoning"
    ? "FROSS 0.6 проверил личную память, сравнил несколько заметок и объединил подходящие тезисы в связное объяснение."
    : "Нашёл подходящие материалы в личной базе знаний FROSS. Вот связное объяснение по нескольким заметкам:"];
  if (synthesis) lines.push("**Объяснение простыми словами:** " + synthesis);
  for (const [index, item] of matches.entries()) {
    const note = item.note;
    const title = String(note.title || note.topic || "Заметка");
    lines.push((index + 1) + ". **" + title + "**");
    if (note.summary) lines.push("**Кратко:** " + String(note.summary).slice(0, 650));
    const points = Array.isArray(note.keyPoints) ? note.keyPoints.filter(Boolean).slice(0, mode === "reasoning" ? 4 : 2) : [];
    if (points.length) lines.push("**Главное:**\n" + points.map(point => "• " + String(point).slice(0, 260)).join("\n"));
    else if (note.snippet || note.content) lines.push(String(note.snippet || note.content).slice(0, 700));
    if (Array.isArray(note.keywords) && note.keywords.length) {
      lines.push("**Термины:** " + note.keywords.slice(0, 12).join(", "));
    }
    if (note.url && /^https?:\/\//i.test(note.url)) lines.push("Источник: [" + String(note.url) + "](" + note.url + ")");
  }
  if (mode === "reasoning") lines.push("**Ограничение:** материалы извлечены из открытых источников, а краткое изложение построено выбором предложений, а не генерацией обученной нейросетью. Проверяй важные утверждения по источникам.");
  return {
    reply: lines.join("\n"),
    kind: "saved-memory-v06",
    sources: matches.map(item => item.note),
    trace: mode === "reasoning"
      ? ["Нормализовать запрос", "Сравнить тему, заголовки, ключевые слова и резюме", "Оценить релевантность материалов", "Выбрать лучшие заметки", "Сформировать ответ из сохранённых тезисов", "Указать источники", "Проверить ограничения"]
      : ["Проверить личную базу знаний", "Найти совпадения по ключевым словам и резюме", "Выбрать релевантные заметки", "Указать источники"]
  };
}

function frossConversationReply(message, history = []) {
  const q = normalize(frossCorrectText(message)).replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  const noBotName = q.split(/\s+/).filter(word => !["fross", "фросс", "ботик", "бот"].includes(word)).join(" ").trim();
  const turns = Array.isArray(history) ? history.slice(-12).filter(item => item && typeof item.content === "string") : [];
  const lastAssistant = turns.slice().reverse().find(item => item.role === "assistant")?.content || "";
  const lastUser = turns.slice().reverse().find(item => item.role !== "assistant")?.content || "";
  const cleanPrevious = String(lastAssistant).replace(/[\u0060]{3}[\s\S]*?[\u0060]{3}/g, " пример кода ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/https?:\/\/\S+/g, "")
    .replace(/^\s{0,3}#{1,6}\s*/gm, "").replace(/[*_#>\u0060]/g, " ").replace(/\s+/g, " ").trim();
  const sentences = cleanPrevious.split(/(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])/)
    .map(sentence => sentence.trim()).filter(sentence => sentence.length >= 18).slice(0, 4);
  const usefulLines = String(lastAssistant).split(/\n/)
    .map(line => line.replace(/^\s*(?:[-•]|\d+[.)])\s*/, "").replace(/[*_#>\u0060]/g, "").trim())
    .filter(line => line.length >= 24 && !/^(источники|ограничение|проверка ограничений)\b/i.test(line)).slice(0, 4);

  if (/^(помнишь|ты помнишь|о чем мы говорили|о чём мы говорили|что я говорил|что я говорила)$/.test(noBotName)) {
    const topic = String(lastUser || "").trim().replace(/\s+/g, " ").slice(0, 220);
    return topic
      ? "В переданном контексте последняя тема от тебя была такой: «" + topic + "». Я могу учитывать только историю, которую сайт приложил к текущему запросу."
      : "Я могу опираться только на переписку, которую сайт передал в текущем запросе. Сейчас предыдущей реплики пользователя в контексте нет.";
  }
  if (/^(как у тебя дела|как у вас дела|как твои дела|как ваши дела|как поживаешь|как поживаете|как жизнь|что у тебя нового|что у вас нового|какие у тебя новости|как сам|как сама)$/.test(noBotName)) {
    return "Я на связи и готов помогать с вопросами и проектами. А у тебя как дела?";
  }
  if (/^(чем занимаешься|что делаешь|что ты делаешь|как настроение|как ты)$/.test(noBotName)) {
    return "Я разбираю твоё сообщение и подбираю полезный ответ. У меня нет человеческого настроения, но я могу помочь с кодом, идеей проекта или просто поддержать беседу.";
  }
  if (/^(давай поговорим|поговорим|мне скучно|не знаю о чем поговорить|не знаю о чём поговорить)$/.test(noBotName)) {
    return "Давай. Можем обсудить игры, программирование, идеи для проекта или что-нибудь совершенно другое. Какая тема сейчас интереснее?";
  }
  if (/^(у меня идея|у меня есть идея|есть идея)$/.test(noBotName)) {
    return "Расскажи идею в двух-трёх предложениях. Я помогу выделить главное, заметить возможные сложности и придумать, с чего начать.";
  }
  if (/^(ты живой|ты настоящий|ты человек|у тебя есть чувства|ты чувствуешь)$/.test(noBotName)) {
    return "Я не человек и не обладаю человеческими чувствами. FROSS — алгоритмический помощник, который может поддерживать диалог по правилам и доступному контексту.";
  }
  const wantsProject = noBotName.match(/^я хочу (?:создать|сделать|разработать|построить) (.{2,140})$/);
  if (wantsProject && !String(message || "").includes("?")) {
    return "Давай превратим идею в понятный план. Для проекта «" + wantsProject[1].slice(0, 120) + "» сначала определим главную функцию, затем соберём небольшой прототип и проверим его. Что проект должен уметь в самой первой версии?";
  }
  const sharesInterest = noBotName.match(/^(?:мне нравится|я люблю) (.{2,100})$/);
  if (sharesInterest) {
    return "Понял, тебе интересно «" + sharesInterest[1].slice(0, 90) + "». Что в этом больше всего нравится — сам процесс, результат или какая-то конкретная часть?";
  }

  if (/^(не то|не так|ты не понял|ты меня не понял|я имел в виду|я имела в виду|я хотел сказать|я хотела сказать|я про другое|не это)$/.test(noBotName) ||
      /^(я имел в виду|я имела в виду|я хотел сказать|я хотела сказать|ты не понял|ты меня не понял)\b/.test(noBotName)) {
    return "Понял, я неверно истолковал запрос. Буду ориентироваться на твоё уточнение: «" + String(message).trim().slice(0, 420) + "». Если я снова упущу важную деталь, поправь меня — я скорректирую ответ.";
  }
  if (/^(я сделал|я сделала|получилось|заработало|исправил|исправила|готово|всё готово|все готово|теперь работает)[.! ]*$/.test(noBotName)) {
    return "Отлично, значит этот шаг сработал. Дальше стоит проверить результат на одном-двух примерах, чтобы убедиться, что исправление не повлияло на другие функции.";
  }
  if (/^(не работает|ничего не работает|не получается|не сработало|у меня ошибка|вылезла ошибка|сломалось|не запускается|опять ошибка)[.! ]*$/.test(noBotName)) {
    return "Давай найдём причину без догадок. Пришли точный текст ошибки и напиши, после какого действия она появляется. По этим деталям выберем следующий шаг.";
  }
  if (/^(да|ага|угу|ок|окей|ладно|хорошо|ясно|понял|понятно|договорились|верно|именно)[.! ]*$/.test(noBotName)) {
    if (lastAssistant.trim().endsWith("?")) return "Хорошо, понял. Продолжим с учётом этого ответа. Напиши следующую деталь или покажи, что получилось.";
    return "Понял. Продолжим — можем разобрать следующий шаг или проверить результат на примере.";
  }
  if (/^(нет|неа|не нужно|не надо|не хочу)[.! ]*$/.test(noBotName)) {
    if (lastAssistant.trim().endsWith("?")) return "Хорошо, не будем идти в этом направлении. Учту ответ; можем изменить подход или перейти к другой задаче.";
    return "Понял, этот вариант не подходит. Скажи, что хочется изменить, и я подстрою следующий ответ.";
  }

  const isFollowUp = /^(а\s+)?(почему|а почему|как так|а как|что это значит|что значит это|поясни|объясни подробнее|расскажи подробнее|подробнее|продолжи|продолжай|и дальше|а дальше|и что дальше|а если|что насчёт этого|что насчет этого)[?!. ]*$/.test(noBotName);
  if (isFollowUp) {
    if (!lastAssistant.trim()) return "Я не вижу предыдущего ответа в переданном контексте. Повтори тему или фразу, к которой относится вопрос, и я продолжу.";
    const math = parseMath(lastUser);
    if (/^(почему|а почему|как так)[?!. ]*$/.test(noBotName) && math && !math.error) {
      return "Ты спрашиваешь о предыдущем вычислении «" + String(lastUser).trim().slice(0, 100) + "». Результат — " + math.value + ". Приоритет такой: сначала скобки и степени, потом умножение/деление, затем сложение/вычитание. Если нужно, разберу выражение по действиям.";
    }
    if (/^(почему|а почему)[?!. ]*$/.test(noBotName)) {
      const cause = sentences.find(sentence => /\b(потому что|так как|поскольку|из-за|поэтому)\b/i.test(sentence));
      if (cause) return "В предыдущем ответе причина была сформулирована так: «" + cause.slice(0, 360) + "». Если вопрос о другом тезисе, укажи его — разберу именно его.";
      return "Связываю вопрос с предыдущим ответом: «" + (sentences[0] || cleanPrevious).slice(0, 360) + "». Причина не была раскрыта достаточно подробно, поэтому не буду её придумывать. Укажи утверждение, к которому относится «почему», и я разберу его точнее.";
    }
    if (/^(а если)[?!. ]*$/.test(noBotName)) {
      return "Можем проверить другой вариант. Сейчас предыдущий контекст — «" + (sentences[0] || cleanPrevious).slice(0, 280) + "». Напиши условие, которое хочешь изменить, и я сравню результат.";
    }
    const points = usefulLines.length ? usefulLines : sentences;
    const summary = points.slice(0, 3).map((line, index) => (index + 1) + ". " + line.slice(0, 260)).join("\n");
    return "Конечно, продолжу предыдущую мысль. Вот её основные части:\n" + (summary || cleanPrevious.slice(0, 500)) + "\n\nЭто разбор уже сказанного, а не новый источник. Напиши, какой пункт раскрыть примером или пошаговым объяснением.";
  }

  if (/^(круто|класс|супер|отлично|здорово|понятно спасибо|всё понятно|все понятно)[.! ]*$/.test(noBotName)) {
    return "Отлично! Хорошо, что стало понятнее. Можем перейти к следующему шагу или проверить идею на конкретном примере.";
  }
  if (/^(что думаешь|а ты как думаешь|твоё мнение|твое мнение|по твоему|как считаешь)[?!. ]*$/.test(noBotName) && lastAssistant.trim()) {
    return "Могу оценить предыдущую идею по трём вещам: насколько она решает задачу, насколько сложна в реализации и какие у неё риски. По текущему контексту лучше сначала проверить маленький рабочий вариант, а затем усложнять его.";
  }
  return null;
}

function quickReply({ message, isCreator = false, mode = "normal", history = [] }) {
  const raw = normalize(frossCorrectText(message));
  const q = raw.replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  const noBotName = q.split(/\s+/).filter(word => !["fross", "фросс", "ботик", "бот"].includes(word)).join(" ").trim();
  const conversational = frossConversationReply(message, history);
  if (conversational) return conversational;

  if (/^(эй\s+)?(привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый вечер)$/.test(noBotName)) {
    return isCreator
      ? "Привет, создатель FROSS! Что будем разбирать или улучшать сегодня?"
      : "Привет! Я на связи. Что разберём?";
  }
  if (/^(спасибо|благодарю|спс|пасиб)$/.test(noBotName)) return "Пожалуйста! Что делаем дальше?";
  if (/^(помоги|помогите|можешь помочь|помоги мне|нужна помощь)$/.test(noBotName)) return "Конечно. Напиши, что именно нужно сделать и какой результат ты хочешь получить. Я сначала проверю известные правила, а при необходимости поищу материалы.";
  if (/^(пока|до свидания|увидимся)$/.test(noBotName)) return "До встречи! Возвращайся, когда понадобится разобраться с задачей.";
  if (/^(как дела|что нового|ты здесь|ты тут|ты на связи|ау)$/.test(noBotName)) return "Я на связи. Могу посчитать выражение, объяснить базовую тему или найти материалы в интернете.";
  if (/^(кто ты|ты кто|как тебя зовут|что ты такое)$/.test(noBotName)) {
    return "Я FROSS Local Brain 0.6 — самописный движок на JavaScript с правилами, безопасным калькулятором, простыми шаблонами и поисковой сводкой. Я не использую готовую языковую модель, поэтому честно скажу, когда моих правил недостаточно.";
  }
  if (/(кто|назови|скажи|подскажи).{0,45}(создател|создал|автор|разработчик|владелец|сделал)/.test(q) || /(кто сделал тебя|кто тебя создал|кто создал фросс|кто автор фросс)/.test(q)) {
    return "Автор проекта FROSS: [@Spamton_BigShot_hahaha](https://t.me/Spamton_BigShot_hahaha).";
  }
  if (/(моя почта|мой email|мой e-mail|какая у меня почта|назови мой email|какой у меня email|мой адрес электронной почты|почт[ауеы].{0,35}(автор|создател|владельц)|email.{0,35}(автор|создател|владельц)|адрес.{0,30}электронной почты)/.test(q)) {
    return isCreator
      ? "Я распознаю подтверждённый аккаунт создателя, но не буду публиковать личную почту в чате. Для связи есть публичный Telegram: @Spamton_BigShot_hahaha."
      : "Я не раскрываю личные адреса электронной почты. Публичный контакт автора FROSS: @Spamton_BigShot_hahaha.";
  }
  if (/(что ты умеешь|твои возможности|помощь|help)[\s?!.,]*$/.test(q)) {
    return "Я умею выполнять арифметические выражения, объяснять несколько базовых технических понятий, выдавать некоторые шаблоны кода и собирать краткие сводки найденных страниц. Для открытых вопросов я использую поиск, но пока не могу свободно генерировать любой ответ как большая языковая модель.";
  }
  if (/(что такое fross local brain|что за версия мышления|версия твоего мышления|какая у тебя версия)/.test(q)) {
    return "Сейчас у меня FROSS Local Brain 0.6. Я определяю тип запроса, пробую локальные правила и вычисления, а если этого недостаточно — сопоставляю заголовки и фрагменты результатов поиска. Это собственный алгоритмический движок, не обученная нейросеть.";
  }
  if (/(твоя модель|какая у тебя модель|какую модель ты используешь|какая нейросеть|ты используешь модель|ты на какой модели|ты просто гуглишь|почему ты ищешь в интернете)/.test(q)) {
    return "Я не использую готовую языковую модель. Я — FROSS Local Brain 0.6: правила, безопасный вычислитель, генерация простого кода, личная база знаний и поиск только тогда, когда сохранённых знаний не хватает. Это не нейросеть и не дообучение весов.";
  }

  const math = parseMath(message);
  if (math?.error === "division-by-zero") return "Делить на ноль нельзя — такое выражение не имеет определённого результата.";
  if (math) {
    const result = String(math.value);
    if (mode === "reasoning") {
      return "Результат: **" + result + "**\n\nПроверка: выражение обработано локальным парсером с приоритетом операций — скобки, степени, умножение и деление, затем сложение и вычитание. Выполнение произвольного кода не используется.";
    }
    return "**Ответ: " + result + "**\n\nВычислено локально.";
  }

  const asksDefinition = /(что такое|объясни|расскажи|определение|что означает|что значит|расскажи про|объяснение)/.test(q);
  const definition = asksDefinition && DEFINITIONS.find(item => item.keys.some(key =>
    key.length <= 2 ? q.split(/\s+/).includes(key) : q.includes(key)
  ));
  if (definition) {
    return definition.answer + (mode === "reasoning"
      ? "\n\n**Проверка:** это базовое определение. Для более точного ответа можно указать пример или контекст, в котором встретился термин."
      : "");
  }

  const generatedCode = generateCodeReply(message);
  if (generatedCode) return generatedCode;

  if (/(с чего начать|план для|составь план|помоги спланировать|как разработать|как создать)/.test(q)) {
    if (/(сайт|веб|вебсайт|страниц)/.test(q)) {
      return "Предлагаю такой план разработки сайта:\n\n1. Определить назначение и основные функции.\n2. Нарисовать структуру экранов и состояния интерфейса.\n3. Собрать HTML-разметку и адаптивные CSS-стили.\n4. Добавить JavaScript-логику и обработку ошибок.\n5. Проверить узкие и широкие экраны, затем опубликовать сайт.\n\nЧтобы перейти от плана к конкретике, нужно знать, какие функции ты хочешь видеть на сайте.";
    }
    if (/(игр|game|роблокс|roblox)/.test(q)) {
      return "План для игрового проекта:\n\n1. Определить основную механику и цель игрока.\n2. Сделать небольшой прототип одного игрового цикла.\n3. Добавить интерфейс, сохранение прогресса и обработку ошибок.\n4. Проверить игру с несколькими тестовыми сценариями.\n5. Расширять проект только после того, как основной цикл стабильно работает.";
    }
    return "Разложим задачу на шаги:\n\n1. Сформулировать желаемый результат.\n2. Определить входные данные и ограничения.\n3. Выделить 2–5 небольших этапов.\n4. Сделать самый простой рабочий прототип.\n5. Проверить результат на примерах и исправить ошибки.\n\nЭто общий план; расскажи, какой именно проект ты задумал, чтобы уточнить шаги.";
  }

  return null;
}
export function frossQuickResponse(options) {
  return quickReply(options || {});
}

export function frossThink({ message, history = [], results = [], knowledge = [], mode = "normal", isCreator = false } = {}) {
  const correctedMessage = frossCorrectText(message);
  const direct = quickReply({ message: correctedMessage, isCreator, mode, history });
  const memoryReply = findKnowledgeReply(correctedMessage, knowledge, mode);
  const simpleIntent = normalize(correctedMessage).replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  const conversationalIntent = /^(эй )?(привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый вечер|спасибо|благодарю|спс|пасиб|пока|до свидания|увидимся|как дела|что нового|ты здесь|ты тут|ты на связи|ау|кто ты|ты кто|как тебя зовут|что ты такое)$/.test(simpleIntent);
  if (direct && conversationalIntent) return { reply: direct, kind: "local-rule-v05", sources: [], trace: ["Распознать короткое разговорное сообщение", "Ответить локальным правилом"] };
  // Learned material takes priority over generic dictionary templates when the query matches it.
  if (memoryReply && !results.length) return memoryReply;
  if (direct) return { reply: direct, kind: "local-rule-v05", sources: [], trace: correctedMessage !== normalize(message) ? ["Нормализовал текст", "Исправил вероятные опечатки по словарю", "Распознал тип задачи", "Ответил локальным правилом"] : ["Нормализовал запрос", "Распознал тип задачи", "Ответил локальным правилом"] };

  if (results.length) {
    const queryWords = [...new Set(normalize(correctedMessage).match(/[a-zа-яё0-9]{2,}/g) || [])]
      .filter(word => !["что", "как", "где", "когда", "для", "это", "the", "and", "with"].includes(word));
    const scoreItem = item => {
      const title = normalize(frossCorrectText(item.title || ""));
      const text = normalize(frossCorrectText((item.title || "") + " " + (item.snippet || "")));
      let titleHits = 0, hits = 0;
      for (const word of queryWords) {
        if (title.includes(word)) titleHits++;
        if (text.includes(word)) hits++;
      }
      const overlap = queryWords.length ? hits / queryWords.length : 0;
      const titleOverlap = queryWords.length ? titleHits / queryWords.length : 0;
      return Math.min(1, (Number(item.score) || 0) * 0.55 + overlap * 0.35 + titleOverlap * 0.2);
    };
    const ranked = results.map(item => ({ ...item, localRelevance: scoreItem(item) }))
      .sort((a, b) => b.localRelevance - a.localRelevance);
    const unique = [];
    const seen = new Set();
    for (const item of ranked) {
      const key = String(item.url || item.title || "").replace(/\/$/, "").toLowerCase();
      if (key && seen.has(key)) continue;
      if (key) seen.add(key);
      unique.push(item);
    }
    const relevant = unique.filter(item => item.localRelevance >= 0.08);
    const top = (relevant.length ? relevant : unique).slice(0, mode === "reasoning" ? 4 : 3);
    const lines = [];
    if (memoryReply) {
      lines.push("**Сначала — что FROSS уже изучил:**\n" + memoryReply.reply);
      lines.push("");
      lines.push("**Теперь — свежие материалы из интернета:**");
    } else {
      lines.push(mode === "reasoning"
        ? "Я сопоставил сохранённые знания с результатами поиска, сравнил заголовки и доступные фрагменты, оценил релевантность и убрал дубликаты. Полный текст страниц не проверяется, если он отдельно не предоставлен."
        : "Вот наиболее подходящие материалы. Ниже поясняю, что можно понять по найденным фрагментам и какие ограничения остаются.");
    }
    top.forEach((item, index) => {
      const snippet = String(item.snippet || "Поисковик не предоставил краткого описания.").slice(0, 480);
      lines.push((index + 1) + ". **" + String(item.title || "Без заголовка") + "**\n" + snippet);
      if (item.explanation) lines.push("**Объяснение:** " + String(item.explanation).slice(0, 1350));
    });
    if (mode === "reasoning") {
      lines.push("**Проверка ограничений:** ранжирование основано на словесных совпадениях и доступных текстах. Это алгоритмическая сводка, а не нейросетевое понимание; важные утверждения нужно сверять с первоисточником.");
    } else {
      lines.push("Это сводка поисковых фрагментов. Важные утверждения стоит проверить на исходных страницах.");
    }
    lines.push("", "**Источники:**");
    top.forEach(item => {
      if (item.url && /^https?:\/\//i.test(item.url)) lines.push("- [" + String(item.title || "Открыть источник") + "](" + item.url + ")");
    });
    return {
      reply: lines.join("\n"),
      kind: "search-summary-v05",
      sources: [...(memoryReply?.sources || []), ...top],
      trace: mode === "reasoning"
        ? ["Нормализовать запрос", "Выделить ключевые термины", "Сопоставить заголовки и фрагменты", "Оценить релевантность", "Убрать дубликаты", "Проверить слабые совпадения", "Сформировать сводку и оговорки"]
        : ["Нормализовать запрос", "Оценить релевантность", "Сформировать краткую сводку", "Добавить источники"]
    };
  }

  const lastAssistant = history.slice().reverse().find(item => item && item.role === "assistant" && typeof item.content === "string");
  if (/(^|\s)(а почему|почему|а как|а если|что значит это|расскажи подробнее|продолжи)(\s|[?!.,]|$)/i.test(String(message || "")) && lastAssistant) {
    return {
      reply: "Похоже, ты уточняешь предыдущую тему, но я не смог определить, к какой детали относится вопрос. Укажи фразу, которую нужно разобрать, — так я смогу точнее ответить.",
      kind: "context-fallback-v05",
      sources: [],
      trace: ["Проверить историю текущего чата", "Не удалось связать уточнение с конкретной деталью", "Запросить узкое уточнение"]
    };
  }

  return {
    reply: "Я пока не нашёл достаточно надёжных данных и не хочу выдумывать ответ. Мой движок 0.6 использует локальные правила, сохранённые знания и поисковые фрагменты, поэтому иногда ему не хватает контекста. Попробуй переформулировать запрос или уточнить желаемый результат.",
    kind: "no-match-v05",
    sources: [],
    trace: ["Нормализовать запрос", "Проверить локальные правила", "Недостаточно надёжных совпадений", "Честно сообщить об ограничении"]
  };
}

