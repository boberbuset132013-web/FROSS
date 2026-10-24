// FROSS Local Brain 0.1 — original rule-based reasoning engine.
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
  { keys: ["массив"], answer: "Массив — упорядоченный набор значений. В JavaScript можно создать его так: const names = [Аня, Илья]; и обратиться к элементу по индексу, начиная с нуля." },
  { keys: ["интернет"], answer: "Интернет — глобальная сеть взаимосвязанных устройств и сетей. Веб-сайты — лишь одна из служб интернета; также существуют электронная почта, обмен файлами, звонки и другие сервисы." },
  { keys: ["искусственный интеллект", "ии"], answer: "Искусственный интеллект — общее название методов, которые помогают программам решать задачи вроде распознавания, поиска закономерностей, планирования и обработки языка. Это не обязательно нейросеть: ИИ может быть и набором правил." },
  { keys: ["нейросеть"], answer: "Нейросеть — вычислительная модель из взаимосвязанных узлов и параметров, которая учится находить закономерности на примерах. Она отличается от простого набора правил тем, что многие её правила представлены обученными числовыми параметрами." },
  { keys: ["память", "контекст"], answer: "В этом прототипе я использую переданные сайтом последние сообщения текущего чата и доступные сохранённые данные. Это не человеческая память: если нужная информация не передана в контексте, я могу её не знать." }
];

function normalize(text) {
  return String(text || "").toLowerCase().replace(/ё/g, "е").trim();
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
  } catch {
    return null;
  }
}

function quickReply({ message, isCreator = false, mode = "normal" }) {
  const q = normalize(message);
  const greeting = /^(привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый вечер)[!. ]*$/.test(q);
  if (greeting) return isCreator
    ? "Привет, создатель FROSS! Что будем разбирать или улучшать сегодня?"
    : "Привет! Я FROSS Local Brain — небольшой самописный помощник. Могу считать, объяснять базовые понятия и искать материалы.";

  if (/^(спасибо|благодарю|спс)[!. ]*$/.test(q)) return "Пожалуйста! Если ответ оказался полезным — отлично. Что дальше?";
  if (/^(пока|до свидания|увидимся)[!. ]*$/.test(q)) return "До встречи! Возвращайся, когда понадобится разобраться с задачей.";
  if (/^(как дела|что нового)[?!. ]*$/.test(q)) return "Работаю штатно: правила, калькулятор и поиск на месте. Что проверим?";
  if (/^(кто ты|ты кто|как тебя зовут|что ты такое)[?!. ]*$/.test(q)) {
    return "Я FROSS Local Brain 0.1 — небольшой собственный движок на JavaScript. Я не использую готовую языковую модель: отвечаю с помощью правил, простых алгоритмов, безопасного вычислителя и сводок результатов поиска. Поэтому в свободном разговоре я слабее больших языковых моделей, зато логика здесь своя и проверяемая.";
  }
  if (/(кто создал fross|кто создатель fross|кто автор fross|кто сделал тебя|кто тебя создал)/.test(q)) {
    return "Автор проекта FROSS: [@Spamton_BigShot_hahaha](https://t.me/Spamton_BigShot_hahaha).";
  }
  if (/(моя почта|мой email|мой e-mail|какая у меня почта|назови мой email|какой у меня email|мой адрес электронной почты|почт[ауеы].{0,35}(автор|создател|владельц)|email.{0,35}(автор|создател|владельц)|адрес.{0,30}электронной почты)/.test(q)) {
    return isCreator
      ? "Я распознаю подтверждённый аккаунт создателя, но не буду выводить личный адрес почты в сообщениях. Для связи с автором есть публичный Telegram: @Spamton_BigShot_hahaha."
      : "Я не раскрываю личные адреса электронной почты. Публичный контакт автора FROSS: @Spamton_BigShot_hahaha.";
  }
  if (/(что ты умеешь|твои возможности|помощь|help)[\\s?!.,]*$/.test(q)) {
    return "Сейчас я умею: вычислять арифметические выражения; объяснять несколько базовых технических понятий; распознавать простые типы запросов; искать веб-страницы и сопоставлять их заголовки и фрагменты; выдавать несколько заготовленных примеров кода. Я пока не умею свободно писать на любую тему так, как большая языковая модель.";
  }

  const math = parseMath(message);
  if (math) {
    const result = String(math.value);
    if (mode === "reasoning") {
      return "Результат: **" + result + "**\n\nКраткая проверка: я разобрал выражение по приоритету операций — сначала скобки и степени, затем умножение/деление, затем сложение/вычитание. Вычисление выполнено локальным парсером без eval.";
    }
    return "**Ответ: " + result + "**\n\nВычислено локально.";
  }

  const definition = DEFINITIONS.find(item => item.keys.some(key => q.includes(key)) &&
    /(что такое|объясни|расскажи|значит|это что|определение|что означает)/.test(q));
  if (definition) {
    return definition.answer + (mode === "reasoning" ? "\n\n**Проверка:** это базовое объяснение термина; для более узкого контекста можно уточнить пример или язык программирования." : "");
  }

  if (/(пример кода|напиши код|сгенерируй код|напиши программу|сделай программу|код на|программа на)/.test(q)) {
    const python = /\b(python|питон)\b/.test(q);
    const js = /\b(javascript|java script|\bjs\b|джаваскрипт)\b/.test(q);
    const html = /\bhtml\b/.test(q);
    const hello = /(hello world|привет мир|выведи привет|напечатай привет)/.test(q);
    const calculator = /(калькулятор|сложи два числа|арифметическ)/.test(q);
    const fibonacci = /(фибоначчи|fibonacci)/.test(q);
    const fence = String.fromCharCode(96).repeat(3);
    if (hello && python) return "Пример на Python:\n\n" + fence + "python\nprint(\"Привет, мир!\")\n" + fence + "\n\nЗапусти файл командой python имя_файла.py.";
    if (hello && js) return "Пример на JavaScript:\n\n" + fence + "javascript\nconsole.log(\"Привет, мир!\");\n" + fence + "\n\nЗапусти в браузере или среде Node.js.";
    if (hello && html) return "Минимальная HTML-страница:\n\n" + fence + "html\n<!doctype html>\n<html lang=\"ru\">\n<head><meta charset=\"utf-8\"><title>Привет</title></head>\n<body><h1>Привет, мир!</h1></body>\n</html>\n" + fence;
    if (calculator && python) return "Небольшой калькулятор на Python:\n\n" + fence + "python\ndef calculate(a, operator, b):\n    if operator == \"+\":\n        return a + b\n    if operator == \"-\":\n        return a - b\n    if operator == \"*\":\n        return a * b\n    if operator == \"/\":\n        if b == 0:\n            raise ValueError(\"На ноль делить нельзя\")\n        return a / b\n    raise ValueError(\"Неизвестная операция\")\n\nprint(calculate(8, \"+\", 4))\n" + fence + "\n\nПример выводит 12. Функция явно проверяет деление на ноль.";
    if (fibonacci && python) return "Пример последовательности Фибоначчи на Python:\n\n" + fence + "python\ndef fibonacci(count):\n    result = []\n    a, b = 0, 1\n    for _ in range(max(0, count)):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fibonacci(10))\n" + fence + "\n\nФункция возвращает первые count чисел последовательности.";
    return "Я могу выдать некоторые проверенные шаблоны кода, но пока не умею надёжно генерировать произвольные программы. Уточни, что именно должна делать программа, на каком языке и какие данные получает или возвращает — я попробую собрать решение из доступных правил и примеров.";
  }
  return null;
}

export function frossQuickResponse(options) {
  return quickReply(options || {});
}

export function frossThink({ message, history = [], results = [], mode = "normal", isCreator = false } = {}) {
  const direct = quickReply({ message, isCreator, mode });
  if (direct) return { reply: direct, kind: "local-rule", sources: [] };

  if (results.length) {
    const ranked = results.slice().sort((a, b) => (b.score || 0) - (a.score || 0));
    const relevant = ranked.filter(item => (item.score || 0) >= 0.08);
    const top = (relevant.length ? relevant : ranked).slice(0, mode === "reasoning" ? 4 : 3);
    const lines = [];
    lines.push(mode === "reasoning"
      ? "Я использовал локальный алгоритм поиска совпадений: сравнил слова запроса с заголовками и фрагментами страниц и расположил подходящие материалы выше."
      : "Вот сводка найденных материалов, отсортированных по совпадению с запросом.");
    top.forEach((item, index) => {
      const snippet = String(item.snippet || "Поисковик не предоставил краткого описания.").slice(0, 520);
      lines.push((index + 1) + ". **" + String(item.title || "Без заголовка") + "**\n" + snippet);
    });
    if (mode === "reasoning") {
      lines.push("**Ограничение проверки:** это анализ заголовков и поисковых фрагментов, а не полное чтение и независимая проверка каждой страницы.");
    } else {
      lines.push("Это краткий обзор фрагментов выдачи; важные детали лучше сверить на исходных страницах.");
    }
    lines.push("", "**Источники:**");
    top.forEach(item => {
      if (item.url && /^https?:\/\//i.test(item.url)) lines.push("- [" + String(item.title || "Открыть источник") + "](" + item.url + ")");
    });
    return { reply: lines.join("\n"), kind: "search-summary", sources: top };
  }

  const lastAssistant = history.slice().reverse().find(item => item && item.role === "assistant" && typeof item.content === "string");
  if (/(^|\s)(а почему|почему|а как|а если|что значит это|расскажи подробнее|продолжи)(\s|[?!.,]|$)/i.test(String(message || "")) && lastAssistant) {
    return { reply: "Похоже, ты уточняешь предыдущую тему, но мой локальный движок не смог определить, к какой части ответа относится вопрос. Укажи слово или фразу, которую нужно разобрать, — я попробую точнее.", kind: "context-fallback", sources: [] };
  }

  return {
    reply: "Я пока небольшой самописный движок, а не большая языковая модель. Я не нашёл достаточно надёжных сведений и не хочу выдумывать ответ. Попробуй переформулировать вопрос или уточнить, что именно нужно узнать; я могу выполнить поиск и кратко сопоставить найденные материалы.",
    kind: "no-match",
    sources: []
  };
}
