// FROSS Local Brain 0.2 — original rule-based reasoning engine.
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
  { keys: ["массив"], answer: "Массив — упорядоченный набор значений. В JavaScript можно создать его так: const names = [имя1, имя2]; и обратиться к элементу по индексу, начиная с нуля." },
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
  } catch (error) {
    if (error?.message === "Делить на ноль нельзя") return { error: "division-by-zero" };
    return null;
  }
}

function quickReply({ message, isCreator = false, mode = "normal" }) {
  const raw = normalize(message);
  const q = raw.replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  const noBotName = q.replace(/(fross|фросс|ботик|бот)/g, " ").replace(/\s+/g, " ").trim();

  if (/^(эй\s+)?(привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый вечер)$/.test(noBotName)) {
    return isCreator
      ? "Привет, создатель FROSS! Что будем разбирать или улучшать сегодня?"
      : "Привет! Я на связи. Что разберём?";
  }
  if (/^(спасибо|благодарю|спс|пасиб)$/.test(noBotName)) return "Пожалуйста! Что делаем дальше?";
  if (/^(пока|до свидания|увидимся)$/.test(noBotName)) return "До встречи! Возвращайся, когда понадобится разобраться с задачей.";
  if (/^(как дела|что нового|ты здесь|ты тут|ты на связи|ау)$/.test(noBotName)) return "Я на связи. Могу посчитать выражение, объяснить базовую тему или найти материалы в интернете.";
  if (/^(кто ты|ты кто|как тебя зовут|что ты такое)$/.test(noBotName)) {
    return "Я FROSS Local Brain 0.2 — самописный движок на JavaScript с правилами, безопасным калькулятором, простыми шаблонами и поисковой сводкой. Я не использую готовую языковую модель, поэтому честно скажу, когда моих правил недостаточно.";
  }
  if (/(кто создал fross|кто создатель fross|кто автор fross|кто сделал тебя|кто тебя создал)/.test(q)) {
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
    return "Сейчас у меня FROSS Local Brain 0.2. Я определяю тип запроса, пробую локальные правила и вычисления, а если этого недостаточно — сопоставляю заголовки и фрагменты результатов поиска. Это собственный алгоритмический движок, не обученная нейросеть.";
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

  const asksDefinition = /(что такое|объясни|расскажи|определение|что означает|что значит)/.test(q);
  const definition = asksDefinition && DEFINITIONS.find(item => item.keys.some(key =>
    key.length <= 2 ? q.split(/\s+/).includes(key) : q.includes(key)
  ));
  if (definition) {
    return definition.answer + (mode === "reasoning"
      ? "\n\n**Проверка:** это базовое определение. Для более точного ответа можно указать пример или контекст, в котором встретился термин."
      : "");
  }

  if (/(пример кода|напиши код|сгенерируй код|напиши программу|сделай программу|код на|программа на)/.test(q)) {
    const python = /(python|питон)/.test(q);
    const js = /(javascript|java script|\bjs\b|джаваскрипт)/.test(q);
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
    return "Я пока умею выдавать только некоторые заранее проверенные шаблоны кода. Опиши конкретную задачу, язык и ожидаемый результат: я попробую подобрать подходящий шаблон, но не буду обещать произвольный готовый проект.";
  }

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

export function frossThink({ message, history = [], results = [], mode = "normal", isCreator = false } = {}) {
  const direct = quickReply({ message, isCreator, mode });
  if (direct) return { reply: direct, kind: "local-rule-v02", sources: [], trace: ["Нормализовал запрос", "Распознал тип задачи", "Ответил локальным правилом"] };

  if (results.length) {
    const queryWords = [...new Set(normalize(message).match(/[a-zа-яё0-9]{2,}/g) || [])]
      .filter(word => !["что", "как", "где", "когда", "для", "это", "the", "and", "with"].includes(word));
    const scoreItem = item => {
      const title = normalize(item.title || "");
      const text = normalize((item.title || "") + " " + (item.snippet || ""));
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
    lines.push(mode === "reasoning"
      ? "Я использовал локальную процедуру FROSS 0.2: выделил слова запроса, сравнил их с заголовками и фрагментами, убрал дубликаты и отсортировал результаты."
      : "Вот наиболее подходящие найденные материалы по совпадению с запросом.");
    top.forEach((item, index) => {
      const snippet = String(item.snippet || "Поисковик не предоставил краткого описания.").slice(0, 480);
      lines.push((index + 1) + ". **" + String(item.title || "Без заголовка") + "**\n" + snippet);
    });
    if (mode === "reasoning") {
      lines.push("**Проверка ограничений:** ранжирование основано на словесных совпадениях и описаниях в выдаче. Это не полное чтение страниц и не гарантия истинности.");
    } else {
      lines.push("Это сводка поисковых фрагментов. Важные утверждения стоит проверить на исходных страницах.");
    }
    lines.push("", "**Источники:**");
    top.forEach(item => {
      if (item.url && /^https?:\/\//i.test(item.url)) lines.push("- [" + String(item.title || "Открыть источник") + "](" + item.url + ")");
    });
    return {
      reply: lines.join("\n"),
      kind: "search-summary-v02",
      sources: top,
      trace: mode === "reasoning"
        ? ["Нормализовать запрос", "Выделить ключевые термины", "Сопоставить заголовки и фрагменты", "Оценить релевантность", "Убрать дубликаты", "Проверить слабые совпадения", "Сформировать сводку и оговорки"]
        : ["Нормализовать запрос", "Оценить релевантность", "Сформировать краткую сводку", "Добавить источники"]
    };
  }

  const lastAssistant = history.slice().reverse().find(item => item && item.role === "assistant" && typeof item.content === "string");
  if (/(^|\s)(а почему|почему|а как|а если|что значит это|расскажи подробнее|продолжи)(\s|[?!.,]|$)/i.test(String(message || "")) && lastAssistant) {
    return {
      reply: "Похоже, ты уточняешь предыдущую тему, но я не смог определить, к какой детали относится вопрос. Укажи фразу, которую нужно разобрать, — так я смогу точнее ответить.",
      kind: "context-fallback-v02",
      sources: [],
      trace: ["Проверить историю текущего чата", "Не удалось связать уточнение с конкретной деталью", "Запросить узкое уточнение"]
    };
  }

  return {
    reply: "Я пока не нашёл достаточно надёжных данных и не хочу выдумывать ответ. Мой движок 0.2 работает на локальных правилах и поисковых фрагментах, поэтому иногда ему не хватает контекста. Попробуй переформулировать запрос или уточнить желаемый результат.",
    kind: "no-match-v02",
    sources: [],
    trace: ["Нормализовать запрос", "Проверить локальные правила", "Недостаточно надёжных совпадений", "Честно сообщить об ограничении"]
  };
}

