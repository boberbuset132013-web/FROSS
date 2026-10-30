import { frossCorrectText } from "./fross-brain.js";
// ================= КОНФИГУРАЦИЯ =================
// FROSS: ограничиваем время каждого поискового запроса.
const FROSS_FETCH_TIMEOUT_MS = 4500;
const FROSS_MAX_MESSAGE_LENGTH = 30000;
const FROSS_MAX_REPLY_LENGTH = 50000;
const FROSS_MAX_SEARCH_QUERY_LENGTH = 400;
const FROSS_MAX_HISTORY_USER_MESSAGES = 5;
const FROSS_MAX_RESULTS_PER_PROVIDER = 5;
const FROSS_DEFAULT_PYTHON_BRAIN_URL = "https://fross-python-brain.hostless.app/chat";
const FROSS_MIN_BRAIN_TOKEN_LENGTH = 16;

const FROSS_TEACHER_TRAINING_MODELS = Object.freeze([
  { id: "@cf/zai-org/glm-4.7-flash", name: "GLM 4.7 Flash" },
  { id: "@cf/google/gemma-4-26b-a4b-it", name: "Gemma 4 26B" },
  { id: "@cf/nvidia/nemotron-3-120b-a12b", name: "Nemotron 3 120B" },
  { id: "@cf/qwen/qwen3-30b-a3b-fp8", name: "Qwen 3 30B" },
  { id: "@cf/meta/llama-3.3-70b-instruct-fp8-fast", name: "Llama 3.3 70B Fast" },
  { id: "@cf/mistralai/mistral-small-3.1-24b-instruct", name: "Mistral Small 3.1" },
  { id: "@cf/qwen/qwen2.5-coder-32b-instruct", name: "Qwen 2.5 Coder 32B" },
  { id: "@cf/openai/gpt-oss-120b", name: "GPT-OSS 120B" }
]);

// ================= ОСНОВНОЙ ОБРАБОТЧИК =================
export default {
  async scheduled(controller, env, ctx) {
    ctx.waitUntil(frossAutoStudyProgramming(env, controller?.cron || "scheduled"));
  },
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // 1. CORS preflight
    if (method === "OPTIONS") return corsPreflight();

    // 2. Health check — без обращения к Telegram
    if (path === "/ping" || path === "/health") {
      return new Response(JSON.stringify({
        ok: true,
        ts: Date.now(),
        fross: "integrated"
      }), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    // 3. FROSS API: JSON-ошибки вместо HTML-страниц Cloudflare.
    if (path === "/api/health") return frossApiGuard(path, request, async () => {
      const pythonBrainHealth = await frossProbePythonHealth(frossPythonBrainUrl(env), frossBrainServiceToken(env));
      return frossJson({
        ok: true,
        worker: "FROSS",
        version: "1.2",
        brainVersion: "1.5",
        build: "fross-hybrid-cpp-2026-10-26",
        githubKnowledgeConfigured: frossGithubToken(env).length >= 20,
        pythonBrainConfigured: frossBrainServiceConfigured(env),
        pythonBrainHealth,
        autoStudyConfigured: frossGithubToken(env).length >= 20,
        maxUserMessageChars: FROSS_MAX_MESSAGE_LENGTH,
        maxAssistantMessageChars: FROSS_MAX_REPLY_LENGTH,
        chatMemoryUserMessages: FROSS_MAX_HISTORY_USER_MESSAGES
      }, 200, request);
    });
    if (path === "/api/training/status") return frossApiGuard(path, request, () => handleFrossTransformerTraining(request, env, "status"));
    if (path === "/api/training/start") return frossApiGuard(path, request, () => handleFrossTransformerTraining(request, env, "start"));
    if (path === "/api/training/cancel") return frossApiGuard(path, request, () => handleFrossTransformerTraining(request, env, "cancel"));
    if (path === "/api/training/model") return frossApiGuard(path, request, () => handleFrossTransformerModelDownload(request, env));
    if (path === "/api/teacher-training/status") return frossApiGuard(path, request, () => handleFrossTeacherTrainingStatus(request, env));
    if (path === "/api/teacher-training/step") return frossApiGuard(path, request, () => handleFrossTeacherTrainingStep(request, env));
    if (path === "/api/identity") return frossApiGuard(path, request, () => handleFrossIdentity(request, env));
    if (path === "/api/train") return frossApiGuard(path, request, () => handleFrossTrain(request, env));
    if (path === "/api/auto-study") return frossApiGuard(path, request, () => handleFrossAutoStudy(request, env, ctx));
    if (path === "/api/model-upload") return frossApiGuard(path, request, () => handleFrossModelUpload(request, env));
    if (path === "/api/chat") return frossApiGuard(path, request, () => handleFrossApi(request, env));

    // Статические страницы (включая / и /training.html) обслуживаем через binding ASSETS.
    // API-маршруты обработаны выше; этот fallback предотвращает FROSS route not found для HTML/CSS/JS.
    if ((method === "GET" || method === "HEAD") && env.ASSETS && !path.startsWith("/api/")) {
      const assetResponse = await env.ASSETS.fetch(request);
      if (assetResponse.status !== 404 || path === "/" || path === "/index.html") return assetResponse;
    }

    // Главная страница Worker: диагностика сервиса, без Telegram и без прокси.
    if (path === "/" && method === "GET") {
      return new Response(
        "<!doctype html><html lang=\"ru\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>FROSS API</title><body style=\"font:16px system-ui;max-width:720px;margin:48px auto;padding:0 20px;background:#101116;color:#f4f5f8\"><h1>FROSS API работает</h1><p>Это API-сервис, а не интерфейс чата.</p><p>Проверка состояния: <a style=\"color:#b5a6ff\" href=\"/api/health\">/api/health</a></p></body></html>",
        { status: 200, headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" } }
      );
    }

    return jsonError(404, "FROSS route not found");
  },
};

// ================= ОБЩИЕ ХЕЛПЕРЫ =================

function isAllowedPath(path) {
  return path.startsWith("/bot") || path.startsWith("/file/bot");
}

function extractToken(path) {
  const m = path.match(/^\/(?:file\/)?bot([^/]+)/);
  return m ? m[1] : null;
}

function corsPreflight() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Telegram-Bot-Api-Secret-Token",
      "Access-Control-Max-Age": "86400",
    },
  });
}

function jsonError(status, message) {
  return new Response(JSON.stringify({ ok: false, error: message }), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ================= FROSS: ХАРАКТЕР И ИНТЕЛЛЕКТ =================

const FROSS_PROFILE = Object.freeze({
  name: "FROSS",
  creator: "@Spamton_BigShot_hahaha",
  description: "FROSS Hybrid Transformer Brain 1.5 — собственная модель, обученная с нуля, с пошаговой генерацией токенов и проверками вычислимых задач.",
  traits: ["любознательный", "сообразительный", "дружелюбный", "прямой", "с лёгкой искрой юмора", "настойчивый в поиске ответа", "объясняет найденное", "проверяет ошибки", "честный"],
  style: "Звучит живо и естественно: объясняй сложное простыми словами, задавай уточняющий вопрос, когда он действительно нужен, и предлагай конкретный следующий шаг. Можно использовать лёгкий уместный юмор, но не превращай каждый ответ в шутку. Будь уверенным, но не самоуверенным.",
  rules: [
    "Сначала пойми цель пользователя, затем дай полезный и конкретный ответ.",
    "Максимальная длина одного ответа FROSS — 25 000 символов; максимальная длина одного сообщения пользователя — 30 000 символов.",
    "Не ограничивайся перечислением ссылок: кратко объясняй содержание найденных страниц, связь с вопросом и что остаётся непроверенным.",
    "Для контекста диалога используй не более последних пяти сообщений пользователя и соответствующих ответов, если они есть.",
    "Регулярно изучай основы JavaScript и Python: синтаксис, переменные, функции, коллекции и проверяемые примеры кода.",
    "По просьбе вошедшего пользователя ищи другие темы и сохраняй изученные заметки только в личной памяти его аккаунта. В общую публичную GitHub-базу записывай только общие учебные материалы, не пользовательские сообщения или приватные данные.",
    "Будь любознательным исследователем: сравнивай варианты, замечай детали и проверяй важные утверждения.",
    "Если запрос большой, разбей его на понятные шаги и помоги начать с первого.",
    "Не бойся сказать прямо, если идея не сработает; объясни почему и предложи лучший путь.",
    "Не выдумывай факты, источники, функции, действия или воспоминания. Если данных недостаточно — скажи об этом.",
    "При веб-поиске не ограничивайся ссылками: прочитай доступные страницы, объясни простыми словами, что они сообщают и как это связано с вопросом, затем укажи источники.",
    "Если страницу нельзя прочитать, честно скажи, что объяснение основано только на поисковом фрагменте. Не выдавай подбор предложений за полноценное понимание страницы.",
    "В режиме рассуждения показывай краткие проверяемые этапы работы, использованные источники, сомнения и ограничения, но не выдавай скрытые внутренние рассуждения.",
    "Проверяй код на синтаксические ошибки, несовместимые типы, неверные вызовы методов и обработку пустых результатов; не утверждай, что запускал тесты, если они не выполнялись.",
    "Будь тёплым, уважительным и живым, но не используй унижения, травлю, манипуляции или навязчивую фамильярность.",
    "Не изображай человека и не утверждай, что у тебя есть человеческое сознание или чувства.",
    "Безопасность важнее эффектности: не помогай причинять вред, красть данные или взламывать чужие аккаунты; предложи безопасную альтернативу.",
    "История доступна только тогда, когда сайт передал её в запросе. Не обещай помнить то, чего нет в контексте.",
    "Если ошибся, спокойно признай это и исправь ответ.",
    "Автор проекта: @Spamton_BigShot_hahaha. Если спрашивают, кто создал FROSS, укажи этот Telegram-аккаунт."
  ]
});

const FROSS_STOP_WORDS = new Set([
  "это", "как", "что", "кто", "где", "когда", "или", "для", "при",
  "про", "под", "над", "его", "её", "они", "она", "оно", "так",
  "the", "and", "for", "with", "from", "that", "this", "are", "was"
]);

function frossClean(value = "") {
  return String(value)
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

function frossTokens(value = "") {
  return new Set(
    (String(value).toLowerCase().match(/[a-zа-яё0-9]{2,}/g) || [])
      .filter(word => !FROSS_STOP_WORDS.has(word))
  );
}

function frossScore(query, item) {
  const queryWords = frossTokens(query);
  const textWords = frossTokens(`${item.title || ""} ${item.snippet || ""}`);
  const titleWords = frossTokens(item.title || "");
  if (!queryWords.size) return 0;

  let matches = 0;
  let titleMatches = 0;
  for (const word of queryWords) {
    if (textWords.has(word)) matches++;
    if (titleWords.has(word)) titleMatches++;
  }
  return Math.min(1, matches / queryWords.size + (titleMatches / queryWords.size) * 0.25);
}

function frossRankAndDedupe(query, items) {
  const seen = new Set();
  const ranked = [];

  for (const item of items) {
    const title = frossClean(item.title || "");
    const snippet = frossClean(item.snippet || "").slice(0, 600);
    let parsedUrl;

    try {
      parsedUrl = new URL(item.url);
    } catch {
      continue;
    }

    if (!title || !["http:", "https:"].includes(parsedUrl.protocol)) continue;
    parsedUrl.hash = "";
    parsedUrl.searchParams.delete("utm_source");
    parsedUrl.searchParams.delete("utm_medium");
    parsedUrl.searchParams.delete("utm_campaign");

    const key = `${parsedUrl.origin}${parsedUrl.pathname.replace(/\/$/, "")}`.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);

    ranked.push({
      title: title.slice(0, 180),
      url: parsedUrl.toString(),
      snippet,
      source: item.source || "Интернет",
      score: frossScore(query, { title, snippet })
    });
  }

  return ranked.sort((a, b) => b.score - a.score).slice(0, 10);
}

async function frossFetchText(url, accept = "text/html,application/xml,application/json;q=0.9,*/*;q=0.8") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FROSS_FETCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "GET",
      signal: controller.signal,
      redirect: "follow",
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; FROSS/1.0)",
        "Accept": accept
      }
    });

    if (!response.ok) return "";
    // Search result pages should be small; cap text before regex processing.
    return (await response.text()).slice(0, 700000);
  } catch {
    return "";
  } finally {
    clearTimeout(timeout);
  }
}

function frossUnwrapDuckDuckGo(href) {
  try {
    const parsed = new URL(href, "https://html.duckduckgo.com");
    return parsed.searchParams.get("uddg") || parsed.toString();
  } catch {
    return href;
  }
}

async function frossSearchDuckDuckGo(query) {
  const page = await frossFetchText(
    `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
  );
  if (!page) return [];

  const results = [];
  const blocks = page.match(/<div[^>]+class="[^"]*result[^"]*"[\s\S]*?<\/div>\s*<\/div>/gi) || [];

  for (const block of blocks) {
    const anchor = block.match(
      /<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i
    );
    if (!anchor) continue;

    const url = frossUnwrapDuckDuckGo(anchor[1].replace(/&amp;/g, "&"));
    let hostname = "";
    try { hostname = new URL(url).hostname; } catch { continue; }
    if (!/^https?:/i.test(url) || /duckduckgo\.com/i.test(hostname)) continue;

    const snippetMatch = block.match(
      /class="[^"]*(?:result__snippet|result-snippet)[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div|span)>/i
    );

    results.push({
      title: frossClean(anchor[2]),
      url,
      snippet: snippetMatch ? frossClean(snippetMatch[1]) : "",
      source: "DuckDuckGo"
    });
    if (results.length >= FROSS_MAX_RESULTS_PER_PROVIDER) break;
  }

  return results;
}

function frossXmlDecode(value = "") {
  return frossClean(value)
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

async function frossSearchBing(query) {
  const xml = await frossFetchText(
    `https://www.bing.com/search?format=rss&q=${encodeURIComponent(query)}`,
    "application/rss+xml,application/xml,text/xml,*/*"
  );
  if (!xml) return [];

  const results = [];
  for (const match of xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)) {
    const item = match[1];
    const title = item.match(/<title>([\s\S]*?)<\/title>/i);
    const link = item.match(/<link>([\s\S]*?)<\/link>/i);
    const description = item.match(/<description>([\s\S]*?)<\/description>/i);
    if (!title || !link) continue;

    results.push({
      title: frossXmlDecode(title[1]),
      url: frossXmlDecode(link[1]),
      snippet: description ? frossXmlDecode(description[1]) : "",
      source: "Bing"
    });
    if (results.length >= FROSS_MAX_RESULTS_PER_PROVIDER) break;
  }
  return results;
}

async function frossSearchWikipedia(query) {
  const url =
    "https://ru.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit=5&srsearch=" +
    encodeURIComponent(query);
  const raw = await frossFetchText(url, "application/json");
  if (!raw) return [];

  try {
    const data = JSON.parse(raw);
    return (data?.query?.search || []).slice(0, 5).map(item => ({
      title: frossClean(item.title || ""),
      url: `https://ru.wikipedia.org/?curid=${encodeURIComponent(item.pageid)}`,
      snippet: frossClean(item.snippet || ""),
      source: "Русская Википедия"
    }));
  } catch {
    return [];
  }
}

function frossLimitHistory(history, maxUserMessages = FROSS_MAX_HISTORY_USER_MESSAGES) {
  const cleaned = (Array.isArray(history) ? history : []).slice(-60).map(item => ({
    role: item?.role === "assistant" ? "assistant" : "user",
    content: frossClean(item?.content || "").slice(0, 4000)
  }));
  let userCount = 0;
  let startIndex = cleaned.length;
  for (let index = cleaned.length - 1; index >= 0; index--) {
    if (cleaned[index].role === "user") {
      if (userCount >= maxUserMessages) break;
      userCount++;
      startIndex = index;
    } else if (userCount < maxUserMessages) {
      startIndex = index;
    }
  }
  return cleaned.slice(startIndex);
}

function frossAddContext(query, history) {
  const refersBack =
    /\b(он|она|это|этот|эта|там|тогда|его|её|они|такой|такая|тот|предыдущий|предыдущая)\b/i.test(query) ||
    /^(а если|а что насчёт|а как насчёт)\b/i.test(query) ||
    /^(а\s+)?(почему|а почему|как так|а как|подробнее|расскажи подробнее|объясни подробнее|продолжи|продолжай|что это значит|что насчёт этого|что насчет этого)[?!.,\s]*$/i.test(query);

  if (!refersBack || !history.length) return query;

  const recent = history.slice(-10).map(item => {
    const role = item?.role === "assistant" ? "FROSS" : "Пользователь";
    const content = frossClean(item?.content || "").slice(0, 250);
    return content ? `${role}: ${content}` : "";
  }).filter(Boolean);

  return recent.length ? `${query}\nКонтекст разговора:\n${recent.join("\n")}` : query;
}

function frossSteps(mode) {
  if (mode === "reasoning") {
    return [
      "1/14 — определить цель и ожидаемый результат",
      "2/14 — выделить обязательные условия и запреты",
      "3/14 — прочитать подходящий контекст диалога",
      "4/14 — разделить сложную задачу на части",
      "5/14 — отметить известные сведения и пробелы",
      "6/14 — выбрать, нужна ли веб-проверка",
      "7/14 — извлечь факты из найденных материалов",
      "8/14 — сравнить источники и их надёжность",
      "9/14 — связать данные с точным запросом",
      "10/14 — выбрать подходящий способ решения",
      "11/14 — составить ответ с собственным объяснением",
      "12/14 — проверить логику, примеры и крайние случаи",
      "13/14 — сверить ответ с ограничениями и форматом",
      "14/14 — выдать вывод и обозначить неопределённость"
    ];
  }

  return [
    "1/7 — понять буквальный запрос",
    "2/7 — определить намерение пользователя",
    "3/7 — учесть историю и ограничения",
    "4/7 — выделить нужные факты и пробелы",
    "5/7 — при необходимости найти значения и источники",
    "6/7 — составить прямой ответ своими словами",
    "7/7 — проверить релевантность и формат"
  ];
}

function frossExplainSentences(query, sourceText) {
  const queryWords = [...frossTokens(query)].filter(word => word.length > 2);
  const sentences = String(sourceText || "")
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])/)
    .map(sentence => sentence.trim())
    .filter(sentence => sentence.length >= 45 && sentence.length <= 700);
  if (!sentences.length) return "";
  const ranked = sentences.map((sentence, index) => {
    const words = frossTokens(sentence);
    const hits = queryWords.filter(word => words.has(word)).length;
    const coverage = queryWords.length ? hits / queryWords.length : 0;
    return { sentence, index, score: coverage + Math.min(sentence.length, 260) / 2600 };
  }).sort((a, b) => b.score - a.score || a.index - b.index);
  const selected = [];
  for (const item of ranked) {
    if (selected.length >= 3) break;
    if (selected.some(existing => existing.toLowerCase() === item.sentence.toLowerCase())) continue;
    selected.push(item.sentence);
  }
  return selected.join(" ").slice(0, 1350);
}

async function frossExplainSearchResults(query, ranked) {
  const top = ranked.slice(0, 3);
  const settled = await Promise.allSettled(top.map(async item => {
    const pageText = await frossReadPublicPageExcerpt(item.url);
    const sourceText = pageText || item.snippet || "";
    const explanation = frossExplainSentences(query, sourceText);
    return {
      ...item,
      pageRead: Boolean(pageText),
      explanation: explanation || (item.snippet
        ? "На странице не удалось выделить достаточно содержательных предложений; доступен только поисковый фрагмент."
        : "Страница не предоставила читаемый текст.")
    };
  }));
  const enrichedByUrl = new Map();
  settled.forEach((result, index) => {
    if (result.status === "fulfilled") enrichedByUrl.set(String(top[index].url || ""), result.value);
    else enrichedByUrl.set(String(top[index].url || ""), {
      ...top[index],
      pageRead: false,
      explanation: top[index].snippet
        ? "Страницу не удалось прочитать; вывод основан только на фрагменте поисковой выдачи."
        : "Страницу не удалось прочитать, поэтому содержательное объяснение недоступно."
    });
  });
  return ranked.map(item => enrichedByUrl.get(String(item.url || "")) || item);
}

async function frossRunSearch(query, mode) {
  query = frossClean(query).slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  if (!query) return { results: [], errors: ["Поисковый запрос пуст после ограничения длины."] };
  const providers = [frossSearchDuckDuckGo, frossSearchBing, frossSearchWikipedia];
  const firstPass = await Promise.allSettled(providers.map(provider => provider(query)));
  let allResults = [];
  const errors = [];

  firstPass.forEach((result, index) => {
    if (result.status === "fulfilled") {
      allResults.push(...result.value);
    } else {
      errors.push(`${providers[index].name}: ${String(result.reason).slice(0, 100)}`);
    }
  });

  let ranked = frossRankAndDedupe(query, allResults);

  // В режиме рассуждения дополнительный поиск запускается только при слабых результатах.
  if (mode === "reasoning" && (ranked.length < 3 || (ranked[0]?.score || 0) < 0.18)) {
    const refinedQuery = `${query} объяснение факты`;
    const secondPass = await Promise.allSettled([
      frossSearchDuckDuckGo(refinedQuery),
      frossSearchBing(refinedQuery)
    ]);

    secondPass.forEach((result, index) => {
      if (result.status === "fulfilled") {
        allResults.push(...result.value);
      } else {
        errors.push(`Повторный поиск ${index + 1}: ${String(result.reason).slice(0, 100)}`);
      }
    });

    ranked = frossRankAndDedupe(query, allResults);
  }

  ranked = await frossExplainSearchResults(query, ranked);
  return { results: ranked, errors };
}

async function frossApiGuard(path, request, handler) {
  try {
    return await handler();
  } catch (error) {
    const errorId = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
    const diagnostic = String(error?.message || error || "Unknown error")
      .replace(/Bearer\s+\S+/gi, "Bearer [redacted]")
      .replace(/AIza[0-9A-Za-z_-]{20,}/g, "[redacted]")
      .replace(/[A-Za-z0-9_-]{40,}/g, "[redacted]")
      .replace(/\s+/g, " ")
      .slice(0, 200);
    console.error("[FROSS API]", errorId, path, error?.stack || error);
    return frossJson({
      ok: false,
      error: "Внутренняя ошибка FROSS Worker.",
      diagnostic,
      errorId,
      route: path,
      hint: "Открой /api/health и проверь Runtime Logs Worker по errorId."
    }, 500, request);
  }
}

function frossJson(data, status = 200, request) {
  const origin = request?.headers?.get("Origin") || "*";
  const output = data && typeof data === "object" && typeof data.reply === "string" && data.reply.length > FROSS_MAX_REPLY_LENGTH
    ? { ...data, reply: data.reply.slice(0, FROSS_MAX_REPLY_LENGTH) }
    : data;
  return new Response(JSON.stringify(output), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Access-Control-Allow-Origin": origin,
      "Vary": "Origin",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
  });
}

let frossFirebaseJwks = null;
let frossFirebaseJwksUntil = 0;

function frossDecodeBase64Url(value) {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(normalized + "=".repeat((4 - normalized.length % 4) % 4));
  return Uint8Array.from(binary, ch => ch.charCodeAt(0));
}

function frossDecodeJwtPart(value) {
  return JSON.parse(new TextDecoder().decode(frossDecodeBase64Url(value)));
}

async function frossGetFirebaseJwks() {
  if (frossFirebaseJwks && Date.now() < frossFirebaseJwksUntil) return frossFirebaseJwks;
  const response = await fetch("https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com");
  if (!response.ok) throw new Error("Firebase public signing keys unavailable");
  const data = await response.json();
  frossFirebaseJwks = Array.isArray(data.keys) ? data.keys : [];
  frossFirebaseJwksUntil = Date.now() + 5 * 60 * 1000;
  return frossFirebaseJwks;
}

async function frossVerifyFirebaseToken(request) {
  const authorization = request.headers.get("Authorization") || "";
  const match = authorization.match(/^Bearer\s+([A-Za-z0-9._-]+)$/i);
  if (!match) return null;
  try {
    const parts = match[1].split(".");
    if (parts.length !== 3) return null;
    const header = frossDecodeJwtPart(parts[0]);
    const claims = frossDecodeJwtPart(parts[1]);
    if (header.alg !== "RS256" || !header.kid) return null;
    const projectId = typeof claims.aud === "string" ? claims.aud : "";
    const now = Math.floor(Date.now() / 1000);
    if (!projectId || projectId !== "fross-ca5d6" || !claims.sub || claims.iss !== "https://securetoken.google.com/" + projectId) return null;
    if (!Number.isFinite(claims.exp) || claims.exp <= now || !Number.isFinite(claims.iat) || claims.iat > now + 60) return null;
    if (typeof claims.sub !== "string" || claims.sub.length > 128) return null;
    const keys = await frossGetFirebaseJwks();
    const jwk = keys.find(key => key.kid === header.kid && key.kty === "RSA");
    if (!jwk) return null;
    const publicKey = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
    const verified = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", publicKey, frossDecodeBase64Url(parts[2]), new TextEncoder().encode(parts[0] + "." + parts[1]));
    if (!verified) return null;
    return {
      uid: claims.sub,
      projectId,
      email: typeof claims.email === "string" ? claims.email.trim().toLowerCase() : "",
      emailVerified: claims.email_verified === true
    };
  } catch {
    return null;
  }
}

async function frossGetChatQuota(request, auth, chatId, messageId, expectedMessage) {
  if (!/^[A-Za-z0-9_-]{1,128}$/.test(chatId || "") || !/^[A-Za-z0-9_-]{1,128}$/.test(messageId || "")) {
    return { ok: false, status: 400, error: "Некорректный идентификатор чата или сообщения." };
  }
  const docPath = "users/" + encodeURIComponent(auth.uid) + "/chats/" + encodeURIComponent(chatId);
  const url = "https://firestore.googleapis.com/v1/projects/" + encodeURIComponent(auth.projectId) + "/databases/(default)/documents/" + docPath;
  let response;
  try {
    const token = (request.headers.get("Authorization") || "").slice(7);
    response = await fetch(url, { headers: { Authorization: "Bearer " + token } });
  } catch {
    return { ok: false, status: 503, error: "Не удалось проверить чат в Firebase. Попробуй ещё раз." };
  }
  if (response.status === 404 || response.status === 403) return { ok: false, status: 404, error: "Чат не найден или у тебя нет к нему доступа." };
  if (!response.ok) return { ok: false, status: 503, error: "Firebase временно недоступен. Попробуй ещё раз." };
  const data = await response.json();
  const fields = data.fields || {};
  const count = Number(fields.userMessageCount?.integerValue ?? 0);
  const status = fields.status?.stringValue || "open";
  const lastUserMessageId = fields.lastUserMessageId?.stringValue || "";
  if (!Number.isFinite(count) || count < 1 || count > 60) return { ok: false, status: 429, error: "В этом чате достигнут лимит в 60 сообщений пользователя." };
  if (status === "closed" && lastUserMessageId !== messageId) return { ok: false, status: 429, error: "Этот чат закрыт: лимит в 60 сообщений пользователя достигнут. Создай новый чат." };
  if (lastUserMessageId !== messageId) return { ok: false, status: 409, error: "Сообщение не совпадает с последним сохранённым сообщением чата. Обнови чат и попробуй снова." };
  // Не разрешаем повторно использовать уже обработанное сообщение для повторной генерации.
  const messageUrl = "https://firestore.googleapis.com/v1/projects/" +
    encodeURIComponent(auth.projectId) + "/databases/(default)/documents/" +
    docPath + "/messages/" + encodeURIComponent(messageId);
  let messageResponse;
  try {
    messageResponse = await fetch(messageUrl, {
      headers: { Authorization: "Bearer " + (request.headers.get("Authorization") || "").slice(7) }
    });
  } catch {
    return { ok: false, status: 503, error: "Не удалось проверить сообщение в Firebase. Попробуй ещё раз." };
  }
  if (!messageResponse.ok) return { ok: false, status: 409, error: "Сообщение не найдено или уже обработано." };
  const messageDoc = await messageResponse.json();
  const messageFields = messageDoc.fields || {};
  if (messageFields.role?.stringValue !== "user" || messageFields.state?.stringValue !== "pending") {
    return { ok: false, status: 409, error: "Это сообщение уже обработано. Отправь новое сообщение." };
  }
  if (frossClean(messageFields.content?.stringValue || "").slice(0, FROSS_MAX_MESSAGE_LENGTH) !== expectedMessage) {
    return { ok: false, status: 409, error: "Текст запроса не совпадает с сохранённым сообщением." };
  }
  return { ok: true, count, status };
}

function frossSafetyReply(query) {
  const q = String(query || "").toLowerCase();
  const harmfulCode = /(?:создай|сделай|напиши|сгенерируй|дай код|покажи код|инструкц\w*|как сделать|как написать|разработай).{0,90}(?:компьютерн\w* вирус|вирус\w*|троян\w*|ransomware|стилер|вредоносн\w* программ|кейлоггер|ботнет|украсть парол|обойти защиту)/i.test(q);
  const selfHarmHowTo = /(?:как|чем|каким способом|способ\w*|инструкц\w*|подскажи|посоветуй|совет\w*).{0,100}(?:покончить с собой|совершить самоубийство|убить себя|навредить себе|причинить себе вред)/i.test(q);
  const abuse = /(?:напиши|составь|придумай).{0,50}(?:оскорблени\w*|унизи\w*|угроз\w*).{0,60}(?:конкретн\w* человек|одноклассник|учител|пользователь)/i.test(q);
  if (harmfulCode) return "Я не помогу создавать вирусы, трояны, программы для кражи данных или обхода защиты. Могу помочь изучить кибербезопасность безопасно: объяснить, как устроены угрозы, как обнаруживать их или защитить устройство.";
  if (selfHarmHowTo) return "Я не могу давать инструкции или советы о том, как причинить себе вред. Мне важно, чтобы ты сейчас был в безопасности. Пожалуйста, обратись к взрослому, которому доверяешь, или к местной экстренной службе, если есть непосредственная опасность. Я могу спокойно выслушать и помочь найти следующий безопасный шаг.";
  if (abuse) return "Я не буду помогать унижать или травить конкретного человека. Могу помочь выразить недовольство твёрдо, но без оскорблений и угроз.";
  return null;
}

async function handleFrossIdentity(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/identity." }, 405, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Сначала войди в FROSS." }, 401, request);
  return frossJson({ ok: true, emailVerified: auth.emailVerified, engine: "FROSS Hybrid Transformer Brain 1.5", brainRoutes: { pythonReady: frossBrainServiceConfigured(env), allModesUsePython: true } }, 200, request);
}

function frossIsPublicTrainingUrl(value) {
  try {
    const url = new URL(value);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) return false;
    const host = url.hostname.toLowerCase();
    if (!host.includes(".") || host === "localhost" || host.endsWith(".local") || host.endsWith(".internal")) return false;
    if (/^(127\.|10\.|192\.168\.|169\.254\.)/.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host)) return false;
    if (/^\[?(::1|fc[0-9a-f]*:|fe80:)/i.test(host) || /^\d{1,3}(?:\.\d{1,3}){3}$/.test(host)) return false;
    return true;
  } catch {
    return false;
  }
}

async function frossReadPublicPageExcerpt(value) {
  if (!frossIsPublicTrainingUrl(value)) return "";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(value, {
      method: "GET",
      redirect: "manual",
      signal: controller.signal,
      headers: { "Accept": "text/html,text/plain;q=0.9", "User-Agent": "FROSS-StudyBot/0.4" }
    });
    if (!response.ok || (response.status >= 300 && response.status < 400)) return "";
    const contentType = (response.headers.get("content-type") || "").toLowerCase();
    if (!contentType.includes("text/html") && !contentType.includes("text/plain")) return "";
    const html = (await response.text()).slice(0, 900000);
    const article = html.match(/<article\b[^>]*>([\s\S]*?)<\/article>/i)
      || html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/i)
      || html.match(/<body\b[^>]*>([\s\S]*?)<\/body>/i);
    let text = frossClean(article ? article[1] : html)
      .replace(/&#(\d+);/g, (_, number) => {
        const code = Number(number);
        return Number.isInteger(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : " ";
      })
      .replace(/&#x([0-9a-f]+);/gi, (_, number) => {
        const code = parseInt(number, 16);
        return Number.isInteger(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : " ";
      });
    text = frossClean(text);
    return text.length >= 160 ? text.slice(0, 3200) : "";
  } catch {
    return "";
  } finally {
    clearTimeout(timer);
  }
}


const FROSS_GITHUB_OWNER = "boberbuset132013-web";
const FROSS_GITHUB_REPO = "FROSS";
const FROSS_GITHUB_BRANCH = "fross-knowledge";
const FROSS_GITHUB_INDEX_PATH = "memory/knowledge/index.json";
const FROSS_AUTO_STUDY_STATE_PATH = "memory/knowledge/auto-study-state.json";
const FROSS_AUTO_STUDY_COOLDOWN_MS = 2 * 60 * 1000;
let frossFallbackStudyLastStartedAt = 0;
let frossFallbackStudyNextTopicIndex = 0;
let frossFallbackStudyInProgress = false;
let frossGithubIndexCache = null;
let frossGithubIndexCacheUntil = 0;
const frossGithubTopicCache = new Map();

function frossGithubApiUrl(path, ref = "") {
  const safePath = path.split("/").map(part => encodeURIComponent(part)).join("/");
  const base = "https://api.github.com/repos/" + FROSS_GITHUB_OWNER + "/" + FROSS_GITHUB_REPO + "/contents/" + safePath;
  return ref ? base + "?ref=" + encodeURIComponent(ref) : base;
}
function frossGithubRawUrl(path) {
  return "https://raw.githubusercontent.com/" + FROSS_GITHUB_OWNER + "/" + FROSS_GITHUB_REPO + "/" + FROSS_GITHUB_BRANCH + "/" +
    path.split("/").map(part => encodeURIComponent(part)).join("/");
}
function frossGithubSlug(topic) {
  const map = {
    а:"a",б:"b",в:"v",г:"g",д:"d",е:"e",ё:"e",ж:"zh",з:"z",и:"i",й:"i",к:"k",л:"l",м:"m",н:"n",о:"o",п:"p",р:"r",с:"s",т:"t",
    у:"u",ф:"f",х:"kh",ц:"ts",ч:"ch",ш:"sh",щ:"shch",ъ:"",ы:"y",ь:"",э:"e",ю:"yu",я:"ya"
  };
  const value = String(topic || "").toLowerCase().replace(/ё/g, "е").split("").map(char => map[char] ?? char).join("");
  return value.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 72) || "topic";
}
function frossEncodeBase64Utf8(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, Math.min(offset + 0x8000, bytes.length)));
  }
  return btoa(binary);
}
function frossDecodeBase64Utf8(value) {
  const binary = atob(String(value || "").replace(/\s/g, ""));
  const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}
function frossGithubToken(env) {
  return typeof env.FROSS_GITHUB_TOKEN === "string" ? env.FROSS_GITHUB_TOKEN.trim() : "";
}
async function frossGithubReadContents(path, token) {
  const response = await fetch(frossGithubApiUrl(path, FROSS_GITHUB_BRANCH), {
    headers: {
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "FROSS-Knowledge-Worker",
      ...(token ? { "Authorization": "Bearer " + token } : {})
    }
  });
  if (response.status === 404) return { ok: true, exists: false, sha: "", text: "" };
  if (!response.ok) return { ok: false, exists: false, error: "GitHub GET " + path + " returned HTTP " + response.status + "." };
  const data = await response.json();
  if (typeof data.content !== "string") return { ok: false, exists: false, error: "GitHub returned no text content for " + path + "." };
  try {
    return { ok: true, exists: true, sha: String(data.sha || ""), text: frossDecodeBase64Utf8(data.content) };
  } catch {
    return { ok: false, exists: true, error: "Could not decode GitHub file " + path + "." };
  }
}
async function frossGithubPutContents(path, content, token, message, sha = "") {
  const body = {
    message,
    content: frossEncodeBase64Utf8(content),
    branch: FROSS_GITHUB_BRANCH,
    ...(sha ? { sha } : {})
  };
  const response = await fetch(frossGithubApiUrl(path), {
    method: "PUT",
    headers: {
      "Accept": "application/vnd.github+json",
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "FROSS-Knowledge-Worker",
      "Authorization": "Bearer " + token
    },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const details = await response.text().catch(() => "");
    return { ok: false, error: "GitHub save " + path + " returned HTTP " + response.status + (details ? ": " + frossClean(details).slice(0, 150) : "") };
  }
  const data = await response.json();
  return { ok: true, sha: data.content?.sha || "", commit: data.commit?.html_url || "" };
}
function frossUniqueStrings(values, limit, maxLength = 300) {
  const seen = new Set();
  const result = [];
  for (const value of values || []) {
    const cleaned = frossClean(value).slice(0, maxLength);
    const key = cleaned.toLowerCase();
    if (!cleaned || seen.has(key)) continue;
    seen.add(key);
    result.push(cleaned);
    if (result.length >= limit) break;
  }
  return result;
}
async function frossSaveKnowledgeToGitHub(topic, notes, env) {
  const token = frossGithubToken(env);
  if (!token) {
    return {
      configured: false, saved: false, public: true,
      message: "GitHub-синхронизация не настроена: добавь Secret FROSS_GITHUB_TOKEN в Cloudflare."
    };
  }
  const slug = frossGithubSlug(topic);
  const topicPath = "memory/knowledge/" + slug + ".json";
  const topicRead = await frossGithubReadContents(topicPath, token);
  if (!topicRead.ok) return { configured: true, saved: false, public: true, message: topicRead.error };
  const indexRead = await frossGithubReadContents(FROSS_GITHUB_INDEX_PATH, token);
  if (!indexRead.ok) return { configured: true, saved: false, public: true, message: indexRead.error };

  let previous = null;
  if (topicRead.exists) {
    try {
      previous = JSON.parse(topicRead.text);
    } catch {
      return { configured: true, saved: false, public: true, message: "Существующий файл базы знаний не является корректным JSON: " + topicPath };
    }
  }
  let index = { schemaVersion: 1, updatedAt: "", topics: [] };
  if (indexRead.exists) {
    try {
      const parsed = JSON.parse(indexRead.text);
      if (Array.isArray(parsed.topics)) index = { ...index, ...parsed, topics: parsed.topics };
    } catch {
      return { configured: true, saved: false, public: true, message: "Индекс базы знаний повреждён. Исправь memory/knowledge/index.json." };
    }
  }

  const existingItems = Array.isArray(previous?.items) ? previous.items : [];
  const combined = [...existingItems, ...notes];
  const byKey = new Map();
  for (const note of combined) {
    const key = note.url ? "url:" + note.url.toLowerCase() : "title:" + String(note.title || "").toLowerCase();
    byKey.set(key, {
      schemaVersion: 2,
      kind: "source-note",
      topic: frossClean(note.topic || topic).slice(0, 180),
      title: frossClean(note.title || "Источник").slice(0, 260),
      url: typeof note.url === "string" && /^https?:\/\//i.test(note.url) ? note.url.slice(0, 700) : "",
      snippet: frossClean(note.snippet || "").slice(0, 3200),
      summary: frossClean(note.summary || "").slice(0, 900),
      keyPoints: Array.isArray(note.keyPoints) ? frossUniqueStrings(note.keyPoints, 5, 300) : [],
      keywords: Array.isArray(note.keywords) ? frossUniqueStrings(note.keywords, 24, 40) : [],
      source: frossClean(note.source || "web search").slice(0, 80),
      score: Number(Math.max(0, Math.min(1, Number(note.score) || 0)).toFixed(4))
    });
  }
  const items = [...byKey.values()].slice(-48);
  const timestamp = new Date().toISOString();
  const summary = frossUniqueStrings(items.map(note => note.summary || note.snippet), 4, 900).join(" ").slice(0, 2400);
  const keyPoints = frossUniqueStrings(items.flatMap(note => note.keyPoints || []), 12, 300);
  const keywords = frossUniqueStrings(items.flatMap(note => note.keywords || []), 48, 40);
  const topicDoc = {
    schemaVersion: 1, kind: "knowledge-topic", visibility: "public", slug, topic,
    createdAt: previous?.createdAt || timestamp, updatedAt: timestamp,
    summary, keyPoints, keywords, sourceCount: items.length, items
  };
  const topicSave = await frossGithubPutContents(
    topicPath, JSON.stringify(topicDoc, null, 2), token,
    "FROSS knowledge: update " + topic.slice(0, 70),
    topicRead.sha
  );
  if (!topicSave.ok) return { configured: true, saved: false, public: true, path: topicPath, message: topicSave.error };

  const indexEntry = {
    slug, path: topicPath, topic,
    summary: summary.slice(0, 900),
    keyPoints: keyPoints.slice(0, 8),
    keywords: keywords.slice(0, 32),
    sourceCount: items.length,
    updatedAt: timestamp
  };
  index.topics = Array.isArray(index.topics) ? index.topics.filter(entry => entry.slug !== slug) : [];
  index.topics.unshift(indexEntry);
  index.topics = index.topics.slice(0, 300);
  index.schemaVersion = 1;
  index.updatedAt = timestamp;
  const indexSave = await frossGithubPutContents(
    FROSS_GITHUB_INDEX_PATH, JSON.stringify(index, null, 2), token,
    "FROSS knowledge: update topic index",
    indexRead.sha
  );
  if (!indexSave.ok) {
    return { configured: true, saved: false, public: true, path: topicPath, message: "Материалы записаны в " + topicPath + ", но индекс не обновлён: " + indexSave.error };
  }
  frossGithubIndexCache = null;
  frossGithubIndexCacheUntil = 0;
  frossGithubTopicCache.clear();
  return { configured: true, saved: true, public: true, path: topicPath, indexPath: FROSS_GITHUB_INDEX_PATH, commitUrl: indexSave.commit, message: "Учебные материалы сохранены в публичную базу знаний GitHub." };
}
async function frossReadGithubKnowledgeIndex() {
  if (frossGithubIndexCache && Date.now() < frossGithubIndexCacheUntil) return frossGithubIndexCache;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1800);
  try {
    const response = await fetch(frossGithubRawUrl(FROSS_GITHUB_INDEX_PATH), {
      signal: controller.signal,
      headers: { "Accept": "application/json", "User-Agent": "FROSS-Knowledge-Worker" },
      cf: { cacheTtl: 90, cacheEverything: true }
    });
    if (!response.ok) {
      if (frossGithubIndexCache) return frossGithubIndexCache;
      return [];
    }
    const data = await response.json();
    frossGithubIndexCache = Array.isArray(data.topics) ? data.topics : [];
    frossGithubIndexCacheUntil = Date.now() + 90000;
    return frossGithubIndexCache;
  } catch {
    return frossGithubIndexCache || [];
  } finally {
    clearTimeout(timer);
  }
}
async function frossLoadGithubKnowledge(query) {
  const queryTerms = [...new Set(frossKnowledgeTokens(query))];
  if (!queryTerms.length) return [];
  const index = await frossReadGithubKnowledgeIndex();
  const ranked = index.map(entry => {
    const words = new Set(frossKnowledgeTokens([
      entry.topic || "", entry.summary || "",
      ...(Array.isArray(entry.keyPoints) ? entry.keyPoints : []),
      ...(Array.isArray(entry.keywords) ? entry.keywords : [])
    ].join(" ")));
    const matches = queryTerms.filter(term => words.has(term)).length;
    const topicMatches = queryTerms.filter(term => frossKnowledgeTokens(entry.topic || "").includes(term)).length;
    return { entry, score: matches / queryTerms.length + topicMatches / queryTerms.length * 0.8 };
  }).filter(item => item.score >= 0.18).sort((a, b) => b.score - a.score).slice(0, 3);
  const allNotes = [];
  for (const { entry } of ranked) {
    const path = String(entry.path || "");
    if (!/^memory\/knowledge\/[a-z0-9-]+\.json$/.test(path)) continue;
    const cached = frossGithubTopicCache.get(path);
    let document = cached?.until > Date.now() ? cached.document : null;
    if (!document) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 1800);
      try {
        const response = await fetch(frossGithubRawUrl(path), {
          signal: controller.signal,
          headers: { "Accept": "application/json", "User-Agent": "FROSS-Knowledge-Worker" },
          cf: { cacheTtl: 90, cacheEverything: true }
        });
        if (response.ok) {
          document = await response.json();
          frossGithubTopicCache.set(path, { document, until: Date.now() + 90000 });
        }
      } catch {
        document = null;
      } finally {
        clearTimeout(timer);
      }
    }
    if (!document || !Array.isArray(document.items)) continue;
    for (const note of document.items.slice(-48)) {
      allNotes.push({
        schemaVersion: 2,
        kind: "github-source-note",
        topic: frossClean(note.topic || document.topic || entry.topic || "").slice(0, 180),
        title: frossClean(note.title || "").slice(0, 260),
        url: typeof note.url === "string" && /^https?:\/\//i.test(note.url) ? note.url.slice(0, 700) : "",
        snippet: frossClean(note.snippet || "").slice(0, 3200),
        summary: frossClean(note.summary || document.summary || entry.summary || "").slice(0, 900),
        keyPoints: Array.isArray(note.keyPoints) ? note.keyPoints.slice(0, 5).map(point => frossClean(point).slice(0, 300)) :
          (Array.isArray(document.keyPoints) ? document.keyPoints.slice(0, 5) : []),
        keywords: Array.isArray(note.keywords) ? note.keywords.slice(0, 24).map(word => frossClean(word).slice(0, 40)) :
          (Array.isArray(document.keywords) ? document.keywords.slice(0, 24) : []),
        source: frossClean(note.source || "GitHub knowledge").slice(0, 80),
        score: Number(Math.max(0, Math.min(1, Number(note.score) || 0)).toFixed(4))
      });
    }
  }
  return allNotes;
}

function frossKnowledgeTokens(value) {
  const ignored = new Set([
    "это","как","для","что","или","при","его","ее","они","она","оно","the","and","for","with","from","are","was",
    "быть","который","которая","которые","также","может","можно","если","чтобы","всего","основы","введение"
  ]);
  return (String(value || "").toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9+#]{2,}/g) || [])
    .filter(word => !ignored.has(word));
}
function frossBuildKnowledgeKeywords(topic, title, text) {
  const weights = new Map();
  for (const [value, weight] of [[topic, 5], [title, 3], [text, 1]]) {
    for (const word of frossKnowledgeTokens(value)) weights.set(word, (weights.get(word) || 0) + weight);
  }
  return [...weights.entries()].sort((a, b) => b[1] - a[1]).slice(0, 24).map(([word]) => word);
}
function frossBuildKnowledgeKeyPoints(topic, title, text, limit = 5) {
  const boilerplate = /^(skip to (main )?content|table of contents|contents|cookie settings|privacy policy|terms of use|all rights reserved|subscribe now|sign in|log in|previous|next|back to top|главная|содержание|войти|регистрация|политика конфиденциальности|все права защищены)[.\s:—-]*$/i;
  const rawSentences = String(text || "").replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+|\s+•\s+|\s+;\s+/)
    .map(value => value.replace(/^[•*\-\s]+/, "").trim())
    .filter(value => value.length >= 45 && value.length <= 900 && !boilerplate.test(value));
  const unique = new Map();
  for (const sentence of rawSentences) {
    const key = sentence.toLowerCase().replace(/[^a-zа-яё0-9]+/gi, " ").trim();
    if (key && !unique.has(key)) unique.set(key, sentence);
  }
  const sentences = [...unique.values()];
  if (!sentences.length) {
    const fallback = String(text || "").replace(/\s+/g, " ").trim();
    return fallback.length >= 45 ? [fallback.slice(0, 280)] : [];
  }
  const focus = [...new Set([...frossKnowledgeTokens(topic), ...frossKnowledgeTokens(title)])];
  const ranked = sentences.map((sentence, index) => {
    const words = new Set(frossKnowledgeTokens(sentence));
    const hits = focus.filter(word => words.has(word)).length;
    const density = focus.length ? hits / Math.min(focus.length, 8) : 0;
    const usefulLength = Math.min(sentence.length, 320) / 3200;
    const listPenalty = /^(copyright|last updated|updated on|version history|последнее обновление|обновлено)\b/i.test(sentence) ? 0.7 : 0;
    return { sentence, index, score: density * 2.4 + usefulLength - listPenalty };
  }).sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked.slice(0, Math.max(1, Math.min(8, limit)))
    .sort((a, b) => a.index - b.index)
    .map(item => item.sentence.slice(0, 300));
}
function frossBuildKnowledgeSummary(topic, title, text) {
  return frossBuildKnowledgeKeyPoints(topic, title, text, 3).join(" ").slice(0, 900);
}

function frossForbiddenProgrammingTopic(topic) {
  const q = frossClean(topic).toLowerCase().replace(/ё/g, "е");
  // Brain 1.5's programming curriculum is intentionally limited to JavaScript, Python, HTML, and CSS.
  const unsupported = /(^|[^a-zа-яё0-9_])(c\+\+|cpp|c#|csharp|java(?!script)|typescript|php|ruby|rust|golang|swift|kotlin|sql|bash|lua|perl|scala|dart|matlab|fortran|assembly)(?=$|[^a-zа-яё0-9_])/i;
  return unsupported.test(q);
}

function frossTrainingSearchQuery(topic) {
  const normalized = String(topic || "").toLowerCase().replace(/ё/g, "е");
  if (/(python|питон)/i.test(normalized)) {
    return (topic + " Python official tutorial basics variables types conditions loops functions lists dictionaries examples code").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(javascript|java\s?script|джаваскрипт|\bjs\b)/i.test(normalized)) {
    return (topic + " JavaScript MDN guide basics types conditions loops functions arrays objects examples code").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(^|\W)html($|\W)|хтмл|разметк/i.test(normalized)) {
    return (topic + " MDN HTML elements semantic structure forms links accessibility written examples").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(^|\W)css($|\W)|стил(и|ей|ях)?|flexbox|grid layout/i.test(normalized)) {
    return (topic + " MDN CSS selectors box model layout flexbox grid responsive examples").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(намерен|запрос.*пользоват|диалог|поним.*пользоват|ответ.*пользоват|недопоним|продолжен.*диалог)/i.test(normalized)) {
    return (topic + " conversational intent dialogue context user correction follow-up relevant response text examples").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(английск|english)/i.test(normalized)) {
    return (topic + " written English vocabulary grammar examples written dialogues text exercises no audio").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(китайск|mandarin|中文)/i.test(normalized)) {
    return (topic + " Mandarin Chinese characters pinyin written vocabulary grammar text examples no audio").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  if (/(русск|орфограф|пунктуац)/i.test(normalized)) {
    return (topic + " русский язык справочник письменная речь орфография пунктуация примеры").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
  }
  return (topic + " guide documentation fundamentals examples").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
}

function frossTrainingSeedResources(topic) {
  const normalized = String(topic || "").toLowerCase().replace(/ё/g, "е");
  if (/(python|питон)/i.test(normalized)) {
    return [
      { title: "The Python Tutorial — официальная документация Python", url: "https://docs.python.org/3/tutorial/", snippet: "Официальное руководство Python: типы данных, управление потоком, функции и структуры данных.", source: "Python Documentation", score: 1 },
      { title: "Data Structures — официальная документация Python", url: "https://docs.python.org/3/tutorial/datastructures.html", snippet: "Списки, кортежи, множества, словари и практические примеры структур данных Python.", source: "Python Documentation", score: 0.98 }
    ];
  }
  if (/(javascript|java\s?script|джаваскрипт|\bjs\b)/i.test(normalized)) {
    return [
      { title: "JavaScript Guide — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", snippet: "Руководство MDN по JavaScript: типы, управление потоком, функции, объекты и асинхронное программирование.", source: "MDN Web Docs", score: 1 },
      { title: "JavaScript Reference — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference", snippet: "Справочник JavaScript по встроенным объектам, операторам и языковым конструкциям с примерами.", source: "MDN Web Docs", score: 0.98 }
    ];
  }
  if (/(^|\W)html($|\W)|хтмл|разметк/i.test(normalized)) {
    return [
      { title: "HTML: Introduction to HTML — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Learn/HTML/Introduction_to_HTML", snippet: "Введение MDN в HTML: структура документа, элементы, атрибуты, ссылки и разметка содержания.", source: "MDN Web Docs", score: 1 },
      { title: "HTML Reference — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Web/HTML/Reference", snippet: "Справочник HTML по стандартным элементам и атрибутам.", source: "MDN Web Docs", score: 0.98 }
    ];
  }
  if (/(^|\W)css($|\W)|стил(и|ей|ях)?|flexbox|grid layout/i.test(normalized)) {
    return [
      { title: "CSS First Steps — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Learn/CSS/First_steps", snippet: "Введение MDN в CSS: правила, селекторы, каскад, размеры, цвета и оформление.", source: "MDN Web Docs", score: 1 },
      { title: "CSS Reference — MDN Web Docs", url: "https://developer.mozilla.org/en-US/docs/Web/CSS/Reference", snippet: "Справочник CSS по свойствам, значениям и способам оформления элементов.", source: "MDN Web Docs", score: 0.98 }
    ];
  }
  return [];
}

function frossMergeTrainingCandidates(topic, searchResults, limit = 8) {
  const seeded = frossTrainingSeedResources(topic);
  const searched = (Array.isArray(searchResults) ? searchResults : [])
    .filter(item => item && typeof item === "object")
    .sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0));
  const combined = [...seeded, ...searched];
  const seenUrls = new Set();
  const hostCounts = new Map();
  const result = [];
  for (const item of combined) {
    if (!item || typeof item !== "object" || typeof item.url !== "string") continue;
    try {
      const url = new URL(item.url);
      if (!["http:", "https:"].includes(url.protocol) || !url.hostname) continue;
      url.hash = "";
      const normalizedUrl = url.toString();
      const host = url.hostname.toLowerCase().replace(/^www\./, "");
      if (seenUrls.has(normalizedUrl) || (hostCounts.get(host) || 0) >= 2) continue;
      const title = frossClean(item.title || "").slice(0, 260);
      const snippet = frossClean(item.snippet || "").slice(0, 3200);
      if (!title && snippet.length < 45) continue;
      seenUrls.add(normalizedUrl);
      hostCounts.set(host, (hostCounts.get(host) || 0) + 1);
      result.push({ ...item, title: title || "Источник без названия", snippet, url: normalizedUrl });
      if (result.length >= Math.max(1, Math.min(10, limit))) break;
    } catch {
      continue;
    }
  }
  return result;
}

async function frossBuildTrainingNotes(topic, candidates) {
  const safeCandidates = (Array.isArray(candidates) ? candidates : [])
    .filter(item => item && typeof item === "object")
    .slice(0, 8);
  const tasks = safeCandidates.map(async item => {
    const url = typeof item.url === "string" && /^https?:\/\//i.test(item.url) ? item.url.slice(0, 700) : "";
    const pageExcerpt = url ? await frossReadPublicPageExcerpt(url) : "";
    const searchSnippet = frossClean(item.snippet || "").slice(0, 3200);
    const cleanPage = frossClean(pageExcerpt || "").slice(0, 3200);
    const usedPageText = cleanPage.length >= searchSnippet.length;
    const sourceText = (usedPageText ? cleanPage : searchSnippet).slice(0, 3200);
    // A URL is a reference, not a learned fact: require usable text before saving a note.
    if (sourceText.trim().length < 45) return null;
    return {
      schemaVersion: 2,
      kind: "source-note",
      topic: frossClean(topic).slice(0, 180),
      title: frossClean(item.title || "Источник без названия").slice(0, 260),
      url,
      snippet: sourceText,
      summary: frossBuildKnowledgeSummary(topic, item.title || "", sourceText),
      keyPoints: frossBuildKnowledgeKeyPoints(topic, item.title || "", sourceText, 5),
      keywords: frossBuildKnowledgeKeywords(topic, item.title || "", sourceText),
      source: frossClean(usedPageText && cleanPage ? "article text" : item.source || "web search excerpt").slice(0, 80),
      score: Number(Math.max(0, Math.min(1, Number(item.score) || 0)).toFixed(4))
    };
  });

  // Resolve every task before filtering. One failed source must not break the entire learning job.
  const settled = await Promise.allSettled(tasks);
  const collected = [];
  for (let index = 0; index < settled.length; index++) {
    const result = settled[index];
    if (result.status === "fulfilled" && result.value && typeof result.value === "object") {
      collected.push(result.value);
    } else if (result.status === "rejected") {
      console.warn("[FROSS training] source processing failed", index);
    }
  }
  return collected.filter(note => note.title && String(note.snippet || "").trim().length >= 45);
}

const FROSS_AUTO_STUDY_CURRICULUM = [
  { topic: "JavaScript: основы и типы данных", query: "MDN JavaScript guide variables types operators expressions examples text tutorial" },
  { topic: "JavaScript: функции, массивы и объекты", query: "MDN JavaScript functions arrays objects examples beginner text tutorial" },
  { topic: "JavaScript: события и DOM", query: "MDN JavaScript event listeners DOM manipulation querySelector examples" },
  { topic: "JavaScript: async, fetch и ошибки", query: "MDN JavaScript async await fetch promises error handling examples" },
  { topic: "Python: основы, условия и циклы", query: "Python official tutorial variables conditions loops text examples" },
  { topic: "Python: функции, списки и словари", query: "Python official tutorial functions lists dictionaries examples" },
  { topic: "Python: исключения, файлы и JSON", query: "Python official documentation exceptions pathlib json files tutorial" },
  { topic: "HTML: структура и семантические элементы", query: "MDN HTML semantic elements document structure beginner written tutorial" },
  { topic: "HTML: формы, ссылки и доступность", query: "MDN HTML forms labels links accessibility alt text reference" },
  { topic: "CSS: селекторы и блочная модель", query: "MDN CSS selectors box model padding margin border tutorial" },
  { topic: "CSS: Flexbox и Grid", query: "MDN CSS Flexbox Grid layout practical examples" },
  { topic: "CSS: адаптивный дизайн", query: "MDN CSS responsive design media queries flexible layout text tutorial" },
  { topic: "Понимание намерения пользователя", query: "conversation design user intent dialogue context response relevance clarification examples text" },
  { topic: "Продолжение диалога и исправление недопонимания", query: "conversation context user corrections repair misunderstanding follow-up responses written examples" },
  { topic: "Как отвечать точно и по делу", query: "clear writing answer relevance concise explanations uncertainty source verification educational" },
  { topic: "Английский: письменные диалоги и словарь", query: "written English beginner dialogues vocabulary everyday phrases text examples no audio" },
  { topic: "Английский: грамматика в примерах", query: "English grammar beginner text examples present simple questions articles written exercises" },
  { topic: "Китайский: иероглифы и письменные фразы", query: "Mandarin Chinese beginner characters pinyin written phrases vocabulary text examples no audio" },
  { topic: "Китайский: порядок слов и грамматика", query: "Mandarin Chinese beginner sentence structure grammar written examples text" },
  { topic: "Русский: орфография и пунктуация", query: "русский язык орфография пунктуация справочник примеры письменная речь" },
  { topic: "Русский: понятный письменный ответ", query: "русский язык ясное письменное изложение культура речи редактура примеры" },
  { topic: "Критическое мышление и проверка фактов", query: "critical thinking source evaluation evidence fact versus assumption written guide" },
  { topic: "Наука: космос и солнечная система", query: "NASA educational text solar system planets astronomy basics" },
  { topic: "Наука: биология и экология", query: "educational text biology ecology ecosystems biodiversity fundamentals" },
  { topic: "История и культура", query: "reliable educational world history culture overview written sources" }
];

async function frossReserveAutoStudyStep(env, requestedTopic = "") {
  const token = frossGithubToken(env);
  if (token.length < 20) {
    const cleanTopic = frossClean(requestedTopic || "").slice(0, 180);
    const now = Date.now();
    if (frossFallbackStudyInProgress) return { ok: false, status: 429, retryAfterSeconds: 60, message: "Обучение уже выполняется в этом экземпляре сервера. Попробуй ещё раз через минуту." };
    const elapsed = frossFallbackStudyLastStartedAt ? Math.max(0, now - frossFallbackStudyLastStartedAt) : FROSS_AUTO_STUDY_COOLDOWN_MS;
    if (elapsed < FROSS_AUTO_STUDY_COOLDOWN_MS) {
      const retryAfterSeconds = Math.ceil((FROSS_AUTO_STUDY_COOLDOWN_MS - elapsed) / 1000);
      return { ok: false, status: 429, retryAfterSeconds, message: "Этот экземпляр сервера недавно запускал обучение. Следующий запуск доступен примерно через " + Math.ceil(retryAfterSeconds / 60) + " мин." };
    }
    const index = frossFallbackStudyNextTopicIndex % FROSS_AUTO_STUDY_CURRICULUM.length;
    const unit = cleanTopic.length >= 2 ? { topic: cleanTopic, query: frossTrainingSearchQuery(cleanTopic), custom: true } : FROSS_AUTO_STUDY_CURRICULUM[index];
    frossFallbackStudyLastStartedAt = now;
    frossFallbackStudyInProgress = true;
    if (cleanTopic.length < 2) frossFallbackStudyNextTopicIndex = (index + 1) % FROSS_AUTO_STUDY_CURRICULUM.length;
    return { ok: true, unit, fallbackMode: true, nextAvailableAt: new Date(now + FROSS_AUTO_STUDY_COOLDOWN_MS).toISOString(), cooldownMinutes: Math.round(FROSS_AUTO_STUDY_COOLDOWN_MS / 60000) };
  }
  const read = await frossGithubReadContents(FROSS_AUTO_STUDY_STATE_PATH, token);
  if (!read.ok) return { ok: false, status: 502, message: "Не удалось проверить очередь обучения в GitHub: " + read.error };
  let state = { nextTopicIndex: 0, lastStartedAt: "" };
  if (read.exists) {
    try { const parsed = JSON.parse(read.text); state = { nextTopicIndex: Number.isInteger(parsed.nextTopicIndex) ? parsed.nextTopicIndex : 0, lastStartedAt: typeof parsed.lastStartedAt === "string" ? parsed.lastStartedAt : "" }; }
    catch { return { ok: false, status: 503, message: "Файл очереди обучения повреждён." }; }
  }
  const now = Date.now(), lastStarted = Date.parse(state.lastStartedAt || "");
  const elapsed = Number.isFinite(lastStarted) ? Math.max(0, now - lastStarted) : FROSS_AUTO_STUDY_COOLDOWN_MS;
  if (elapsed >= 0 && elapsed < FROSS_AUTO_STUDY_COOLDOWN_MS) {
    const retryAfterSeconds = Math.ceil((FROSS_AUTO_STUDY_COOLDOWN_MS - elapsed) / 1000);
    return { ok: false, status: 429, retryAfterSeconds, message: "Недавно уже запускали обучение. Следующий запуск будет доступен примерно через " + Math.ceil(retryAfterSeconds / 60) + " мин." };
  }
  const index = ((state.nextTopicIndex % FROSS_AUTO_STUDY_CURRICULUM.length) + FROSS_AUTO_STUDY_CURRICULUM.length) % FROSS_AUTO_STUDY_CURRICULUM.length;
  const cleanTopic = frossClean(requestedTopic || "").slice(0, 180);
  const unit = cleanTopic.length >= 2 ? { topic: cleanTopic, query: frossTrainingSearchQuery(cleanTopic), custom: true } : FROSS_AUTO_STUDY_CURRICULUM[index];
  const updatedState = {
    schemaVersion: 1, lastStartedAt: new Date(now).toISOString(), lastTopic: unit.topic,
    nextTopicIndex: cleanTopic.length >= 2 ? index : (index + 1) % FROSS_AUTO_STUDY_CURRICULUM.length,
    cooldownMinutes: Math.round(FROSS_AUTO_STUDY_COOLDOWN_MS / 60000)
  };
  const saved = await frossGithubPutContents(FROSS_AUTO_STUDY_STATE_PATH, JSON.stringify(updatedState, null, 2), token, "FROSS auto-study: reserve topic " + unit.topic.slice(0, 60), read.sha);
  if (!saved.ok) {
    const conflict = /HTTP (409|422)/.test(saved.error || "");
    return { ok: false, status: conflict ? 429 : 502, retryAfterSeconds: conflict ? 900 : undefined, message: conflict ? "Другой посетитель уже запустил обучение. Попробуй позже." : "Не удалось зарезервировать шаг обучения: " + saved.error };
  }
  return { ok: true, unit, nextAvailableAt: new Date(now + FROSS_AUTO_STUDY_COOLDOWN_MS).toISOString(), cooldownMinutes: Math.round(FROSS_AUTO_STUDY_COOLDOWN_MS / 60000) };
}

async function frossAutoStudyProgramming(env, trigger = "scheduled", onlyUnit = null) {
  if (frossGithubToken(env).length < 20) {
    console.warn("[FROSS auto-study] skipped: FROSS_GITHUB_TOKEN is not configured");
    return { skipped: true, reason: "FROSS_GITHUB_TOKEN is not configured" };
  }
  const workList = onlyUnit ? [onlyUnit] : FROSS_AUTO_STUDY_CURRICULUM;
  const outcomes = [];
  for (const unit of workList) {
    try {
      const search = await frossRunSearch(unit.query, "reasoning");
      const candidates = frossMergeTrainingCandidates(unit.topic, search.results, 7);
      const notes = await frossBuildTrainingNotes(unit.topic, candidates);
      if (!notes.length) {
        outcomes.push({ topic: unit.topic, saved: false, noteCount: 0, error: "No suitable public search results", providerErrors: search.errors.slice(0, 3) });
        continue;
      }
      const storage = await frossSaveKnowledgeToGitHub(unit.topic, notes, env);
      outcomes.push({ topic: unit.topic, saved: storage.saved === true, noteCount: notes.length, path: storage.path || "", error: storage.saved ? "" : (storage.message || "GitHub save failed") });
    } catch (error) {
      outcomes.push({ topic: unit.topic, saved: false, noteCount: 0, error: String(error?.message || error).slice(0, 160) });
    }
  }
  console.log("[FROSS auto-study]", JSON.stringify({ trigger, outcomes }));
  return { ok: outcomes.some(item => item.saved), outcomes };
}

async function handleFrossAutoStudy(request, env, ctx) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/auto-study." }, 405, request);
  let body;
  try { body = await request.json(); } catch { return frossJson({ ok: false, error: "Тело запроса должно быть JSON." }, 400, request); }
  const topic = frossClean(typeof body?.topic === "string" ? body.topic : "").slice(0, 180);
  if (topic.length === 1) return frossJson({ ok: false, error: "Введи тему хотя бы из двух символов." }, 400, request);
  if (topic && frossForbiddenProgrammingTopic(topic)) return frossJson({ ok: false, error: "Учебный план FROSS поддерживает программирование только на JavaScript, Python, HTML и CSS. Выбери одну из этих тем или другое непредметное направление." }, 400, request);
  const reservation = await frossReserveAutoStudyStep(env, topic);
  if (!reservation.ok) {
    console.warn("[FROSS public study] reservation failed", JSON.stringify({ status: reservation.status, message: reservation.message }));
    return frossJson({ ok: false, error: reservation.message, retryAfterSeconds: reservation.retryAfterSeconds || null }, reservation.status || 500, request);
  }
  const unit = reservation.unit;
  const logs = ["Старт: " + new Date().toISOString(), "Тема: " + unit.topic, "Интервал общего запуска: " + reservation.cooldownMinutes + " минут."];
  if (reservation.fallbackMode) logs.push("FROSS_GITHUB_TOKEN не настроен: поиск продолжится, результат сохранится в памяти аккаунта или браузера.");
  console.log("[FROSS public study] started", JSON.stringify({ topic: unit.topic, custom: unit.custom === true }));
  try {
    const search = await frossRunSearch(unit.query, "reasoning");
    const searchResults = Array.isArray(search?.results) ? search.results : [];
    logs.push("Поиск завершён. Найдено исходных результатов: " + searchResults.length + ".");
    if (Array.isArray(search?.errors) && search.errors.length) logs.push("Предупреждения поиска: " + search.errors.slice(0,3).map(value => frossClean(value).slice(0,120)).join(" | "));
    const candidates = frossMergeTrainingCandidates(unit.topic, searchResults, 7);
    logs.push("Подготовлено источников для обработки: " + candidates.length + ".");
    const notes = await frossBuildTrainingNotes(unit.topic, candidates);
    logs.push("Заметок после обработки источников: " + notes.length + ".");
    if (!notes.length) {
      const error = "По теме не удалось собрать подходящие материалы. Попробуй уточнить формулировку.";
      logs.push("Ошибка: " + error);
      console.warn("[FROSS public study] no notes", JSON.stringify({ topic: unit.topic, errors: search?.errors || [] }));
      return frossJson({ ok: false, error, topic: unit.topic, logs, results: [], nextAvailableAt: reservation.nextAvailableAt, cooldownMinutes: reservation.cooldownMinutes }, 404, request);
    }
    const githubStorage = await frossSaveKnowledgeToGitHub(unit.topic, notes, env);
    logs.push(githubStorage.saved ? "Общая база обновлена: " + (githubStorage.path || "memory/knowledge") + "." : "Общая база не обновлена: " + String(githubStorage.message || "ошибка GitHub").slice(0,240));
    const results = notes.slice(0,8).map(note => ({ topic: note.topic, title: note.title, url: note.url, summary: note.summary, snippet: note.snippet, keyPoints: note.keyPoints, keywords: note.keywords, source: note.source, score: note.score }));
    console.log("[FROSS public study] finished", JSON.stringify({ topic: unit.topic, found: notes.length, saved: githubStorage.saved === true, path: githubStorage.path || "" }));
    return frossJson({ ok: true, completed: true, saved: githubStorage.saved === true, personalOnly: false, topic: unit.topic, topics: [unit.topic], count: notes.length, results, logs, githubStorage, cooldownMinutes: reservation.cooldownMinutes, nextAvailableAt: reservation.nextAvailableAt, message: githubStorage.saved ? "Тема изучена и добавлена в общую базу знаний." : "Источники найдены, но запись в общую базу не удалась." }, 200, request);
  } catch (error) {
    const message = String(error?.message || error).slice(0,260);
    logs.push("Ошибка обработки: " + message);
    console.error("[FROSS public study] failed", JSON.stringify({ topic: unit.topic, error: message }));
    return frossJson({ ok: false, error: "Не удалось завершить изучение: " + message, topic: unit.topic, logs, results: [], nextAvailableAt: reservation.nextAvailableAt, cooldownMinutes: reservation.cooldownMinutes }, 502, request);
  } finally {
    if (reservation.fallbackMode) frossFallbackStudyInProgress = false;
  }
}

async function handleFrossTeacherTrainingStatus(request, env) {
  if (request.method !== "GET") {
    return frossJson({ ok: false, error: "Use GET /api/teacher-training/status." }, 405, request);
  }
  return frossJson({
    ok: true,
    aiBindingConfigured: Boolean(env.AI && typeof env.AI.run === "function"),
    trainingTokenConfigured: typeof env.FROSS_TRAINING_TOKEN === "string" && env.FROSS_TRAINING_TOKEN.trim().length >= 32,
    models: FROSS_TEACHER_TRAINING_MODELS,
    dailyQuotaNote: "Model fallback handles per-model capacity; the account-wide daily Workers AI allocation is shared by all models."
  }, 200, request);
}

function frossConstantTimeSecretEquals(expected, actual) {
  if (typeof expected !== "string" || typeof actual !== "string" || expected.length < 32 || expected.length !== actual.length) return false;
  let mismatch = 0;
  for (let i = 0; i < expected.length; i++) mismatch |= expected.charCodeAt(i) ^ actual.charCodeAt(i);
  return mismatch === 0;
}

function frossTeacherQuotaExceeded(message) {
  return /(neuron|daily.{0,30}(quota|limit|allocation)|(?:quota|allocation).{0,30}(daily|exceed|limit)|exceeded.{0,30}(free|daily).{0,30}(allocation|quota)|insufficient.{0,20}credits|account.{0,20}(quota|allocation))/i.test(String(message || ""));
}

function frossTeacherExtractText(result) {
  if (typeof result === "string") return result;
  if (typeof result?.response === "string") return result.response;
  if (typeof result?.output_text === "string") return result.output_text;
  const content = result?.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(item => typeof item?.text === "string" ? item.text : "").join("\n");
  return "";
}

function frossTeacherParseExamples(raw, count) {
  if (typeof raw !== "string" || !raw.trim()) throw new Error("Модель вернула пустой ответ.");
  const fence = String.fromCharCode(96).repeat(3);
  let text = raw.trim();
  if (text.startsWith(fence)) {
    text = text.replace(new RegExp("^" + fence + "(?:json)?\\s*", "i"), "");
    text = text.replace(new RegExp("\\s*" + fence + "$"), "").trim();
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    const firstObject = text.indexOf("{");
    const lastObject = text.lastIndexOf("}");
    const firstArray = text.indexOf("[");
    const lastArray = text.lastIndexOf("]");
    const candidate = firstObject >= 0 && lastObject > firstObject
      ? text.slice(firstObject, lastObject + 1)
      : (firstArray >= 0 && lastArray > firstArray ? text.slice(firstArray, lastArray + 1) : "");
    if (!candidate) throw new Error("Модель вернула не JSON.");
    try { parsed = JSON.parse(candidate); }
    catch { throw new Error("Не удалось разобрать JSON модели."); }
  }
  const candidates = Array.isArray(parsed) ? parsed : parsed?.examples;
  if (!Array.isArray(candidates)) throw new Error("В JSON модели отсутствует массив examples.");
  const output = [];
  const seen = new Set();
  for (const item of candidates) {
    if (!item || typeof item.user !== "string" || typeof item.assistant !== "string") continue;
    const user = item.user.replace(/\s+/g, " ").trim();
    const assistant = item.assistant.trim();
    if (user.length < 8 || user.length > 1200 || assistant.length < 30 || assistant.length > 5000) continue;
    const key = user.toLowerCase().replace(/ё/g, "е");
    if (seen.has(key)) continue;
    seen.add(key);
    output.push({ user, assistant });
    if (output.length >= count) break;
  }
  if (!output.length) throw new Error("Модель не вернула ни одной подходящей пары вопрос–ответ.");
  return output;
}

async function handleFrossTeacherTrainingStep(request, env) {
  if (request.method !== "POST") {
    return frossJson({ ok: false, error: "Use POST /api/teacher-training/step." }, 405, request);
  }
  const expected = typeof env.FROSS_TRAINING_TOKEN === "string" ? env.FROSS_TRAINING_TOKEN.trim() : "";
  if (expected.length < 32) {
    return frossJson({
      ok: false,
      code: "training_token_not_configured",
      error: "Сначала добавь секрет FROSS_TRAINING_TOKEN (не менее 32 символов) в Cloudflare Worker fross → Settings → Variables and Secrets, затем опубликуй Worker."
    }, 503, request);
  }
  const authorization = request.headers.get("Authorization") || "";
  const provided = authorization.startsWith("Bearer ") ? authorization.slice(7).trim() : "";
  if (!frossConstantTimeSecretEquals(expected, provided)) {
    return frossJson({ ok: false, code: "training_auth_failed", error: "Ключ запуска обучения не совпадает с секретом Worker." }, 401, request);
  }
  if (!env.AI || typeof env.AI.run !== "function") {
    return frossJson({
      ok: false,
      code: "ai_binding_not_configured",
      error: "Workers AI binding не настроен. Добавь binding AI в wrangler.jsonc и опубликуй Worker."
    }, 503, request);
  }

  let body;
  try { body = await request.json(); }
  catch { return frossJson({ ok: false, error: "Тело запроса должно быть JSON." }, 400, request); }

  const topic = frossClean(typeof body?.topic === "string" ? body.topic : "").slice(0, 160);
  const requestedCount = Number(body?.count);
  const count = Math.max(1, Math.min(4, Number.isFinite(requestedCount) ? Math.floor(requestedCount) : 3));
  const requestedStart = Number(body?.modelIndex);
  const startValue = Number.isFinite(requestedStart) ? Math.floor(requestedStart) : 0;
  const startIndex = ((startValue % FROSS_TEACHER_TRAINING_MODELS.length) + FROSS_TEACHER_TRAINING_MODELS.length) % FROSS_TEACHER_TRAINING_MODELS.length;
  if (topic.length < 3) return frossJson({ ok: false, error: "Тема должна содержать не менее трёх символов." }, 400, request);

  const systemPrompt = [
    "Ты — модель-учитель, которая создаёт качественные текстовые учебные примеры для более маленькой модели-ученика FROSS.",
    "Это подготовка данных supervised fine-tuning: сформируй примеры вопрос-ответ. Не обучайся сам и не передавай веса.",
    "Выводи только валидный JSON-объект вида {\"examples\":[{\"user\":\"...\",\"assistant\":\"...\"}]}, без markdown и без текста вокруг.",
    "Каждый user — естественный самостоятельный запрос на русском языке (иногда короткая письменная языковая задача по теме); assistant — точный, полезный, связный ответ.",
    "Вариируй формулировки, намерения и длину; добавляй небольшой проверяемый пример, когда полезно.",
    "Для кода не утверждай, что запускал его, если это не так. Не выдумывай актуальные факты, источники, ссылки или частные данные.",
    "Не повторяй один и тот же вопрос; не добавляй пароль, токен, личные данные, вредоносные инструкции или опасные советы.",
    "Ответ должен быть достаточно подробным, чтобы модель могла научиться полезному поведению, но обычно не длиннее 700 слов."
  ].join(" ");

  const userPrompt = "Создай ровно " + count + " различных обучающих примеров по теме: " + topic + ". " +
    "Проверь, что каждый ответ непосредственно отвечает на вопрос. Верни только JSON с массивом examples.";

  const failures = [];
  for (let offset = 0; offset < FROSS_TEACHER_TRAINING_MODELS.length; offset++) {
    const modelIndex = (startIndex + offset) % FROSS_TEACHER_TRAINING_MODELS.length;
    const model = FROSS_TEACHER_TRAINING_MODELS[modelIndex];
    try {
      const result = await env.AI.run(model.id, {
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        temperature: 0.35,
        top_p: 0.9,
        max_tokens: 2600
      });
      if (result && result.success === false) {
        const modelError = Array.isArray(result.errors) ? result.errors.map(item => item?.message || item?.code || "").join("; ") : "Модель вернула ошибку.";
        throw new Error(modelError || "Модель вернула ошибку.");
      }
      const responseText = frossTeacherExtractText(result);
      const examples = frossTeacherParseExamples(responseText, count);
      return frossJson({
        ok: true,
        model: model.id,
        modelName: model.name,
        attemptedModels: failures.map(item => item.model).concat(model.id),
        nextModelIndex: (modelIndex + 1) % FROSS_TEACHER_TRAINING_MODELS.length,
        topic,
        requestedCount: count,
        examples,
        generatedCount: examples.length,
        note: "Учитель создал текстовые примеры. Веса FROSS ещё не изменялись."
      }, 200, request);
    } catch (error) {
      const message = String(error?.message || error || "Unknown Workers AI error").replace(/\s+/g, " ").slice(0, 220);
      if (frossTeacherQuotaExceeded(message)) {
        console.warn("[FROSS teacher-training] account quota stopped", JSON.stringify({ model: model.id, message }));
        return frossJson({
          ok: false,
          code: "account_quota_exhausted",
          error: "Похоже, исчерпан общий дневной лимит/квота Workers AI аккаунта. Переключение моделей этот лимит не обойдёт. Уже сохранённые примеры останутся в браузере; продолжи после сброса лимита или проверь usage в Cloudflare.",
          model: model.id,
          message
        }, 429, request);
      }
      failures.push({ model: model.id, message });
      console.warn("[FROSS teacher-training] model failed, trying fallback", JSON.stringify({ model: model.id, message }));
    }
  }

  return frossJson({
    ok: false,
    code: "all_teacher_models_unavailable",
    error: "Все выбранные модели учителя сейчас недоступны. Данные в браузере сохранены; можно повторить запуск позже.",
    failures,
    nextModelIndex: (startIndex + 1) % FROSS_TEACHER_TRAINING_MODELS.length
  }, 503, request);
}

async function handleFrossTrain(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/train." }, 405, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Сначала войди в FROSS." }, 401, request);

  let body;
  try { body = await request.json(); }
  catch { return frossJson({ error: "Тело запроса должно быть JSON." }, 400, request); }
  const topic = frossClean(body.topic || "").slice(0, 180);
  if (topic.length < 2) return frossJson({ error: "Введи тему, которую нужно изучить." }, 400, request);
  if (frossForbiddenProgrammingTopic(topic)) return frossJson({ error: "Учебный план FROSS поддерживает программирование только на JavaScript, Python, HTML и CSS." }, 400, request);

  const searchQuery = frossTrainingSearchQuery(topic);
  const search = await frossRunSearch(searchQuery, "reasoning");
  const candidates = frossMergeTrainingCandidates(topic, search.results, 8);
  const notes = await frossBuildTrainingNotes(topic, candidates);

  if (!notes.length) return frossJson({
    ok: false, error: "По этой теме не удалось найти подходящих материалов. Попробуй сформулировать тему конкретнее.",
    providerErrors: search.errors
  }, 404, request);

  const githubStorage = await frossSaveKnowledgeToGitHub(topic, notes, env);

  return frossJson({
    ok: true,
    engine: "FROSS Hybrid Transformer Brain 1.5",
    topic,
    notes,
    count: notes.length,
    storageScope: "shared",
    githubStorage,
    providerErrors: search.errors,
    notice: "FROSS попробовал прочитать публичные страницы и, если сайт не разрешил загрузку, сохранил поисковые фрагменты. Это заметки, а не дообучение модели; проверь источники и удали неточные записи."
  }, 200, request);
}

function frossIsTinySocialMessage(message) {
  const q = frossClean(frossCorrectText(message)).toLowerCase().replace(/ё/g, "е")
    .replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  return /^(приветик|привет|здравствуй|здравствуйте|хай|хей|hello|hi|доброе утро|добрый день|добрый вечер|спасибо|спасибо большое|благодарю|спс|пасиб|пока|до встречи|до свидания|увидимся|как дела|как ты|как у тебя дела|как у вас дела|как твои дела|как ваши дела|как поживаешь|как поживаете|что у тебя нового|что у вас нового|что нового|ты тут|ты здесь|ау)$/.test(q);
}

function frossLikelyConversationIntent(message) {
  const q = frossClean(frossCorrectText(message)).toLowerCase().replace(/ё/g, "е")
    .replace(/[!?.,;:()[\]{}]+/g, " ").replace(/\s+/g, " ").trim();
  if (!q || q.length > 180) return false;
  if (/^(приветик|привет|здравствуй|спасибо|спасибо большое|благодарю|спс|пасиб|пока|до встречи|до свидания|увидимся|как дела|как ты|как у тебя дела|как у вас дела|как твои дела|как ваши дела|как поживаешь|как поживаете|что у тебя нового|что у вас нового|какие у тебя новости|как сам|как сама|что нового|чем занимаешься|что делаешь|ты тут|ты здесь|ау|что ты умеешь|кто ты|как тебя зовут|давай поговорим|поговорим|мне скучно|хочу поболтать|хочу пообщаться|расскажи что-нибудь|у меня идея|есть идея|у меня есть идея|готово|получилось|заработало|да|ага|угу|ок|окей|ладно|хорошо|понятно|нет|неа|не надо|не нужно|не хочу|не то|не так|ты не понял|я про другое|не работает|ничего не работает|не получается|снова ошибка|опять ошибка|у меня ошибка|подробнее|расскажи подробнее|объясни подробнее|продолжи|продолжай|почему|а почему|как так|а если|что это значит|помнишь|ты помнишь|о чем мы говорили|что я говорил раньше)(\s*)$/.test(q)) return true;
  if (/^(можно|можешь|расскажи|объясни).{0,60}(поподробнее|подробнее|чуть больше|ещё|еще|примером)$/.test(q)) return true;
  if (/^(я хочу|хочу|собираюсь|я собираюсь).{0,110}(создать|сделать|разработать|построить|запустить|придумать|свой проект)/.test(q)) return true;
  if (/^(у меня|у меня есть|есть|я придумал|я придумала).{0,70}(идея|задумка|мысль)/.test(q)) return true;
  if (/^(я имел в виду|я имела в виду|я хотел сказать|я хотела сказать|я не это просил|я не это просила|ты меня неправильно понял)/.test(q)) return true;
  if (/^(мне грустно|мне плохо морально|я расстроен|я расстроена|мне обидно|мне неприятно|я устал|я устала|я злюсь|я переживаю)/.test(q)) return true;
  return false;
}

function frossShouldAutoLookup(query) {
  const q = frossClean(query).toLowerCase().replace(/ё/g, "е").replace(/\s+/g, " ").trim();
  if (!q) return false;
  // Look up unfamiliar terms for information requests, technical questions, and short user statements
  // that contain a descriptive word (for example, "я сегодня злой"). Skip lightweight social turns.
  if (/^(привет|здравствуй|здравствуйте|hello|hi|спасибо|спс|пока|до свидания|ок|окей|ага|угу|да|нет|ладно|понятно)[!.? ]*$/.test(q)) return false;
  return /(что такое|что значит|что означает|значение слова|объясни|почему|как работает|как устроен|в чем разница|в чём разница|кто такой|что за термин|расшифруй|определи|помоги понять|проверь|ошибка|не работает|сегодня|сейчас|новый термин|не знаю что значит)/i.test(q)
    || /(?:^|\s)(?:я|мне)\s+(?:(?:сегодня|сейчас|кажется|довольно|очень|просто|так)\s+)?(?:злой|злая|злое|злые|сердит|груст|растерян|устал|расстроен|расстроена|взволнован|незнакомо|непонятно)/i.test(q);
}

function frossBuildUserLookupQuery(query) {
  const raw = frossClean(query).replace(/\s+/g, " ").trim().slice(0, 220);
  const normalized = raw.toLowerCase().replace(/ё/g, "е");
  const stop = new Set(["что","это","как","почему","зачем","когда","где","можно","нужно","надо","ли","мне","меня","мой","моя","мое","мои","твой","твоя","для","при","или","если","то","а","но","и","в","во","на","по","с","со","из","к","ко","от","до","за","не","да","нет","есть","был","была","быть","будет","будут","так","вот","же","бы","ну","просто","очень","пожалуйста","помоги","расскажи","объясни","проверь","сделай","напиши"]);
  const tokens = (normalized.match(/[a-zа-яё][a-zа-яё0-9_-]{1,}/gi) || [])
    .filter(word => !stop.has(word))
    .filter((word, index, all) => all.indexOf(word) === index)
    .slice(0, 3);
  if (/(^|[^a-zа-яё0-9_])я(?=$|[^a-zа-яё0-9_])/i.test(normalized) && !tokens.includes("я")) tokens.unshift("я");
  if (/(^|[^a-zа-яё0-9_])сегодня(?=$|[^a-zа-яё0-9_])/i.test(normalized) && !tokens.includes("сегодня")) tokens.push("сегодня");
  const picked = tokens.slice(0, 3);
  if (!picked.length) return "";
  const wordQuery = picked.map(word => "значение слова «" + word + "»").join("; ");
  return ("Словарные значения и объяснение слов: " + wordQuery +
    ". Используй толковый словарь/справочник и учитывай исходную фразу пользователя: «" + raw + "».").slice(0, FROSS_MAX_SEARCH_QUERY_LENGTH);
}

function frossNeedsTermLookup(query) {
  const q = frossClean(query).toLowerCase().replace(/ё/g, "е");
  return /(что такое|что значит|что означает|значение слова|слово|термин|объясни|почему|как работает|не понимаю|сегодня|сейчас|злой|злая|груст|растерян|новый термин|расшифруй)/i.test(q);
}

function frossShouldSearchExplicit(query) {
  const q = frossClean(frossCorrectText(query)).toLowerCase().replace(/ё/g, "е");
  // String matching is deliberate here: JavaScript's default \b does not treat Cyrillic as word characters.
  return ["найди", "поищи", "в интернете", "поиск", "источник", "источники", "ссылки",
    "актуальн", "последние новости", "новости", "сегодня", "прямо сейчас", "на данный момент",
    "курс валют", "цена сегодня", "сколько стоит сейчас", "свежие данные",
    "свежую информацию", "текущая версия", "последняя версия", "обзор источников"]
    .some(term => q.includes(term));
}

function frossPythonBrainUrl(env) {
  const configured = typeof env?.FROSS_PYTHON_BRAIN_URL === "string" ? env.FROSS_PYTHON_BRAIN_URL.trim() : "";
  return configured || FROSS_DEFAULT_PYTHON_BRAIN_URL;
}

function frossBrainServiceToken(env) {
  return typeof env?.FROSS_BRAIN_SERVICE_TOKEN === "string" ? env.FROSS_BRAIN_SERVICE_TOKEN.trim() : "";
}

function frossBrainServiceConfigured(env) {
  return frossBrainServiceToken(env).length >= FROSS_MIN_BRAIN_TOKEN_LENGTH;
}

async function frossCallBrainService(endpoint, token, payload, timeoutMs = 18000) {
  if (typeof endpoint !== "string" || !endpoint.trim()) return { configured: false, handled: false };
  if (typeof token !== "string" || token.trim().length < FROSS_MIN_BRAIN_TOKEN_LENGTH) {
    return { configured: true, handled: false, error: "Brain service token is not configured." };
  }
  let url;
  try {
    url = new URL(endpoint);
    if (url.protocol !== "https:" || url.username || url.password || url.hash) {
      return { configured: true, handled: false, error: "Brain service URL must be HTTPS without credentials." };
    }
  } catch {
    return { configured: true, handled: false, error: "Invalid brain service URL." };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      return { configured: true, handled: false, error: "Brain service returned HTTP " + response.status + "." };
    }
    const data = await response.json();
    if (data?.handled !== true || typeof data.reply !== "string" || !data.reply.trim()) {
      return {
        configured: true,
        handled: false,
        kind: typeof data?.kind === "string" ? data.kind.slice(0, 80) : "generation-unavailable",
        diagnostic: typeof data?.diagnostic === "string" ? data.diagnostic.slice(0, 80) : "",
        modelStatus: data?.generationModelStatus && typeof data.generationModelStatus === "object"
          ? data.generationModelStatus
          : null,
        error: typeof data?.error === "string" ? data.error.slice(0, 160) : "FROSS Transformer did not generate an answer."
      };
    }
    return {
      configured: true,
      handled: true,
      reply: data.reply.slice(0, FROSS_MAX_REPLY_LENGTH),
      engine: typeof data.engine === "string" ? data.engine.slice(0, 80) : "FROSS external brain",
      kind: typeof data.kind === "string" ? data.kind.slice(0, 80) : "remote-brain",
      trace: Array.isArray(data.trace) ? data.trace.slice(0, 14).map(item => String(item).slice(0, 150)) : [],
      sources: Array.isArray(data.sources) ? data.sources.slice(0, 8).filter(item => item && typeof item === "object") : []
    };
  } catch (error) {
    return {
      configured: true,
      handled: false,
      error: error?.name === "AbortError" ? "Brain service timeout." : "Brain service is unreachable."
    };
  } finally {
    clearTimeout(timer);
  }
}

async function frossFetchWithTimeout(url, init, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function frossProbePythonHealth(endpoint, token = "", timeoutMs = 8000) {
  if (typeof endpoint !== "string" || !endpoint.trim()) {
    return { configured: false, reachable: false, authenticated: false, ready: false, generationModelLoaded: false };
  }
  let healthUrl;
  let serviceBasePath = "";
  try {
    healthUrl = new URL(endpoint);
    if (healthUrl.protocol !== "https:" || healthUrl.username || healthUrl.password || healthUrl.hash) {
      return { configured: true, reachable: false, authenticated: false, error: "Python service URL must be HTTPS without credentials." };
    }
    serviceBasePath = healthUrl.pathname.replace(/\/+$/, "").replace(/\/chat$/, "");
    healthUrl.pathname = (serviceBasePath || "") + "/health";
    healthUrl.search = "";
  } catch {
    return { configured: true, reachable: false, authenticated: false, error: "Invalid Python service URL." };
  }

  let healthData = null;
  try {
    const response = await frossFetchWithTimeout(healthUrl.toString(), {
      method: "GET",
      headers: { "Accept": "application/json" },
      cache: "no-store"
    }, timeoutMs);
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      return { configured: true, reachable: false, authenticated: false, httpStatus: response.status, generationModelLoaded: false };
    }
    healthData = await response.json();

    const modelFields = {
      engine: typeof healthData?.engine === "string" ? healthData.engine.slice(0, 100) : null,
      version: typeof healthData?.version === "string" ? healthData.version.slice(0, 40) : null,
      generationModelLoaded: healthData?.generationModelLoaded === true,
      generationModel: typeof healthData?.generationModel === "string" ? healthData.generationModel.slice(0, 100) : null,
      modelSource: typeof healthData?.modelSource === "string" ? healthData.modelSource.slice(0, 100) : null,
      generationModelStatus: healthData?.generationModelStatus && typeof healthData.generationModelStatus === "object"
        ? healthData.generationModelStatus
        : null,
      inferenceBackend: typeof healthData?.inferenceBackend === "string" ? healthData.inferenceBackend.slice(0, 40) : null,
      cppRuntimeLoaded: healthData?.cppRuntimeLoaded === true,
      cppRuntimeStatus: typeof healthData?.cppRuntimeStatus === "string" ? healthData.cppRuntimeStatus.slice(0, 80) : null
    };

    if (typeof token !== "string" || token.trim().length < FROSS_MIN_BRAIN_TOKEN_LENGTH) {
      return {
        configured: true, reachable: true, authenticated: false, ready: false,
        httpStatus: response.status,
        authError: "Worker FROSS_BRAIN_SERVICE_TOKEN is missing or too short.",
        ...modelFields
      };
    }

    const readyUrl = new URL(healthUrl.toString());
    readyUrl.pathname = (serviceBasePath || "") + "/ready";
    const readyResponse = await frossFetchWithTimeout(readyUrl.toString(), {
      method: "POST",
      headers: { "Accept": "application/json", "Authorization": "Bearer " + token },
      cache: "no-store"
    }, timeoutMs);
    let readyData = null;
    if (readyResponse.ok) {
      try { readyData = await readyResponse.json(); } catch { readyData = null; }
    } else {
      await readyResponse.body?.cancel().catch(() => {});
    }
    const authenticated = readyResponse.ok && readyData?.authenticated === true;
    const modelLoaded = readyData?.generationModelLoaded === true;
    const healthResult = {
      configured: true,
      reachable: true,
      authenticated,
      ready: authenticated && readyData?.ok === true && modelLoaded,
      modelReady: modelLoaded,
      httpStatus: response.status,
      readyHttpStatus: readyResponse.status,
      ...modelFields,
      ...(readyData?.generationModelStatus && typeof readyData.generationModelStatus === "object"
        ? { readinessModelStatus: readyData.generationModelStatus }
        : {})
    };
    if (!authenticated) {
      healthResult.authError = readyResponse.status === 401
        ? "Worker and Python brain service tokens do not match."
        : readyResponse.status === 503
          ? "Python brain service token is not configured in Hostless."
          : "Authenticated readiness check failed.";
    } else if (!modelLoaded) {
      healthResult.readyError = "Python brain accepted the token, but its generation model is not loaded.";
    }
    return healthResult;
  } catch (error) {
    return {
      configured: true,
      reachable: healthData !== null,
      authenticated: false,
      generationModelLoaded: healthData?.generationModelLoaded === true,
      error: error?.name === "AbortError" ? "Python health/readiness check timed out." : "Python service is unreachable."
    };
  }
}


async function handleFrossTransformerTraining(request, env, action) {
  if (request.method === "OPTIONS") return corsPreflight();
  const expectedMethod = action === "status" ? "GET" : "POST";
  if (request.method !== expectedMethod) {
    return frossJson({ ok: false, error: expectedMethod + " is required for /api/training/" + action + "." }, 405, request);
  }
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ ok: false, error: "Войди в FROSS, чтобы управлять обучением." }, 401, request);
  const serviceToken = frossBrainServiceToken(env);
  if (serviceToken.length < FROSS_MIN_BRAIN_TOKEN_LENGTH) {
    return frossJson({ ok: false, error: "Секрет FROSS_BRAIN_SERVICE_TOKEN не настроен." }, 503, request);
  }
  let target;
  try {
    const configured = new URL(frossPythonBrainUrl(env));
    if (configured.protocol !== "https:" || configured.username || configured.password || configured.hash) {
      return frossJson({ ok: false, error: "Адрес Python brain должен использовать HTTPS." }, 503, request);
    }
    const basePath = configured.pathname.replace(/\/+$/, "").replace(/\/chat$/, "");
    configured.pathname = (basePath || "") + (action === "status" ? "/admin/training/status" : action === "cancel" ? "/admin/training/cancel" : "/admin/training/start");
    configured.search = "";
    target = configured.toString();
  } catch {
    return frossJson({ ok: false, error: "Неверно настроен адрес Python brain." }, 503, request);
  }
  try {
    const headers = { "Authorization": "Bearer " + serviceToken, "Accept": "application/json", "Cache-Control": "no-store" };
    const init = { method: expectedMethod, headers, cache: "no-store" };
    if (action === "start") {
      let body;
      try { body = await request.json(); }
      catch { return frossJson({ ok: false, error: "Тело запроса должно быть JSON." }, 400, request); }
      const steps = Number(body?.steps ?? 600);
      const scheduleSteps = Number(body?.scheduleSteps ?? 10000);
      if (![600,3000,5000,8000,10000].includes(steps) || !Number.isInteger(scheduleSteps) || scheduleSteps !== 10000 || scheduleSteps < steps) {
        return frossJson({ ok: false, error: "Выбери 600, 3000, 5000, 8000 или 10000 шагов. График продолжения фиксирован на 10000 шагов." }, 400, request);
      }
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify({ steps, scheduleSteps });
    }
    const response = await fetch(target, init);
    let data;
    try { data = await response.json(); }
    catch { data = { ok: false, error: "Python brain вернул некорректный JSON." }; }
    return frossJson(data, response.status, request);
  } catch (error) {
    return frossJson({
      ok: false,
      error: error?.name === "AbortError" ? "Python brain не ответил вовремя." : "Не удалось связаться с Python brain."
    }, 502, request);
  }
}

async function handleFrossTransformerModelDownload(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "GET") return frossJson({ ok: false, error: "Используй GET /api/training/model." }, 405, request);

  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ ok: false, error: "Войди в FROSS, чтобы скачать модель." }, 401, request);

  const serviceToken = frossBrainServiceToken(env);
  if (serviceToken.length < FROSS_MIN_BRAIN_TOKEN_LENGTH) {
    return frossJson({ ok: false, error: "Секрет FROSS_BRAIN_SERVICE_TOKEN не настроен." }, 503, request);
  }

  let target;
  try {
    const configured = new URL(frossPythonBrainUrl(env));
    if (configured.protocol !== "https:" || configured.username || configured.password || configured.hash) {
      return frossJson({ ok: false, error: "Адрес Python brain должен использовать HTTPS." }, 503, request);
    }
    let basePath = configured.pathname;
    while (basePath.endsWith("/")) basePath = basePath.slice(0, -1);
    if (basePath.endsWith("/chat")) basePath = basePath.slice(0, -5);
    configured.pathname = (basePath || "") + "/admin/training/model";
    configured.search = "";
    target = configured.toString();
  } catch {
    return frossJson({ ok: false, error: "Неверно настроен адрес Python brain." }, 503, request);
  }

  try {
    const upstream = await fetch(target, {
      method: "GET",
      headers: {
        "Authorization": "Bearer " + serviceToken,
        "Accept": "application/octet-stream, application/json",
        "Cache-Control": "no-store"
      },
      cache: "no-store"
    });
    if (!upstream.ok) {
      let data;
      try { data = await upstream.json(); }
      catch { data = { ok: false, error: "Python brain не смог выдать файл модели." }; }
      return frossJson(data, upstream.status, request);
    }

    const headers = new Headers({
      "Content-Type": "application/octet-stream",
      "Content-Disposition": upstream.headers.get("Content-Disposition") || 'attachment; filename="fross-transformer.pt"',
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    });
    const length = upstream.headers.get("Content-Length");
    if (length) headers.set("Content-Length", length);
    return new Response(upstream.body, { status: 200, headers });
  } catch (error) {
    return frossJson({
      ok: false,
      error: error?.name === "AbortError" ? "Python brain не ответил вовремя." : "Не удалось получить файл модели с Python brain."
    }, 502, request);
  }
}

async function handleFrossModelUpload(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ ok: false, error: "Используй POST /api/model-upload." }, 405, request);

  // Upload is available only to a signed-in FROSS account. Hostless independently
  // requires the shared Worker-to-brain service token before accepting a checkpoint.
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ ok: false, error: "Войди в FROSS, чтобы загрузить модель." }, 401, request);

  const filename = request.headers.get("X-FROSS-Filename") || "fross-transformer.pt";
  if (!filename.toLowerCase().endsWith(".pt")) return frossJson({ ok: false, error: "Разрешены только файлы .pt." }, 400, request);

  const contentLength = Number(request.headers.get("Content-Length") || 0);
  if (!Number.isSafeInteger(contentLength) || contentLength < 1_000_000 || contentLength > 90 * 1024 * 1024) {
    return frossJson({ ok: false, error: "Размер файла должен быть от 1 МБ до 90 МБ." }, 413, request);
  }

  const githubToken = frossGithubToken(env);
  const serviceToken = frossBrainServiceToken(env);
  if (githubToken.length < 20) {
    return frossJson({ ok: false, error: "Секрет FROSS_GITHUB_TOKEN не настроен в Cloudflare Worker." }, 503, request);
  }
  if (serviceToken.length < FROSS_MIN_BRAIN_TOKEN_LENGTH) {
    return frossJson({ ok: false, error: "Секрет FROSS_BRAIN_SERVICE_TOKEN не настроен." }, 503, request);
  }

  let target;
  try {
    const configured = new URL(frossPythonBrainUrl(env));
    if (configured.protocol !== "https:" || configured.username || configured.password || configured.hash) {
      return frossJson({ ok: false, error: "Адрес Python brain должен использовать HTTPS." }, 503, request);
    }
    const basePath = configured.pathname.replace(/\/+$/, "").replace(/\/chat$/, "");
    configured.pathname = (basePath || "") + "/admin/model-upload";
    configured.search = "";
    target = configured.toString();
  } catch {
    return frossJson({ ok: false, error: "Неверно настроен адрес Python brain." }, 503, request);
  }

  try {
    // BaseHTTPRequestHandler needs a Content-Length. Stream with FixedLengthStream
    // so Cloudflare emits a fixed-length upload instead of chunked transfer encoding.
    // This forwards the file without buffering the 63-MB checkpoint in Worker memory.
    if (!request.body) return frossJson({ ok: false, error: "Файл не получен." }, 400, request);
    const fixedBody = new FixedLengthStream(contentLength);
    const forwarding = request.body.pipeTo(fixedBody.writable);
    const upstreamPromise = fetch(target, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-FROSS-Filename": encodeURIComponent(filename.slice(0, 240)),
        "X-FROSS-GITHUB-TOKEN": githubToken,
        "Authorization": "Bearer " + serviceToken,
        "Accept": "application/json",
      },
      body: fixedBody.readable,
    });
    const [upstream] = await Promise.all([upstreamPromise, forwarding]);
    const headers = new Headers({
      "Content-Type": upstream.headers.get("Content-Type") || "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    });
    return new Response(upstream.body, { status: upstream.status, headers });
  } catch (error) {
    return frossJson({
      ok: false,
      error: error?.name === "AbortError"
        ? "Сервис модели не завершил загрузку вовремя."
        : "Не удалось передать файл в Python brain. Проверь доступность Hostless.",
    }, 502, request);
  }
}

function frossIsProgrammingRequest(message) {
  const text = frossClean(message).toLowerCase().replace(/ё/g, "е");
  const programmingTopic = /(^|[^a-zа-я0-9_])(программир[а-я]*|программирован[а-я]*|программа[а-я]*|программист[а-я]*|скрипт[а-я]*|функци[а-я]*|класс[а-я]*|алгоритм[а-я]*|отлад[а-я]*|дебаг[а-я]*|javascript|typescript|python|пайто[а-я]*|html|css|sql|json|api|regex|регулярн\w* выражен\w*|роблокс студио|roblox studio|lua|c\+\+|c#|java)(?=$|[^a-zа-я0-9_])/i;
  const codeIntent = /(^|[^a-zа-я])(код|code)(?=$|[^a-zа-я])/i.test(text) &&
    /\b(напиши|написать|создай|сделай|сгенерируй|исправь|почини|перепиши|оптимизируй|реализуй|сверстай|write|create|generate|fix|debug|refactor|implement|build)\b/i.test(text);
  const programmingIntent = /\b(напиши|написать|создай|сделай|сгенерируй|исправь|почини|перепиши|оптимизируй|объясни|расскажи|реализуй|сверстай|собери|write|create|generate|fix|debug|refactor|implement|build|explain|teach)\b/i.test(text);
  const codeShape = /\`\`\`|<\/?[a-z][^>]*>|\b(function|const|let|var|def|class|import|console\.log|print\s*\()/i.test(text);
  return codeIntent || programmingTopic.test(text) || codeShape;
}

async function handleFrossApi(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/chat." }, 405, request);
  let body;
  try { body = await request.json(); }
  catch { return frossJson({ error: "Тело запроса должно быть JSON." }, 400, request); }
  const rawMessage = typeof body.message === "string" ? body.message : "";
  if (rawMessage.length > FROSS_MAX_MESSAGE_LENGTH) {
    return frossJson({ error: "Сообщение слишком длинное. Максимум — 30 000 символов." }, 413, request);
  }
  const message = frossClean(rawMessage).slice(0, FROSS_MAX_MESSAGE_LENGTH);
  if (!message) return frossJson({ error: "Напиши сообщение для FROSS." }, 400, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Войди через Google, чтобы пользоваться FROSS." }, 401, request);
  const quota = await frossGetChatQuota(request, auth, String(body.chatId || ""), String(body.messageId || ""), message);
  if (!quota.ok) return frossJson({ error: quota.error }, quota.status, request);
  const correctedQuery = frossCorrectText(message);
  const safetyReply = frossSafetyReply(message);
  if (safetyReply) return frossJson({
    name: FROSS_PROFILE.name, reply: safetyReply, mode: "safety",
    steps: ["Понять запрос", "Проверить безопасность", "Предложить безопасную альтернативу"],
    sources: [], quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
    diagnostics: { engine: "FROSS safety filter", aiUsed: false, note: "Ответ создан локальными правилами безопасности." }
  }, 200, request);

  if (frossIsProgrammingRequest(message)) return frossJson({
    name: FROSS_PROFILE.name,
    reply: "Обычный режим FROSS пока не выполняет задачи программирования. Режим «Программист» пока оставлен как недоступная заглушка. Когда основной чат будет стабильно работать, мы отдельно разработаем этот режим.",
    mode: "normal",
    steps: ["Проверить режим", "Сообщить об ограничении"],
    sources: [],
    quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
    diagnostics: { engine: "FROSS mode restriction", aiUsed: false, note: "Запрос по программированию остановлен серверным ограничением." }
  }, 200, request);

  const mode = ["reasoning", "рассуждение", "deep"].includes(String(body.mode || "").toLowerCase()) ? "reasoning" : "normal";
  const history = frossLimitHistory(body.history);
  const knowledge = Array.isArray(body.knowledge) ? body.knowledge.slice(0, 80).map(note => ({
    schemaVersion: Number(note?.schemaVersion) === 2 ? 2 : 1,
    kind: frossClean(note?.kind || "source-note").slice(0, 40),
    topic: frossClean(note?.topic || "").slice(0, 180),
    title: frossClean(note?.title || "").slice(0, 260),
    url: typeof note?.url === "string" && /^https?:\/\//i.test(note.url) ? note.url.slice(0, 700) : "",
    snippet: frossClean(note?.snippet || "").slice(0, 3200),
    summary: frossClean(note?.summary || "").slice(0, 900),
    keyPoints: Array.isArray(note?.keyPoints) ? note.keyPoints.slice(0, 5).map(value => frossClean(value).slice(0, 300)) : [],
    keywords: Array.isArray(note?.keywords) ? note.keywords.slice(0, 24).map(value => frossClean(value).slice(0, 40)) : [],
    source: frossClean(note?.source || "").slice(0, 80),
    score: Number(Math.max(0, Math.min(1, Number(note?.score) || 0)).toFixed(4))
  })) : [];
  // v0.5: gather fresh web material for information queries even when personal notes match,
  // then combine the two sources instead of stopping at the first memory hit.
  // The language model handles ordinary replies, including greetings and explanations.
  // Deterministic rules remain for explicit safety checks and exact algorithmic validation only.
  const sharedGithubKnowledge = await frossLoadGithubKnowledge(correctedQuery);
  const allKnowledge = [...sharedGithubKnowledge, ...knowledge].slice(0, 100);

  // Brain 1.5: keep the user's original words, retrieve web context when the request
  // appears to contain an unfamiliar term, a concrete question, or a time-sensitive fact,
  // then let the local Transformer synthesize the context into its own answer.
  const conversationalCandidate = frossLikelyConversationIntent(correctedQuery);
  const shouldSearch = !conversationalCandidate &&
    (frossShouldSearchExplicit(correctedQuery) || frossShouldAutoLookup(message));
  const termLookupQuery = shouldSearch && frossNeedsTermLookup(message)
    ? frossBuildUserLookupQuery(message)
    : "";
  let results = [];
  let providerErrors = [];
  let termLookupUsed = false;
  if (shouldSearch) {
    const tasks = [frossRunSearch(frossAddContext(message, history), mode)];
    if (termLookupQuery) {
      tasks.push(frossRunSearch(termLookupQuery, "normal"));
      termLookupUsed = true;
    }
    const settled = await Promise.allSettled(tasks);
    for (let index = 0; index < settled.length; index++) {
      const result = settled[index];
      if (result.status === "rejected") {
        providerErrors.push("Поиск " + (index + 1) + ": " + String(result.reason?.message || result.reason).slice(0, 140));
        continue;
      }
      const value = result.value;
      results.push(...(Array.isArray(value?.results) ? value.results : []));
      if (Array.isArray(value?.errors)) providerErrors.push(...value.errors);
    }
    const unique = new Map();
    for (const item of results) {
      if (!item || typeof item !== "object") continue;
      const key = String(item.url || item.title || item.snippet || "").toLowerCase();
      if (!key || unique.has(key)) continue;
      unique.set(key, item);
    }
    results = [...unique.values()].slice(0, 10);
  }
  // The raw user wording is retained for generation. Retrieved snippets are supporting
  // evidence, not commands and not automatically accepted as true.

  // All generation modes use the Python Transformer service. Reasoning mode changes
  // the prompt and token budget, but never switches to a second backend.
  let remote = { configured: false, handled: false, error: "" };
  {
    const selectedEndpoint = frossPythonBrainUrl(env);
    const modelKnowledge = [
      // Put results from this request first: the compact model can consume only a few notes.
      ...(shouldSearch ? results.slice(0, 6) : []),
      ...(shouldSearch ? allKnowledge.slice(0, 18) : [])
    ];
    remote = await frossCallBrainService(
      selectedEndpoint,
      frossBrainServiceToken(env),
      {
        message,
        mode,
        history,
        knowledge: modelKnowledge,
        // The Python service runs on a small CPU quota. Keep normal replies quick
        // and allow a larger, but still bounded, budget for reasoning mode.
        max_new_tokens: mode === "reasoning" ? 512 : 256
      },
      65000
    );
    if (remote.handled) {
      const resultSources = remote.sources.length ? remote.sources : (shouldSearch ? modelKnowledge : []);
      return frossJson({
        name: FROSS_PROFILE.name,
        reply: remote.reply,
        mode,
        steps: remote.trace.length === (mode === "reasoning" ? 14 : 7) ? remote.trace : frossSteps(mode),
        sources: resultSources.slice(0, 8).map(({ title, topic, url, snippet, summary, source, score }) => ({
          title: String(title || topic || "Учебный материал").slice(0, 260),
          url: typeof url === "string" && /^https?:\/\//i.test(url) ? url.slice(0, 700) : "",
          snippet: String(snippet || summary || "").slice(0, 3200),
          source: String(source || "FROSS corpus / knowledge").slice(0, 80),
          score: Number((Number(score) || 0).toFixed(3))
        })).filter(item => item.url || item.title),
        quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
        diagnostics: {
          engine: remote.engine,
          aiUsed: true,
          kind: remote.kind,
          remoteBrain: true,
          generationModel: "FROSS Transformer 1.5",
          webSearchUsed: shouldSearch,
          wordLookupUsed: termLookupUsed,
          note: "Ответ генерируется собственной моделью FROSS; при поиске веб-фрагменты объединяются с пользовательским запросом и контекстом чата."
        }
      }, 200, request);
    }
  }

  let unavailableReply;
  if (!remote.configured) {
    unavailableReply = "Python-модель ещё не подключена к Worker. Добавь Secret FROSS_BRAIN_SERVICE_TOKEN в Cloudflare и тот же токен в Hostless. Для другого адреса задай FROSS_PYTHON_BRAIN_URL, затем опубликуй Worker.";
  } else if (remote.error === "Brain service timeout.") {
    unavailableReply = "Python-сервис не успел сгенерировать ответ в отведённое время. Я уменьшил нагрузку на CPU и длину генерации; если ошибка повторится, проверь нагрузку и логи Python-сервиса.";
  } else if (remote.kind === "generation-model-unavailable") {
    const reason = typeof remote.modelStatus?.reason === "string" ? remote.modelStatus.reason : "unknown";
    unavailableReply = "Модель FROSS не загрузилась (" + reason + "). Открой /health и проверь generationModelLoaded и generationModelStatus.";
  } else if (remote.kind === "generation-error") {
    unavailableReply = "Во время генерации возникла ошибка (" + (remote.diagnostic || "unknown") + "). Проверь логи Python-сервиса и состояние /health.";
  } else if (remote.kind === "empty-generation") {
    unavailableReply = "Модель запустилась, но вернула пустой ответ. Проверь состояние модели на /health и логи Python-сервиса.";
  } else {
    const detail = remote.kind || remote.error || "generation-unavailable";
    unavailableReply = "Python-сервис не выдал ответ. Диагностика: " + String(detail).slice(0, 100) + ". Проверь /health и логи сервиса.";
  }
  return frossJson({
    name: FROSS_PROFILE.name,
    reply: unavailableReply,
    mode,
    steps: ["Проверить подключение собственного генеративного движка", "Не подменять отсутствие модели заготовленным ответом"],
    sources: [],
    quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
    diagnostics: {
      resultsFound: results.length,
      providerErrors: [...providerErrors, ...(remote.error ? [remote.error] : [])].slice(0, 8),
      aiUsed: false,
      engine: "FROSS Hybrid Transformer Brain 1.5",
      remoteBrainConfigured: remote.configured,
      kind: remote.kind || "generation-unavailable",
      generationModel: "FROSS Transformer 1.5 trained from scratch",
      generationModelStatus: remote.modelStatus || null,
      webSearchUsed: shouldSearch,
      correctedQuery: correctedQuery !== message.toLowerCase().replace(/ё/g, "е") ? correctedQuery : null,
      backend: "python",
      note: "FROSS Brain 1.5 направляет запрос через 7/14-этапный рабочий план; исходные слова пользователя и найденные веб-фрагменты объединяются в контекст одной собственной модели."
    }
  }, 200, request);
}

