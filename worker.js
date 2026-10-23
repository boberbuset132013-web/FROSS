// ================= КОНФИГУРАЦИЯ =================
const BOT_UPDATE_FORWARD_URL = "";
const ALLOWED_TOKENS = "";

// Таймауты Telegram-прокси (мс)
const LONG_POLL_TIMEOUT = 25000;
const SHORT_TIMEOUT = 8000;
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY = 400;
const MAX_RETRY_AFTER_MS = 5000;

// FROSS: ограничиваем время каждого поискового запроса.
const FROSS_FETCH_TIMEOUT_MS = 4500;
const FROSS_MAX_QUERY_LENGTH = 400;
const FROSS_MAX_RESULTS_PER_PROVIDER = 5;

// Заголовки, которые нельзя прокидывать дальше
const STRIP_HEADERS = new Set([
  "host", "content-length", "connection", "keep-alive",
  "proxy-authenticate", "proxy-authorization", "te", "trailer",
  "transfer-encoding", "upgrade",
  "cf-connecting-ip", "cf-ipcountry", "cf-ray", "cf-visitor",
  "x-forwarded-for", "x-forwarded-proto", "x-forwarded-host", "x-real-ip",
]);

// ================= ОСНОВНОЙ ОБРАБОТЧИК =================
export default {
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

    // 3. FROSS API в том же Worker — до фильтра Telegram-путей
    if (path === "/api/chat") {
      return handleFrossApi(request);
    }

    // 4. Вебхуки должны проверяться до фильтра Telegram Bot API путей.
    // Иначе /webhook/... никогда не дойдёт до handleWebhook.
    if (path.startsWith("/webhook")) return handleWebhook(request, path, ctx);

    // 5. Все остальные пути — существующая логика Telegram-прокси
    if (!isAllowedPath(path)) return jsonError(404, "Not found");

    // 6. Проверка токена (если ALLOWED_TOKENS задан)
    if (ALLOWED_TOKENS) {
      const token = extractToken(path);
      if (token) {
        const allowed = ALLOWED_TOKENS.split(",").map(t => t.trim()).filter(Boolean);
        if (!allowed.includes(token)) return jsonError(401, "Unauthorized token");
      }
    }

    // 7. Проксирование в Telegram
    return proxyToTelegram(request, url);
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
  description: "FROSS — самостоятельный поисковый помощник, который старается понять вопрос, проверять сведения и честно отмечать неопределённость.",
  traits: ["спокойный", "любознательный", "дружелюбный", "прямой", "честный"],
  style: "Отвечай понятно и естественно. Не выдумывай факты, источники или воспоминания. Отделяй найденное от предположений.",
  rules: [
    "Если данных недостаточно, прямо скажи об этом.",
    "При поиске показывай ссылки на источники.",
    "Если источники противоречат друг другу, не скрывай это.",
    "Не заявляй, что у тебя есть человеческое сознание или чувства.",
    "История доступна только тогда, когда сайт передал её в запросе."
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

function frossSmallTalk(query) {
  const text = query.trim().toLowerCase();

  if (/^(привет|здравствуй|здравствуйте|хай|hello|hi|доброе утро|добрый вечер)[!. ]*$/.test(text)) {
    return "Привет! Я FROSS. Могу помочь разобраться в вопросе или поискать сведения в интернете.";
  }
  if (/^(кто ты|ты кто|как тебя зовут)[?!. ]*$/.test(text)) {
    return "Я FROSS — поисковый помощник. Анализирую вопрос, ищу материалы в нескольких источниках и стараюсь отделять факты от предположений.";
  }
  if (/^(как дела|что нового)[?!. ]*$/.test(text)) {
    return "Работаю штатно и готов помочь. Что хочешь узнать?";
  }
  if (/^(спасибо|благодарю)[!. ]*$/.test(text)) {
    return "Пожалуйста! Если хочешь, можем разобраться дальше.";
  }
  if (/^(пока|до свидания)[!. ]*$/.test(text)) {
    return "До встречи! Я FROSS — возвращайся, когда понадобится помощь.";
  }
  return null;
}

function frossAddContext(query, history) {
  const refersBack =
    /\b(он|она|это|этот|эта|там|тогда|его|её|они|такой|такая|тот|предыдущий|предыдущая)\b/i.test(query) ||
    /^(а если|а что насчёт|а как насчёт)\b/i.test(query);

  if (!refersBack || !history.length) return query;

  const recent = history.slice(-6).map(item => {
    const role = item?.role === "assistant" ? "FROSS" : "Пользователь";
    const content = frossClean(item?.content || "").slice(0, 250);
    return content ? `${role}: ${content}` : "";
  }).filter(Boolean);

  return recent.length ? `${query}\nКонтекст разговора:\n${recent.join("\n")}` : query;
}

function frossBuildReply(query, results, mode) {
  const smallTalk = frossSmallTalk(query);
  if (smallTalk) return smallTalk;

  if (!results.length) {
    return "Я не смог получить результаты поиска. Возможно, источники временно недоступны или блокируют автоматические запросы. Попробуй переформулировать вопрос или повторить попытку позже.";
  }

  const top = results.slice(0, 4);
  const lines = ["Я нашёл несколько материалов, которые могут помочь. Вот краткая сводка поисковых фрагментов:"];

  top.forEach((item, index) => {
    lines.push(`${index + 1}. **${item.title}**\n${item.snippet || "Поисковик не предоставил описание страницы."}`);
  });

  lines.push("");
  lines.push("**Предварительный вывод:** это сводка найденных фрагментов, а не гарантия полной проверки фактов.");

  if (mode === "reasoning") {
    lines.push("**Дополнительная проверка:** результаты сопоставлены по ключевым словам, а дубликаты удалены. Это алгоритмическая оценка, а не человеческое рассуждение.");
  }

  lines.push("", "**Источники:**", ...top.map(item => `- [${item.title}](${item.url})`));
  return lines.join("\n");
}

function frossSteps(mode) {
  if (mode === "reasoning") {
    return [
      "Понять вопрос и цель пользователя",
      "Выделить ключевые слова и контекст",
      "Искать в нескольких источниках",
      "Сопоставить результаты и удалить дубликаты",
      "При слабых результатах выполнить уточняющий поиск",
      "Сформировать ответ со ссылками и оговорками"
    ];
  }

  return [
    "Понять вопрос и доступный контекст",
    "Найти и отсортировать подходящие сведения",
    "Сформировать ответ со ссылками"
  ];
}

async function frossRunSearch(query, mode) {
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

  return { results: ranked, errors };
}

function frossJson(data, status = 200, request) {
  const origin = request?.headers?.get("Origin") || "*";
  return new Response(JSON.stringify(data), {
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

async function handleFrossApi(request) {
  if (request.method === "OPTIONS") return corsPreflight();

  if (request.method !== "POST") {
    return frossJson({ error: "Используй POST /api/chat." }, 405, request);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return frossJson({ error: "Тело запроса должно быть JSON." }, 400, request);
  }

  const message = frossClean(body.message || "").slice(0, FROSS_MAX_QUERY_LENGTH);
  if (!message) return frossJson({ error: "Напиши сообщение для FROSS." }, 400, request);

  const mode = ["reasoning", "рассуждение", "deep"].includes(
    String(body.mode || "").toLowerCase()
  ) ? "reasoning" : "normal";

  const history = Array.isArray(body.history) ? body.history.slice(-10) : [];
  let results = [];
  let providerErrors = [];
  const directReply = frossSmallTalk(message);

  if (!directReply) {
    const search = await frossRunSearch(frossAddContext(message, history), mode);
    results = search.results;
    providerErrors = search.errors;
  }

  return frossJson({
    name: FROSS_PROFILE.name,
    reply: directReply || frossBuildReply(message, results, mode),
    mode,
    steps: frossSteps(mode),
    sources: results.slice(0, 8).map(({ title, url, snippet, source, score }) => ({
      title, url, snippet, source, score: Number(score.toFixed(3))
    })),
    profile: {
      description: FROSS_PROFILE.description,
      traits: FROSS_PROFILE.traits,
      style: FROSS_PROFILE.style,
      rules: FROSS_PROFILE.rules
    },
    diagnostics: {
      resultsFound: results.length,
      providerErrors,
      note: "FROSS использует эвристики и поисковые фрагменты, а не языковую модель."
    }
  }, 200, request);
}

// ================= ПРОКСИ TELEGRAM (СОХРАНЕНО) =================

async function proxyToTelegram(request, url) {
  const targetUrl = new URL("https://api.telegram.org");
  targetUrl.pathname = url.pathname;
  targetUrl.search = url.search;

  const headers = new Headers();
  for (const [key, value] of request.headers) {
    if (!STRIP_HEADERS.has(key.toLowerCase())) headers.set(key, value);
  }

  let bodyBuf = null;
  if (request.method !== "GET" && request.method !== "HEAD") {
    try {
      bodyBuf = await request.arrayBuffer();
    } catch {
      return jsonError(400, "Cannot read request body");
    }
  }

  const isLongPoll = url.pathname.includes("/getUpdates");
  const maxAttempts = isLongPoll ? 1 : (MAX_RETRIES + 1);
  const perAttemptTimeout = isLongPoll ? LONG_POLL_TIMEOUT : SHORT_TIMEOUT;
  let lastError = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), perAttemptTimeout);

    try {
      const options = {
        method: request.method,
        headers,
        redirect: "manual",
        signal: controller.signal,
      };
      if (bodyBuf !== null) options.body = bodyBuf;

      const response = await fetch(targetUrl.toString(), options);

      if (response.status === 429) {
        if (attempt < maxAttempts - 1) {
          const wait = Math.min(parseRetryAfterMs(response), MAX_RETRY_AFTER_MS);
          await sleep(wait);
          continue;
        }
        return response;
      }

      if (response.ok || (response.status >= 400 && response.status < 500)) {
        return response;
      }

      lastError = new Error(`Telegram API error: ${response.status}`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (lastError.name === "AbortError") break;
    } finally {
      clearTimeout(timeoutId);
    }

    if (attempt < maxAttempts - 1) {
      await sleep(RETRY_BASE_DELAY * Math.pow(2, attempt));
    }
  }

  return jsonError(502, lastError ? lastError.message : "Proxy error");
}

function parseRetryAfterMs(response) {
  const raw = response.headers.get("Retry-After");
  if (!raw) return RETRY_BASE_DELAY * 4;
  const seconds = parseInt(raw, 10);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  return RETRY_BASE_DELAY * 4;
}

// ================= ВЕБХУКИ TELEGRAM (СОХРАНЕНО) =================

async function handleWebhook(request, path, ctx) {
  const match = path.match(/^\/webhook\/bot([^/]+)/);
  if (!match) return jsonError(400, "Invalid webhook path");

  if (!BOT_UPDATE_FORWARD_URL) {
    return jsonError(500, "Forward URL not configured");
  }

  try {
    const body = await request.text();
    const secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token") || "";

    const response = await fetch(BOT_UPDATE_FORWARD_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Telegram-Bot-Api-Secret-Token": secret,
      },
      body,
    });

    return new Response(response.body, {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    if (ctx && ctx.waitUntil) {
      ctx.waitUntil(Promise.resolve(console.error("Webhook forward error:", error)));
    }
    return jsonError(502, "Webhook forward failed");
  }
}