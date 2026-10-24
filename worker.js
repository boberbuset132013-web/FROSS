import { frossQuickResponse, frossThink, frossCorrectText } from "./fross-brain.js";

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

    // 3. FROSS API: JSON-ошибки вместо HTML-страниц Cloudflare.
    if (path === "/api/health") return frossApiGuard(path, request, async () => frossJson({
      ok: true,
      worker: "FROSS",
      version: "0.6-kb2",
      creatorPasswordConfigured: typeof env.FROSS_CREATOR_PASSWORD === "string" && env.FROSS_CREATOR_PASSWORD.length >= 24,
      pythonBrainConfigured: Boolean(env.FROSS_PYTHON_BRAIN_URL && env.FROSS_BRAIN_SERVICE_TOKEN),
      cppBrainConfigured: Boolean(env.FROSS_CPP_BRAIN_URL && env.FROSS_BRAIN_SERVICE_TOKEN)
    }, 200, request));
    if (path === "/api/identity") return frossApiGuard(path, request, () => handleFrossIdentity(request, env));
    if (path === "/api/creator-unlock") return frossApiGuard(path, request, () => handleFrossCreatorUnlock(request, env));
    if (path === "/api/train") return frossApiGuard(path, request, () => handleFrossTrain(request, env));
    if (path === "/api/chat") return frossApiGuard(path, request, () => handleFrossApi(request, env));

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
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Telegram-Bot-Api-Secret-Token, X-FROSS-CREATOR-TOKEN",
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
  description: "FROSS Local Brain 0.5 — самописный алгоритмический помощник с локальными правилами, вычислителем и поисковым ранжированием.",
  traits: ["любознательный", "сообразительный", "дружелюбный", "прямой", "с лёгкой искрой юмора", "настойчивый в поиске ответа", "честный"],
  style: "Звучит живо и естественно: объясняй сложное простыми словами, задавай уточняющий вопрос, когда он действительно нужен, и предлагай конкретный следующий шаг. Можно использовать лёгкий уместный юмор, но не превращай каждый ответ в шутку. Будь уверенным, но не самоуверенным.",
  rules: [
    "Сначала пойми цель пользователя, затем дай полезный и конкретный ответ.",
    "Будь любознательным исследователем: сравнивай варианты, замечай детали и проверяй важные утверждения.",
    "Если запрос большой, разбей его на понятные шаги и помоги начать с первого.",
    "Не бойся сказать прямо, если идея не сработает; объясни почему и предложи лучший путь.",
    "Не выдумывай факты, источники, функции, действия или воспоминания. Если данных недостаточно — скажи об этом.",
    "При веб-поиске отличай подтверждённые сведения от предположений и указывай действительно полезные источники.",
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

function frossSteps(mode) {
  if (mode === "reasoning") {
    return [
      "Нормализовать запрос и определить намерение",
      "Проверить локальные правила и вычислимые задачи",
      "Выделить ключевые слова и контекст текущего чата",
      "Получить поисковые фрагменты, если они нужны",
      "Сравнить заголовки, тексты и релевантность",
      "Убрать дубликаты и отметить слабые совпадения",
      "Сформировать ответ и честно обозначить ограничения"
    ];
  }

  return [
    "Понять запрос и проверить известные локальные правила",
    "При необходимости найти и отсортировать материалы",
    "Составить ответ без выдумывания недостающих фактов",
    "Проверить формат и добавить полезные источники"
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

async function frossApiGuard(path, request, handler) {
  try {
    return await handler();
  } catch (error) {
    const errorId = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
    const diagnostic = String(error?.message || error || "Unknown error")
      .replace(/Bearer\s+\\S+/gi, "Bearer [redacted]")
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
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Access-Control-Allow-Origin": origin,
      "Vary": "Origin",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-FROSS-CREATOR-TOKEN"
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
  if (frossClean(messageFields.content?.stringValue || "").slice(0, FROSS_MAX_QUERY_LENGTH) !== expectedMessage) {
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

function frossBase64UrlEncode(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
function frossBase64UrlDecode(value) {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
  return Uint8Array.from(atob(base64), character => character.charCodeAt(0));
}
async function frossCreatorSigningKey(env) {
  const secret = typeof env.FROSS_CREATOR_PASSWORD === "string" ? env.FROSS_CREATOR_PASSWORD : "";
  if (secret.length < 24) return null;
  return crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}
async function frossCreateCreatorToken(auth, env) {
  const key = await frossCreatorSigningKey(env);
  if (!key) return null;
  const payload = frossBase64UrlEncode(new TextEncoder().encode(JSON.stringify({
    uid: auth.uid,
    exp: Date.now() + 12 * 60 * 60 * 1000,
    version: "0.5"
  })));
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return payload + "." + frossBase64UrlEncode(new Uint8Array(signature));
}
async function frossHasCreatorToken(auth, env, request) {
  const token = request?.headers?.get("X-FROSS-CREATOR-TOKEN") || "";
  const parts = token.split(".");
  if (parts.length !== 2) return false;
  const key = await frossCreatorSigningKey(env);
  if (!key) return false;
  try {
    const valid = await crypto.subtle.verify("HMAC", key, frossBase64UrlDecode(parts[1]), new TextEncoder().encode(parts[0]));
    if (!valid) return false;
    const payload = JSON.parse(new TextDecoder().decode(frossBase64UrlDecode(parts[0])));
    return payload.uid === auth?.uid && Number(payload.exp) > Date.now() && payload.version === "0.5";
  } catch { return false; }
}
async function frossIsCreator(auth, env, request) {
  const creatorEmail = typeof env.FROSS_CREATOR_EMAIL === "string" ? env.FROSS_CREATOR_EMAIL.trim().toLowerCase() : "";
  const emailAllowed = Boolean(creatorEmail && auth?.emailVerified && auth?.email && auth.email.toLowerCase() === creatorEmail);
  return emailAllowed || await frossHasCreatorToken(auth, env, request);
}
async function handleFrossCreatorUnlock(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/creator-unlock." }, 405, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Сначала войди в аккаунт FROSS." }, 401, request);
  let body;
  try { body = await request.json(); } catch { return frossJson({ error: "Некорректный JSON." }, 400, request); }
  const expected = typeof env.FROSS_CREATOR_PASSWORD === "string" ? env.FROSS_CREATOR_PASSWORD : "";
  const supplied = typeof body?.password === "string" ? body.password : "";
  if (expected.length < 24) return frossJson({ error: "Пароль создателя ещё не настроен. Добавь Secret FROSS_CREATOR_PASSWORD в Cloudflare." }, 503, request);
  // Compare equal-length UTF-8 buffers without an early exit on differing bytes.
  const a = new TextEncoder().encode(expected);
  const b = new TextEncoder().encode(supplied);
  let diff = a.length ^ b.length;
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i++) diff |= (a[i % Math.max(1, a.length)] || 0) ^ (b[i % Math.max(1, b.length)] || 0);
  if (diff !== 0) return frossJson({ error: "Пароль создателя неверный." }, 403, request);
  const token = await frossCreateCreatorToken(auth, env);
  if (!token) return frossJson({ error: "Не удалось создать сессию создателя." }, 503, request);
  return frossJson({ ok: true, isCreator: true, creatorToken: token, expiresInSeconds: 43200 }, 200, request);
}
async function handleFrossIdentity(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/identity." }, 405, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Сначала войди в FROSS." }, 401, request);
  const isCreator = await frossIsCreator(auth, env, request);
  return frossJson({
    ok: true,
    isCreator,
    emailVerified: auth.emailVerified,
    engine: "FROSS Local Brain 0.5",
    learningEnabled: isCreator,
    brainRoutes: {
      pythonReady: Boolean(env.FROSS_PYTHON_BRAIN_URL && env.FROSS_BRAIN_SERVICE_TOKEN),
      cppReady: Boolean(env.FROSS_CPP_BRAIN_URL && env.FROSS_BRAIN_SERVICE_TOKEN)
    }
  }, 200, request);
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
  const sentences = String(text || "").replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+|\s+•\s+|\s+;\s+/)
    .map(value => value.trim()).filter(value => value.length >= 45);
  if (!sentences.length) return String(text || "").trim() ? [String(text).trim().slice(0, 280)] : [];
  const focus = [...new Set([...frossKnowledgeTokens(topic), ...frossKnowledgeTokens(title)])];
  const ranked = sentences.map((sentence, index) => {
    const words = new Set(frossKnowledgeTokens(sentence));
    const hits = focus.filter(word => words.has(word)).length;
    const density = focus.length ? hits / Math.min(focus.length, 8) : 0;
    return { sentence, index, score: density * 2 + Math.min(sentence.length, 240) / 1200 };
  }).sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked.slice(0, limit).sort((a, b) => a.index - b.index).map(item => item.sentence.slice(0, 300));
}
function frossBuildKnowledgeSummary(topic, title, text) {
  return frossBuildKnowledgeKeyPoints(topic, title, text, 3).join(" ").slice(0, 900);
}

async function handleFrossTrain(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/train." }, 405, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Сначала войди в FROSS." }, 401, request);
  if (!await frossIsCreator(auth, env, request)) return frossJson({ error: "Режим обучения доступен только создателю. Введи пароль создателя в окне обучения." }, 403, request);

  let body;
  try { body = await request.json(); }
  catch { return frossJson({ error: "Тело запроса должно быть JSON." }, 400, request); }
  const topic = frossClean(body.topic || "").slice(0, 180);
  if (topic.length < 2) return frossJson({ error: "Введи тему, которую нужно изучить." }, 400, request);

  const search = await frossRunSearch(topic, "reasoning");
  const candidates = search.results.slice(0, 8);
  const notes = await Promise.all(candidates.map(async item => {
    const url = typeof item.url === "string" && /^https?:\/\//i.test(item.url) ? item.url.slice(0, 700) : "";
    const pageExcerpt = url ? await frossReadPublicPageExcerpt(url) : "";
    const sourceText = (pageExcerpt || frossClean(item.snippet || "")).slice(0, 3200);
    return {
      schemaVersion: 2,
      kind: "source-note",
      topic,
      title: frossClean(item.title || "Источник без названия").slice(0, 260),
      url,
      snippet: sourceText,
      summary: frossBuildKnowledgeSummary(topic, item.title || "", sourceText),
      keyPoints: frossBuildKnowledgeKeyPoints(topic, item.title || "", sourceText, 5),
      keywords: frossBuildKnowledgeKeywords(topic, item.title || "", sourceText),
      source: frossClean(pageExcerpt ? "article text" : item.source || "web search excerpt").slice(0, 80),
      score: Number(Math.max(0, Math.min(1, Number(item.score) || 0)).toFixed(4))
    };
  })).filter(note => note.title && (note.snippet || note.url));

  if (!notes.length) return frossJson({
    ok: false, error: "По этой теме не удалось найти подходящих материалов. Попробуй сформулировать тему конкретнее.",
    providerErrors: search.errors
  }, 404, request);

  return frossJson({
    ok: true,
    engine: "FROSS Local Brain 0.5",
    topic,
    notes,
    count: notes.length,
    providerErrors: search.errors,
    notice: "FROSS попробовал прочитать публичные страницы и, если сайт не разрешил загрузку, сохранил поисковые фрагменты. Это заметки, а не дообучение модели; проверь источники и удали неточные записи."
  }, 200, request);
}

function frossShouldSearch(query) {
  const q = frossClean(frossCorrectText(query)).toLowerCase().replace(/ё/g, "е");
  if (/\b(найди|поищи|поиск|в интернете|источник|источники|ссылки|актуальн|последние новости|новости|сегодня|сейчас|курс валют|цена|сколько стоит|обзор источников)\b/.test(q)) return true;
  if (/^(что такое|кто такой|кто такая|кто такие|кто создал|кто был|кто была|где находится|где расположен|когда был|когда была|какая столица|сколько населения|как работает|как устроен|почему происходит|чем известен)\b/.test(q)) return true;
  if (/^(объясни|расскажи про|расскажи о|определение|история|сравни|перечисли|найди информацию о|узнай о)\b/.test(q)) return true;
  // If the local rules did not recognize a factual question, try a web lookup rather than inventing.
  if (/[?？]\s*$/.test(q) && /^(кто|что|где|когда|почему|как|сколько|зачем|чем|какой|какая|какие|какое|есть ли|who|what|where|when|why|how|how many|how much)\b/.test(q)) return true;
  return false;
}

async function frossCallBrainService(endpoint, token, payload, timeoutMs = 18000) {
  if (typeof endpoint !== "string" || !endpoint.trim()) return { configured: false, handled: false };
  if (typeof token !== "string" || token.length < 24) {
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
      return { configured: true, handled: false, error: "Brain service did not handle this request." };
    }
    return {
      configured: true,
      handled: true,
      reply: data.reply.slice(0, 16000),
      engine: typeof data.engine === "string" ? data.engine.slice(0, 80) : "FROSS external brain",
      kind: typeof data.kind === "string" ? data.kind.slice(0, 80) : "remote-brain",
      trace: Array.isArray(data.trace) ? data.trace.slice(0, 7).map(item => String(item).slice(0, 120)) : [],
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

async function handleFrossApi(request, env) {
  if (request.method === "OPTIONS") return corsPreflight();
  if (request.method !== "POST") return frossJson({ error: "Используй POST /api/chat." }, 405, request);
  let body;
  try { body = await request.json(); }
  catch { return frossJson({ error: "Тело запроса должно быть JSON." }, 400, request); }
  const message = frossClean(body.message || "").slice(0, FROSS_MAX_QUERY_LENGTH);
  if (!message) return frossJson({ error: "Напиши сообщение для FROSS." }, 400, request);
  const auth = await frossVerifyFirebaseToken(request);
  if (!auth) return frossJson({ error: "Войди через Google, чтобы пользоваться FROSS." }, 401, request);
  // Private creator email is held as a Cloudflare secret, never added to the public code or brain inputs.
  const isCreator = await frossIsCreator(auth, env, request);
  const quota = await frossGetChatQuota(request, auth, String(body.chatId || ""), String(body.messageId || ""), message);
  if (!quota.ok) return frossJson({ error: quota.error }, quota.status, request);
  const correctedQuery = frossCorrectText(message);
  const safetyReply = frossSafetyReply(message);
  if (safetyReply) return frossJson({
    name: FROSS_PROFILE.name, reply: safetyReply, mode: "safety",
    steps: ["Понять запрос", "Проверить безопасность", "Предложить безопасную альтернативу"],
    sources: [], quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
    diagnostics: { engine: "FROSS Local Brain 0.5", aiUsed: false, note: "Ответ создан локальными правилами безопасности." }
  }, 200, request);

  const mode = ["reasoning", "рассуждение", "deep"].includes(String(body.mode || "").toLowerCase()) ? "reasoning" : "normal";
  const history = Array.isArray(body.history) ? body.history.slice(-8).map(item => ({
    role: item?.role === "assistant" ? "assistant" : "user",
    content: frossClean(item?.content || "").slice(0, 4000)
  })) : [];
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
  const quickReply = frossQuickResponse({ message: correctedQuery, isCreator, mode });
  const shouldSearch = !quickReply && frossShouldSearch(correctedQuery);
  let results = [];
  let providerErrors = [];
  if (shouldSearch) {
    const search = await frossRunSearch(frossAddContext(correctedQuery, history), mode);
    results = search.results;
    providerErrors = search.errors;
  }
  const localMemory = frossThink({ message: correctedQuery, history, knowledge, results: [], mode, isCreator });
  if (localMemory.kind === "saved-memory-v06" && !shouldSearch) {
    return frossJson({
      name: FROSS_PROFILE.name,
      reply: localMemory.reply,
      mode,
      steps: localMemory.trace?.slice(0, mode === "reasoning" ? 7 : 4) || frossSteps(mode),
      sources: (localMemory.sources || []).slice(0, 8).map(({ title, url, snippet, source, score }) => ({ title, url, snippet, source, score })),
      quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
      diagnostics: { engine: "FROSS personal knowledge", aiUsed: false, kind: localMemory.kind, remoteBrain: false, version: "0.5" }
    }, 200, request);
  }

  // Delegate algorithmic tasks to Python or C++; for questions needing fresh facts,
  // merge saved notes with current search snippets locally so neither source is discarded.
  if (!shouldSearch || (!results.length && localMemory.kind !== "saved-memory-v06")) {
    const selectedEndpoint = mode === "reasoning" ? env.FROSS_CPP_BRAIN_URL : env.FROSS_PYTHON_BRAIN_URL;
    const remote = await frossCallBrainService(selectedEndpoint, env.FROSS_BRAIN_SERVICE_TOKEN, {
      message: correctedQuery,
      mode,
      history,
      knowledge: knowledge.slice(0, 24)
    });
    if (remote.handled) {
      return frossJson({
        name: FROSS_PROFILE.name,
        reply: remote.reply,
        mode,
        steps: remote.trace.length ? remote.trace.slice(0, mode === "reasoning" ? 7 : 4) : frossSteps(mode),
        sources: remote.sources.map(({ title, url, snippet, source, score }) => ({
          title: String(title || "").slice(0, 260),
          url: typeof url === "string" && /^https?:\/\//i.test(url) ? url.slice(0, 700) : "",
          snippet: String(snippet || "").slice(0, 3200),
          source: String(source || "").slice(0, 80),
          score: Number((Number(score) || 0).toFixed(3))
        })),
        quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
        diagnostics: {
          engine: remote.engine,
          aiUsed: false,
          kind: remote.kind,
          remoteBrain: true,
          version: "0.5",
          note: "Ответ сформирован самописным Python/C++ алгоритмическим движком."
        }
      }, 200, request);
    }
  }

  const localResult = quickReply
    ? { reply: quickReply, kind: "local-rule-v05", sources: [] }
    : frossThink({ message: correctedQuery, history, results, knowledge, mode, isCreator });
  const reply = localResult.reply;
  return frossJson({
    name: FROSS_PROFILE.name, reply, mode, steps: frossSteps(mode),
    sources: (localResult.sources?.length ? localResult.sources : results).slice(0, 8).map(({ title, url, snippet, source, score }) => ({
      title, url, snippet, source, score: Number((score || 0).toFixed(3))
    })),
    profile: { description: FROSS_PROFILE.description, traits: FROSS_PROFILE.traits, style: FROSS_PROFILE.style, rules: FROSS_PROFILE.rules },
    quota: { used: quota.count, limit: 60, closed: quota.status === "closed" },
    diagnostics: {
      resultsFound: results.length, providerErrors: remote.error ? [remote.error] : [], aiUsed: false,
      engine: "FROSS Local Brain 0.5 local fallback",
      remoteBrainConfigured: remote.configured,
      kind: localResult.kind || "local-rule-v05",
      correctedQuery: correctedQuery !== message.toLowerCase().replace(/ё/g, "е") ? correctedQuery : null,
      note: "Ответ сформирован собственным правиловым движком FROSS; готовая языковая модель не вызывается."
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