// FROSS JavaScript request preparation.
// Text generation is performed by the original Transformer in fross_transformer.py.
// This module intentionally contains no canned answers or local response templates.

const TYPO_ALIASES = Object.freeze({
  "созрдайтель": "создатель", "создотель": "создатель", "создатиль": "создатель",
  "создател": "создатель", "создител": "создатель", "саздатель": "создатель",
  "создатeль": "создатель", "создательь": "создатель",
  "разроботчик": "разработчик", "разработчк": "разработчик",
  "привте": "привет", "привит": "привет", "приве": "привет", "превет": "привет", "преветт": "привет",
  "фрос": "фросс", "фроссс": "фросс",
  "памаги": "помоги", "помаги": "помоги", "обясни": "объясни", "объесни": "объясни",
  "пачему": "почему", "почиму": "почему", "сколька": "сколько", "скока": "сколько",
  "обесни": "объясни", "обьясни": "объясни", "програма": "программа",
  "програмирование": "программирование", "интелект": "интеллект",
  "нейросит": "нейросеть", "нейросетьь": "нейросеть",
  "функцыя": "функция", "алгаритм": "алгоритм", "алгорит": "алгоритм",
  "неправельно": "неправильно", "неправельные": "неправильные", "неправельный": "неправильный"
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

function editDistance(left, right, stopAt = 2) {
  if (Math.abs(left.length - right.length) > stopAt) return stopAt + 1;
  let previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let i = 1; i <= left.length; i++) {
    const current = [i];
    let rowMin = i;
    for (let j = 1; j <= right.length; j++) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1;
      current[j] = Math.min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost);
      rowMin = Math.min(rowMin, current[j]);
    }
    if (rowMin > stopAt) return stopAt + 1;
    previous = current;
  }
  return previous[right.length];
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

/** Correct known typos without changing the user's original capitalization or code. */
export function frossCorrectText(text) {
  return String(text || "").replace(/[a-zа-яё]+/gi, token => {
    const lower = token.toLowerCase().replace(/ё/g, "е");
    const corrected = correctToken(lower);
    if (corrected === lower) return token;
    if (token === token.toUpperCase()) return corrected.toUpperCase();
    if (token[0] === token[0].toUpperCase()) {
      return corrected.charAt(0).toUpperCase() + corrected.slice(1);
    }
    return corrected;
  });
}
