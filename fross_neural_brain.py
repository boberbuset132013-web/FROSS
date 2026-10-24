#!/usr/bin/env python3
"""FROSS Neural Intent Router 0.2.

A small feed-forward neural network built with NumPy: hashed text features ->
ReLU hidden layer -> softmax intent classes. It is trained with cross-entropy
and backpropagation, and its learned weights are saved in an NPZ artifact.
It classifies intent for routing/fallback. Free-form responses are produced by the configured language-model client; this network is not a text generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "FROSS Neural Intent Router 0.2"
FEATURE_COUNT = 384
HIDDEN_COUNT = 48
EPOCHS = 300
LEARNING_RATE = 0.75
DEFAULT_MODEL_PATH = Path(__file__).with_name("fross-neural-model.npz")
MIN_CONFIDENCE = 0.58
MIN_MARGIN = 0.08

# Balanced, hand-labelled examples. Add real, reviewed examples to improve it.
TRAINING_EXAMPLES: dict[str, tuple[str, ...]] = {
    "greeting": (
        "привет", "приветик", "привет фросс", "здравствуй", "здравствуйте",
        "хай", "хей", "йо", "салют", "доброе утро", "добрый день", "добрый вечер",
        "ну привет", "здарова", "рад тебя видеть",
    ),
    "thanks": (
        "спасибо", "большое спасибо", "спасибо большое", "спс", "пасиб",
        "благодарю", "очень помог", "спасибо за помощь", "респект", "это помогло",
        "спасибо фросс", "благодарю тебя", "спасибо огромное", "вот теперь понятно",
    ),
    "goodbye": (
        "пока", "до встречи", "до свидания", "увидимся", "я пошёл", "я пошла",
        "до скорого", "ладно я ушёл", "спокойной ночи", "мне пора", "пока фросс",
        "ещё увидимся", "на сегодня хватит", "до завтра",
    ),
    "small_talk": (
        "как дела", "как ты", "что нового", "чем занимаешься", "ты тут",
        "ты на связи", "ау", "как настроение", "как поживаешь", "что делаешь",
        "ты здесь", "как жизнь", "чем занят", "как у тебя дела",
        "что у тебя нового", "что у вас нового", "что новенького", "есть новости",
        "как у тебя дела сегодня", "как проходит день", "как ты сегодня",
        "как настроение сегодня", "как жизнь проходит", "чем живёшь",
        "как дела у тебя", "есть что рассказать", "какие новости",
    ),
    "capabilities": (
        "что ты умеешь", "что ты можешь", "какие у тебя возможности",
        "чем можешь помочь", "расскажи о своих функциях", "покажи что умеешь",
        "какие у тебя способности", "для чего ты нужен", "что умеет фросс",
        "чем ты полезен", "какие задачи решаешь", "что можно у тебя спросить",
    ),
    "identity": (
        "кто ты", "как тебя зовут", "что ты за бот", "ты человек",
        "ты нейросеть", "какая у тебя модель", "ты живой", "у тебя есть чувства",
        "ты настоящий", "ты искусственный интеллект", "ты программа", "ты фросс",
    ),
    "casual_chat": (
        "давай поговорим", "поговорим", "мне скучно", "хочу поболтать",
        "не знаю о чём поговорить", "давай поболтаем", "хочу пообщаться",
        "расскажи что-нибудь", "можем просто поговорить", "давай отвлечёмся",
        "хочется с кем-то поговорить", "давай обсудим что-то интересное",
    ),
    "idea": (
        "у меня идея", "у меня есть идея", "есть идея", "у меня задумка",
        "я придумал кое-что", "я придумала кое-что", "хочу поделиться идеей",
        "есть одна задумка", "мне пришла мысль", "я кое-что придумал",
        "у меня есть интересная задумка", "у меня появилась идея",
    ),
    "project_start": (
        "я хочу создать игру", "хочу сделать сайт", "я хочу разработать приложение",
        "хочу создать своего бота", "я собираюсь сделать игру", "помоги начать проект",
        "хочу сделать проект", "я хочу создать программу", "как начать делать сайт",
        "хочу разработать свою игру", "собираюсь создать приложение",
        "хочу сделать что-то своё", "я хочу построить свой проект",
    ),
    "positive": (
        "да", "ага", "угу", "ок", "окей", "ладно", "хорошо", "понятно",
        "договорились", "верно", "именно", "конечно", "согласен", "давай",
        "всё ясно", "ясно", "точно", "подходит", "отлично, продолжай",
    ),
    "negative": (
        "нет", "неа", "не надо", "не нужно", "не хочу", "не подходит",
        "не согласен", "не делай так", "не продолжай", "это не подходит",
        "давай по-другому", "так не надо", "мне это не нужно",
    ),
    "correction": (
        "не то", "не так", "ты не понял", "ты меня не понял",
        "я имел в виду другое", "я имела в виду другое", "я про другое",
        "не это я имел в виду", "ты неправильно понял", "я хотел сказать другое",
        "я хотела сказать другое", "я не это просил", "я не это просила",
    ),
    "error_help": (
        "не работает", "ничего не работает", "не получается", "не сработало",
        "у меня ошибка", "вылезла ошибка", "сломалось", "не запускается",
        "опять ошибка", "код падает", "программа не запускается",
        "получаю ошибку", "не могу запустить", "всё сломалось",
    ),
    "followup_detail": (
        "подробнее", "расскажи подробнее", "объясни подробнее", "продолжи",
        "продолжай", "можно подробнее", "расскажи чуть больше", "а дальше",
        "и что потом", "разверни ответ", "объясни попроще", "можно пример",
        "расскажи больше", "продолжи мысль", "поясни этот момент",
        "можно рассказать поподробнее", "можно рассказать подробнее",
        "можешь рассказать подробнее", "объясни чуть поподробнее",
    ),
    "followup_why": (
        "почему", "а почему", "а почему так", "почему это происходит",
        "а в чём причина", "из-за чего", "почему именно так", "а с чего это",
        "почему так получилось", "в чём причина", "почему это случилось",
        "а почему это важно",
    ),
    "followup_whatif": (
        "а если иначе", "что если поменять условие", "а что будет если",
        "а если я изменю", "предположим что", "а что если так",
        "что произойдёт если", "а если сделать по-другому",
        "а если использовать другой вариант", "а если изменить это",
    ),
    "memory_question": (
        "помнишь", "ты помнишь", "о чём мы говорили", "что я тебе рассказывал",
        "помнишь наш разговор", "что я говорил раньше", "что я тебе писал",
        "помнишь мою идею", "о чём я спрашивал", "ты помнишь предыдущий вопрос",
        "что было в прошлом сообщении", "помнишь что я сказал",
    ),
    "feelings": (
        "мне грустно", "мне плохо морально", "я расстроен", "я расстроена",
        "у меня плохое настроение", "мне обидно", "мне неприятно", "я устал",
        "я устала", "я злюсь", "я переживаю", "у меня тяжёлый день",
        "мне одиноко", "мне сейчас непросто",
    ),
}

INTENT_RESPONSES = {
    "greeting": "Привет! Я на связи. Что обсудим?",
    "thanks": "Пожалуйста! Хорошо, что пригодилось. Что ещё разберём?",
    "goodbye": "До встречи! Возвращайся, когда понадобится помощь.",
    "small_talk": "Я на связи и разбираю твои сообщения. А у тебя как дела? Что интересного происходит?",
    "capabilities": "Я могу считать выражения, объяснять известные технические темы, помогать с шаблонами кода, искать материалы и разбирать короткие продолжения разговора. На сложных новых темах мои возможности пока ограничены.",
    "identity": "Я FROSS — экспериментальный помощник с правилами, поиском и небольшим обучаемым классификатором намерений. Я не человек и не полноценная языковая модель.",
    "casual_chat": "Давай. Можем поговорить об играх, программировании, твоих проектах или выбрать совершенно другую тему. Что сейчас интереснее?",
    "idea": "Расскажи идею в двух-трёх предложениях. Я помогу выделить главное, заметить возможные сложности и придумать, с чего начать.",
    "project_start": "Давай превратим задумку в план: определим главную функцию, соберём маленький прототип и проверим его. Что проект должен уметь в самой первой версии?",
    "positive": "Понял. Продолжим с учётом этого ответа — можешь прислать следующую деталь или показать результат.",
    "negative": "Понял, этот вариант не подходит. Можем изменить подход или перейти к другой задаче.",
    "correction": "Похоже, я неправильно понял запрос. Напиши, какую именно деталь я упустил, — скорректирую ответ, а не буду повторять прежний.",
    "error_help": "Давай найдём причину без догадок. Пришли точный текст ошибки и укажи, после какого действия она появляется.",
    "followup_detail": "Продолжу предыдущую тему. Я могу разложить прошлый ответ на шаги, привести пример или объяснить отдельный пункт. Что полезнее?",
    "followup_why": "Связываю вопрос с предыдущей темой. Укажи утверждение, которое вызывает вопрос «почему», и я объясню его, не придумывая отсутствующие детали.",
    "followup_whatif": "Можем сравнить варианты. Напиши, какое условие хочешь изменить, и я попробую объяснить последствия.",
    "memory_question": "Я могу учитывать только ту историю чата, которую сайт передал в текущем запросе. Если напомнишь тему или фразу, я продолжу с неё.",
    "feelings": "Жаль, что сейчас так. Можем спокойно разобрать, что произошло, или переключиться на более лёгкую тему — как тебе удобнее.",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold().replace("ё", "е")).strip()


def _hash_feature(name: str) -> tuple[int, float]:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "little") % FEATURE_COUNT
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


def encode_text(text: str) -> np.ndarray:
    """Encode a message with deterministic word, bigram and character features."""
    normalized = normalize(text)[:500]
    vector = np.zeros(FEATURE_COUNT, dtype=np.float64)
    tokens = re.findall(r"[a-zа-я0-9+#]+", normalized)
    features: list[tuple[str, float]] = []
    features.extend(("w:" + token, 2.0) for token in tokens)
    features.extend(("b:" + tokens[i] + "_" + tokens[i + 1], 1.35)
                    for i in range(len(tokens) - 1))
    padded = "  " + normalized + "  "
    for size in (3, 4, 5):
        features.extend(("c" + str(size) + ":" + padded[i:i + size], 0.28)
                        for i in range(max(0, len(padded) - size + 1)))
    for name, weight in features:
        bucket, sign = _hash_feature(name)
        vector[bucket] += weight * sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


class NeuralIntentModel:
    def __init__(self, path: str | Path):
        with np.load(str(path), allow_pickle=False) as data:
            if int(data["format_version"][0]) != 1:
                raise ValueError("Unsupported FROSS neural model format.")
            self.w1 = data["w1"].astype(np.float64, copy=True)
            self.b1 = data["b1"].astype(np.float64, copy=True)
            self.w2 = data["w2"].astype(np.float64, copy=True)
            self.b2 = data["b2"].astype(np.float64, copy=True)
            self.labels = tuple(str(item) for item in data["labels"].tolist())
            self.train_accuracy = float(data["train_accuracy"][0])
            self.final_loss = float(data["final_loss"][0])
        if self.w1.shape != (FEATURE_COUNT, HIDDEN_COUNT):
            raise ValueError("Invalid first-layer weight shape.")
        if self.w2.shape != (HIDDEN_COUNT, len(self.labels)):
            raise ValueError("Invalid output-layer weight shape.")

    def probabilities(self, text: str) -> np.ndarray:
        x = encode_text(text).reshape(1, -1)
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        return _softmax(hidden @ self.w2 + self.b2)[0]

    def predict(self, text: str) -> dict[str, Any]:
        probs = self.probabilities(text)
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else best
        return {
            "intent": self.labels[best],
            "confidence": float(probs[best]),
            "margin": float(probs[best] - probs[runner_up]),
            "alternatives": [
                {"intent": self.labels[int(index)], "confidence": float(probs[int(index)])}
                for index in order[:3]
            ],
        }


def training_arrays() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels = tuple(TRAINING_EXAMPLES.keys())
    texts: list[str] = []
    targets: list[int] = []
    for index, label in enumerate(labels):
        for text in TRAINING_EXAMPLES[label]:
            texts.append(text)
            targets.append(index)
    x = np.stack([encode_text(text) for text in texts])
    y = np.asarray(targets, dtype=np.int64)
    return x, y, labels


def train_model(output_path: str | Path = DEFAULT_MODEL_PATH, epochs: int = EPOCHS) -> dict[str, Any]:
    """Train a one-hidden-layer classifier with softmax loss and backpropagation."""
    x, y, labels = training_arrays()
    rng = np.random.default_rng(20261024)
    w1 = rng.normal(0.0, np.sqrt(2.0 / FEATURE_COUNT), (FEATURE_COUNT, HIDDEN_COUNT))
    b1 = np.zeros((1, HIDDEN_COUNT), dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / HIDDEN_COUNT), (HIDDEN_COUNT, len(labels)))
    b2 = np.zeros((1, len(labels)), dtype=np.float64)
    one_hot = np.zeros((len(y), len(labels)), dtype=np.float64)
    one_hot[np.arange(len(y)), y] = 1.0
    final_loss = 0.0

    for _ in range(max(1, int(epochs))):
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        probs = _softmax(a1 @ w2 + b2)
        final_loss = float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))))
        dz2 = (probs - one_hot) / len(y)
        dw2 = a1.T @ dz2 + 0.00015 * w2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ w2.T
        dz1 = da1 * (z1 > 0.0)
        dw1 = x.T @ dz1 + 0.00015 * w1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        # A stable global gradient clip prevents a bad batch from destabilizing training.
        norm = np.sqrt(np.sum(dw1 * dw1) + np.sum(db1 * db1) + np.sum(dw2 * dw2) + np.sum(db2 * db2))
        scale = min(1.0, 5.0 / max(float(norm), 1e-12))
        w1 -= LEARNING_RATE * scale * dw1
        b1 -= LEARNING_RATE * scale * db1
        w2 -= LEARNING_RATE * scale * dw2
        b2 -= LEARNING_RATE * scale * db2

    z1 = x @ w1 + b1
    probs = _softmax(np.maximum(z1, 0.0) @ w2 + b2)
    accuracy = float(np.mean(np.argmax(probs, axis=1) == y))
    # Measure final loss after the last parameter update.
    final_loss = float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        format_version=np.asarray([1], dtype=np.int64),
        w1=w1, b1=b1, w2=w2, b2=b2,
        labels=np.asarray(labels, dtype=np.str_),
        train_accuracy=np.asarray([accuracy], dtype=np.float64),
        final_loss=np.asarray([final_loss], dtype=np.float64),
    )
    return {
        "version": VERSION,
        "output": str(output),
        "examples": int(len(y)),
        "intents": len(labels),
        "epochs": int(epochs),
        "train_accuracy": accuracy,
        "final_loss": final_loss,
    }


def load_or_train(path: str | Path = DEFAULT_MODEL_PATH) -> tuple[NeuralIntentModel | None, dict[str, Any]]:
    target = Path(path)
    try:
        return NeuralIntentModel(target), {"loaded": True, "path": str(target)}
    except (OSError, ValueError, KeyError, EOFError):
        try:
            metrics = train_model(target)
            return NeuralIntentModel(target), {"loaded": False, **metrics}
        except (OSError, ValueError, FloatingPointError) as error:
            return None, {"loaded": False, "error": type(error).__name__}


def build_reply(
    message: str,
    prediction: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    """Turn a high-confidence class into a safe, short conversational action."""
    intent = str(prediction.get("intent", ""))
    confidence = float(prediction.get("confidence", 0.0))
    margin = float(prediction.get("margin", 0.0))
    if confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
        return None
    turns = [item for item in (history or [])[-12:] if isinstance(item, dict)]
    last_assistant = next((str(item.get("content", "")) for item in reversed(turns)
                           if item.get("role") == "assistant"), "")
    last_user = next((str(item.get("content", "")) for item in reversed(turns)
                      if item.get("role") != "assistant"), "")
    clean_previous = re.sub(r"[\u0060]{3}[\s\S]*?[\u0060]{3}", " пример кода ", last_assistant)
    clean_previous = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_previous)
    clean_previous = re.sub(r"https?://\S+", "", clean_previous)
    clean_previous = re.sub(r"[*_#>\u0060]", " ", clean_previous)
    clean_previous = re.sub(r"\s+", " ", clean_previous).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])", clean_previous)
                 if len(part.strip()) >= 18][:3]

    if intent == "followup_detail" and clean_previous:
        fragments = [part for part in sentences if not part.lower().startswith(("источники", "ограничение"))]
        summary = "\n".join(f"{index}. {part[:240]}" for index, part in enumerate(fragments[:3], 1))
        return "Продолжу предыдущую тему. Вот основные части прежнего ответа:\n" + (summary or clean_previous[:450]) + \
               "\n\nМогу объяснить любой пункт проще или привести пример. Что раскрыть первым?"
    if intent == "followup_why" and clean_previous:
        cause = next((part for part in sentences if re.search(r"\b(потому что|так как|поскольку|из-за|поэтому)\b", part, re.I)), "")
        if cause:
            return "В предыдущем ответе причина была сформулирована так: «" + cause[:300] + "». Если ты спрашиваешь о другой детали, укажи её."
        return "Связываю вопрос с предыдущим ответом: «" + clean_previous[:300] + "». Там причина раскрыта не полностью, поэтому не буду придумывать факты. Укажи, какое утверждение разобрать."
    if intent == "followup_whatif":
        return "Чтобы сравнить варианты, нужно понять, что именно изменить. Последняя тема из переписки: «" + (clean_previous[:220] or "контекст не передан") + "». Какое условие проверим?"
    if intent == "memory_question":
        if last_user:
            return "В доступном мне контексте последняя тема от тебя была такой: «" + last_user[:220] + "». Я могу использовать только историю, которую сайт приложил к запросу."
        return INTENT_RESPONSES[intent]
    if intent == "project_start":
        match = re.search(r"\b(?:создать|сделать|разработать|построить|запустить|придумать)\s+(.{2,120})", normalize(message))
        if match:
            project = match.group(1).strip(" .!?")
            return "Давай начнём с проекта «" + project[:100] + "». Сначала определим главную функцию, затем соберём минимальный прототип и проверим его. Что он должен уметь в первой версии?"
    if intent == "positive" and last_assistant.strip().endswith("?"):
        return "Хорошо, учту твой ответ. Продолжим с этой договорённостью."
    return INTENT_RESPONSES.get(intent)


def main() -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("command", choices=("train", "test", "predict"))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--text", default="")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train_model(args.output), ensure_ascii=False, indent=2))
        return 0
    model, status = load_or_train(args.output)
    if model is None:
        print(json.dumps({"ok": False, "status": status}, ensure_ascii=False))
        return 1
    if args.command == "predict":
        print(json.dumps(model.predict(args.text), ensure_ascii=False, indent=2))
        return 0
    x, y, labels = training_arrays()
    probs = model.probabilities(args.text or "приветик")
    prediction = model.predict(args.text or "приветик")
    checks = [
        ("приветик", "greeting"),
        ("как у тебя дела", "small_talk"),
        ("что у тебя нового", "small_talk"),
        ("спасибо огромное", "thanks"),
        ("хочу просто пообщаться", "casual_chat"),
        ("я не это имел в виду", "correction"),
        ("у меня опять ошибка", "error_help"),
        ("можно рассказать поподробнее", "followup_detail"),
        ("а почему так получилось", "followup_why"),
        ("а что будет если поменять условие", "followup_whatif"),
        ("помнишь наш прошлый разговор", "memory_question"),
        ("у меня есть задумка", "idea"),
    ]
    failures = []
    for text, expected in checks:
        actual = model.predict(text)["intent"]
        if actual != expected:
            failures.append({"text": text, "expected": expected, "actual": actual})
    print(json.dumps({
        "ok": not failures,
        "version": VERSION,
        "train_accuracy": model.train_accuracy,
        "final_loss": model.final_loss,
        "self_tests": len(checks),
        "failures": failures,
        "example_prediction": prediction,
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
