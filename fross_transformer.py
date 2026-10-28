#!/usr/bin/env python3
"""FROSS Transformer 1.5: a small decoder-only language model trained from random weights.

No pretrained checkpoint, hosted LLM, or downloaded model is used. The current
training path learns a SentencePiece Unigram tokenizer and a causal Transformer from
the repository's own corpus; legacy BPE checkpoints remain supported.
This educational model's quality is bounded by its data and compute budget.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, OrderedDict
import json
import math
import os
from pathlib import Path
import random
import re
import string
import tempfile
from typing import Any

import torch
import sentencepiece as spm
from torch import nn
import torch.nn.functional as F

VERSION = "FROSS Transformer 1.5"


def _preferred_device() -> torch.device:
    """Use CUDA only when the installed PyTorch build can actually access it."""
    requested = os.environ.get("FROSS_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_runtime_threads() -> int:
    """Limit PyTorch CPU thread pools for small containers and shared CPU quotas."""
    try:
        requested = int(os.environ.get("FROSS_TORCH_THREADS", "1"))
    except (TypeError, ValueError):
        requested = 1
    thread_count = max(1, min(2, requested))
    try:
        torch.set_num_threads(thread_count)
    except RuntimeError:
        # PyTorch may reject changes after parallel work has already started.
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # This can only be set before PyTorch starts inter-op work.
        pass
    return thread_count
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
# Keep the response prefix identical in training examples and generation prompts.
RESPONSE_PREFIX = "FROSS: "
BASE_CHARS = "".join(dict.fromkeys(
    string.ascii_letters + string.digits + string.punctuation +
    " \n\r\t" +
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" +
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" +
    "іІїЇєЄґҐ" +
    "—–…«»„“”‘’№°·•→←✓✗≤≥≈"
))
DEFAULT_CORPUS = Path(__file__).with_name("data") / "fross_corpus.jsonl"
DEFAULT_CHECKPOINT = Path(__file__).with_name("fross-transformer.pt")
MAX_GENERATION_TOKENS = 5000

# Sized for the 1 GiB / low-CPU Hostless runtime: wider and deeper than Brain 1.3
# while keeping CPU training/inference practical. Embeddings are weight-tied.
DEFAULT_CONFIG = {
    "block_size": 320,
    "n_embd": 192,
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.08,
}


_PRETOKEN_PATTERN = re.compile(r"\s+|[^\W_]+|_+|[^\w\s]+", flags=re.UNICODE)


class BPETokenizer:
    """BPE tokenizer trained on word/whitespace/punctuation pieces.

    New checkpoints use pretokenized=True so BPE never merges across word boundaries.
    Older checkpoints omit that flag and retain their original character-stream encoding.
    """

    def __init__(
        self,
        base_chars: str = BASE_CHARS,
        merges: list[tuple[int, int]] | None = None,
        pretokenized: bool = False,
        max_piece_chars: int = 24,
    ):
        self.base_chars = "".join(dict.fromkeys(base_chars))
        self.char_to_id = {char: index + 4 for index, char in enumerate(self.base_chars)}
        self.merges: list[tuple[int, int]] = []
        self.pieces = list(SPECIAL_TOKENS) + list(self.base_chars)
        self.pretokenized = bool(pretokenized)
        self.max_piece_chars = max(2, int(max_piece_chars))
        for left, right in (merges or []):
            left, right = int(left), int(right)
            if left < 0 or right < 0 or left >= len(self.pieces) or right >= len(self.pieces):
                raise ValueError("Invalid BPE merge in tokenizer checkpoint.")
            self.merges.append((left, right))
            self.pieces.append(self.pieces[left] + self.pieces[right])

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    @classmethod
    def train(
        cls,
        texts: list[str],
        merge_count: int | None = None,
        min_frequency: int | None = None,
        target_vocab_size: int | None = None,
        max_piece_chars: int = 24,
    ) -> "BPETokenizer":
        """Train a weighted BPE vocabulary, optionally targeting an exact vocabulary size."""
        # Preserve every Unicode character present in the training text as a
        # base token before BPE merges. This matters for scripts such as Chinese
        # that are not in the small hand-written Latin/Cyrillic fallback alphabet.
        corpus_characters = "".join(str(text) for text in texts)
        base_chars = "".join(dict.fromkeys(BASE_CHARS + corpus_characters))
        tokenizer = cls(base_chars=base_chars, pretokenized=True, max_piece_chars=max_piece_chars)
        base_vocab_size = len(tokenizer.pieces)
        if target_vocab_size is not None:
            target_vocab_size = int(target_vocab_size)
            if target_vocab_size < base_vocab_size:
                raise ValueError(
                    f"target_vocab_size must be at least {base_vocab_size} for the base character set"
                )
        if merge_count is None:
            merge_count = (
                max(0, target_vocab_size - base_vocab_size)
                if target_vocab_size is not None else 384
            )
        merge_count = max(0, int(merge_count))
        threshold = max(1, int(min_frequency if min_frequency is not None else (
            1 if target_vocab_size is not None else 2
        )))

        # Count distinct pre-tokenized word-like pieces. We weight pair counts by
        # corpus frequency, so repeated words influence BPE without repeatedly
        # scanning identical character sequences.
        segment_counts: Counter[str] = Counter()
        for text in texts:
            segment_counts.update(_PRETOKEN_PATTERN.findall(str(text)))
        sequences: list[tuple[list[int], int]] = []
        for segment, frequency in segment_counts.items():
            sequence = [tokenizer.char_to_id.get(char, UNK_ID) for char in segment]
            if sequence:
                sequences.append((sequence, int(frequency)))

        pieces = list(tokenizer.pieces)
        learned: list[tuple[int, int]] = []
        for _ in range(merge_count):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence, weight in sequences:
                for pair in zip(sequence, sequence[1:]):
                    left, right = pair
                    # Do not create trainable tokens that embed the literal <unk> marker.
                    if left == UNK_ID or right == UNK_ID:
                        continue
                    if len(pieces[left]) + len(pieces[right]) <= tokenizer.max_piece_chars:
                        counts[pair] += weight
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < threshold:
                break

            left, right = pair
            new_id = len(pieces)
            learned.append(pair)
            pieces.append(pieces[left] + pieces[right])
            merged_sequences: list[tuple[list[int], int]] = []
            for sequence, weight in sequences:
                result: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                merged_sequences.append((result, weight))
            sequences = merged_sequences

        return cls(
            tokenizer.base_chars,
            learned,
            pretokenized=True,
            max_piece_chars=max_piece_chars,
        )

    def encode(self, text: str) -> list[int]:
        raw = str(text)
        if not self.pretokenized:
            # Backward compatibility for checkpoints trained before Brain 1.4.
            sequence = [self.char_to_id.get(char, UNK_ID) for char in raw]
            for merge_index, pair in enumerate(self.merges):
                new_id = 4 + len(self.base_chars) + merge_index
                result: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                sequence = result
            return sequence

        encoded: list[int] = []
        for segment in _PRETOKEN_PATTERN.findall(raw):
            sequence = [self.char_to_id.get(char, UNK_ID) for char in segment]
            for merge_index, pair in enumerate(self.merges):
                new_id = 4 + len(self.base_chars) + merge_index
                result = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                sequence = result
            encoded.extend(sequence)
        return encoded

    def decode(self, token_ids: list[int], stop_at_eos: bool = True) -> str:
        parts: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == EOS_ID and stop_at_eos:
                break
            if token_id < 4 or token_id >= len(self.pieces):
                continue
            parts.append(self.pieces[token_id])
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_chars": self.base_chars,
            "merges": [list(pair) for pair in self.merges],
            "pretokenized": self.pretokenized,
            "max_piece_chars": self.max_piece_chars,
        }


class SentencePieceUnigramTokenizer:
    """SentencePiece Unigram tokenizer with FROSS-compatible special token IDs."""

    def __init__(self, model_proto: bytes | str):
        # Hex text is safe to serialize in PyTorch weights-only checkpoints.
        self.model_proto = bytes.fromhex(model_proto) if isinstance(model_proto, str) else bytes(model_proto)
        self.processor = spm.SentencePieceProcessor(model_proto=self.model_proto)
        self.pieces = [self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())]
        self.pretokenized = False
        # Cache repeated prompts/history fragments without changing token IDs.
        # Keep it bounded because the service may run with a small memory limit.
        self._encode_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._encode_cache_limit = 256
        if (self.processor.pad_id(), self.processor.unk_id(),
                self.processor.bos_id(), self.processor.eos_id()) != (PAD_ID, UNK_ID, BOS_ID, EOS_ID):
            raise ValueError("SentencePiece special token IDs are incompatible with FROSS.")

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @classmethod
    def train(cls, texts: list[str], target_vocab_size: int = 20000, **_: Any):
        if target_vocab_size < 32:
            raise ValueError("SentencePiece vocabulary target must be at least 32.")
        # SentencePiece requires a corpus file. Keep it temporary; the model proto
        # is embedded in the checkpoint so deployment needs no sidecar file.
        with tempfile.TemporaryDirectory(prefix="fross-unigram-") as temp_dir:
            input_path = Path(temp_dir) / "corpus.txt"
            prefix = Path(temp_dir) / "tokenizer"
            with input_path.open("w", encoding="utf-8", newline="\n") as handle:
                for text in texts:
                    line = str(text).replace("\r", " ").replace("\n", " ")
                    if line.strip():
                        handle.write(line + "\n")
            if input_path.stat().st_size == 0:
                raise ValueError("Cannot train SentencePiece on an empty corpus.")
            spm.SentencePieceTrainer.train(
                input=str(input_path), model_prefix=str(prefix), model_type="unigram",
                vocab_size=int(target_vocab_size), hard_vocab_limit=False,
                character_coverage=0.9995, normalization_rule_name="identity",
                pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
                pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
                shuffle_input_sentence=False, num_threads=1, max_sentence_length=16384,
                minloglevel=2,
            )
            return cls((prefix.with_suffix(".model")).read_bytes())

    def encode(self, text: str) -> list[int]:
        raw = str(text)
        cached = self._encode_cache.get(raw)
        if cached is not None:
            self._encode_cache.move_to_end(raw)
            # Return a new list so callers cannot mutate the cached sequence.
            return list(cached)

        encoded = tuple(self.processor.encode(raw, out_type=int))
        self._encode_cache[raw] = encoded
        self._encode_cache.move_to_end(raw)
        if len(self._encode_cache) > self._encode_cache_limit:
            self._encode_cache.popitem(last=False)
        return list(encoded)

    def decode(self, token_ids: list[int], stop_at_eos: bool = True) -> str:
        ids = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id == EOS_ID and stop_at_eos:
                break
            if 0 <= token_id < self.vocab_size and token_id not in {PAD_ID, BOS_ID, EOS_ID}:
                ids.append(token_id)
        return self.processor.decode(ids)

    def to_dict(self) -> dict[str, Any]:
        return {"model_proto": self.model_proto.hex()}


def _load_tokenizer(data: dict[str, Any]):
    if "model_proto" in data:
        return SentencePieceUnigramTokenizer(data["model_proto"])
    # Backward compatibility: existing BPE checkpoints remain loadable.
    return BPETokenizer(**data)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict[str, int | float]):
        super().__init__()
        n_embd, n_head = int(config["n_embd"]), int(config["n_head"])
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.out = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(float(config["dropout"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.view(batch_size, steps, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch_size, steps, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, steps, self.n_head, self.head_size).transpose(1, 2)
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch_size, steps, width)
        return self.dropout(self.out(output))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict[str, int | float]):
        super().__init__()
        n_embd = int(config["n_embd"])
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(float(config["dropout"])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class FrossTransformer(nn.Module):
    """Decoder-only, autoregressive Transformer. All weights start randomly."""

    def __init__(self, vocab_size: int, config: dict[str, int | float] | None = None):
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.vocab_size = int(vocab_size)
        n_embd = int(self.config["n_embd"])
        self.token_embedding = nn.Embedding(self.vocab_size, n_embd)
        self.position_embedding = nn.Embedding(int(self.config["block_size"]), n_embd)
        self.dropout = nn.Dropout(float(self.config["dropout"]))
        self.blocks = nn.Sequential(*[
            TransformerBlock(self.config) for _ in range(int(self.config["n_layer"]))
        ])
        self.final_norm = nn.LayerNorm(n_embd)
        self.language_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, steps = token_ids.shape
        if steps > int(self.config["block_size"]):
            token_ids = token_ids[:, -int(self.config["block_size"]):]
            steps = token_ids.shape[1]
        positions = torch.arange(steps, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.final_norm(self.blocks(self.dropout(x)))
        return self.language_head(x)


def _load_corpus(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {corpus_path}:{line_number}") from exc
        if not isinstance(row, dict):
            continue
        user = str(row.get("user", "")).strip()
        assistant = str(row.get("assistant", "")).strip()
        if not user or not assistant:
            continue
        # Optional dialogue history teaches the model how to use follow-ups and corrections.
        history: list[dict[str, str]] = []
        raw_history = row.get("history", [])
        if isinstance(raw_history, list):
            for turn in raw_history[-8:]:
                if not isinstance(turn, dict):
                    continue
                content = " ".join(str(turn.get("content", "")).split()).strip()
                if not content:
                    continue
                role = "assistant" if str(turn.get("role", "")).lower() in {"assistant", "fross"} else "user"
                history.append({"role": role, "content": content[:700]})
        records.append({"user": user, "assistant": assistant, "history": history})
    if len(records) < 8:
        raise ValueError("The corpus must contain at least 8 user/assistant examples.")
    return records


def _format_record(record: dict[str, Any]) -> str:
    # Match the same turn labels and ordering used by generate_reply().
    parts: list[str] = []
    history = record.get("history", [])
    if isinstance(history, list):
        for turn in history[-8:]:
            if not isinstance(turn, dict):
                continue
            content = " ".join(str(turn.get("content", "")).split()).strip()[:700]
            if not content:
                continue
            role = "FROSS" if str(turn.get("role", "")).lower() in {"assistant", "fross"} else "Пользователь"
            parts.append(f"{role}: {content}")
    parts.append(f"Пользователь: {str(record['user']).strip()}")
    parts.append(f"{RESPONSE_PREFIX}{str(record['assistant']).strip()}")
    return "\n".join(parts) + "\n"


def _response_strategy(message: str) -> str:
    """Give the compact model a task-specific checklist, without generating a canned reply."""
    query = " ".join(str(message or "").lower().replace("ё", "е").split()).strip()
    plain = query.translate(str.maketrans({char: " " for char in "!?.,;:()[]{}"}))
    plain = " ".join(plain.split()).strip()

    if re.fullmatch(r"(привет|приветик|здравствуй|здравствуйте|доброе утро|добрый вечер|hello|hi|спасибо|благодарю|пока|до встречи|ок|окей|ага|угу)", plain):
        return ("Тип запроса: короткая социальная реплика. Ответь естественно и кратко, "
                "не начинай случайную тему, не придумывай факты и не запускай объяснение без просьбы.")
    if re.fullmatch(r"(почему|а почему|что это значит|а если|продолжи|продолжай|не то|не так|я же сказал|я же сказала|ты не понял|ты меня не понял|я не это имел в виду|я не это имела в виду)", plain):
        return ("Тип запроса: продолжение или исправление диалога. Сначала используй последние подходящие реплики истории. "
                "Не начинай новую тему и не проси повторить уже предоставленные данные. "
                "При исправлении считай последнее уточнение главным.")
    if re.search(r"(^|[^a-zа-яё])(переведи|перевод|translate)(?=$|[^a-zа-яё])", plain):
        return ("Тип запроса: перевод. Сначала выдай точный перевод в запрошенный язык. "
                "Не добавляй разбор грамматики, аудио или дополнительные версии, если их не просили.")
    if re.search(r"(не меняй|оставь без изменений|только эту|только этот|только при наведении|не добавляй|не трогай|исправь только|переделай только)", plain):
        return ("Тип запроса: ограниченная правка. Сохрани все явно названные элементы без изменений, "
                "измени только требуемую часть и проверь результат по каждому ограничению.")
    if re.search(r"(напиши|сделай|создай|исправь|добавь|удали|переделай|сгенерируй|реализуй|код|пример кода)", plain):
        return ("Тип запроса: получить готовый результат или код. Выполни действие, а не подменяй его советом. "
                "Соблюдай указанный язык программирования, формат и ограничения. "
                "Не заявляй, что код запущен или проверен, если этого не было.")
    if re.search(r"(что такое|что значит|что означает|объясни|расскажи|почему|как работает|в чём разница|в чем разница|определи)", plain):
        return ("Тип запроса: вопрос или объяснение. Ответь на конкретный вопрос сначала; "
                "объясни незнакомые термины простыми словами и используй пример по теме. "
                "Разделяй подтверждённые сведения и предположения.")
    if re.search(r"(я сегодня|мне сегодня|я сейчас|мне сейчас|я злюсь|мне грустно|я расстроен|я расстроена|я устал|я устала)", plain):
        return ("Тип запроса: пользователь сообщает о состоянии или событии. "
                "Учитывай слова и контекст, отвечай с поддержкой, не ставь диагноз и не выдумывай причины.")
    return ("Общий контроль ответа: выдели желаемый результат, намерение, ограничения и важный контекст; "
            "выполни просьбу по формату пользователя; не выдумывай отсутствующие сведения; "
            "если данных недостаточно, обозначь пробел и задай только один действительно нужный вопрос.")


def _format_generation_prefix(message: str) -> str:
    """Build the exact response prefix seen in training examples."""
    return "Пользователь: " + str(message).strip()[:3000] + "\n" + RESPONSE_PREFIX


def _encode_stream(records: list[dict[str, Any]], tokenizer: BPETokenizer) -> list[int]:
    stream: list[int] = []
    for record in records:
        stream.append(BOS_ID)
        stream.extend(tokenizer.encode(_format_record(record)))
        stream.append(EOS_ID)
    return stream


@torch.no_grad()
def _evaluate(model: FrossTransformer, stream: list[int], block_size: int, batches: int = 8) -> float:
    if len(stream) <= block_size + 1:
        return float("nan")
    model.eval()
    values = []
    device = next(model.parameters()).device
    last_start = len(stream) - block_size - 1
    for _ in range(batches):
        starts = [random.randint(0, last_start) for _ in range(4)]
        x = torch.tensor([stream[i:i + block_size] for i in starts], dtype=torch.long, device=device)
        y = torch.tensor([stream[i + 1:i + block_size + 1] for i in starts], dtype=torch.long, device=device)
        logits = model(x)
        values.append(float(F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))))
    model.train()
    return sum(values) / len(values)


def train_model(
    corpus_path: str | Path = DEFAULT_CORPUS,
    output_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    steps: int = 600,
    batch_size: int = 4,
    block_size: int = 320,
    merge_count: int | None = None,
    target_vocab_size: int = 20000,
    seed: int = 1337,
    resume: bool = False,
    schedule_steps: int | None = None,
) -> dict[str, Any]:
    """Train to a cumulative step target, optionally resuming an exact saved optimizer state.

    When using staged training, pass the same schedule_steps on both runs. For example,
    train 600 steps with schedule_steps=5000, then resume with steps=5000.
    """
    if steps < 1 or batch_size < 1 or block_size < 32:
        raise ValueError("steps/batch_size must be positive and block_size must be at least 32")
    schedule_total = int(schedule_steps if schedule_steps is not None else steps)
    if schedule_total < steps:
        raise ValueError("schedule_steps must be greater than or equal to the cumulative --steps target")
    if resume and merge_count is not None:
        raise ValueError("Resuming with the legacy --merges tokenizer option is not supported; use --vocab-size.")

    target = Path(output_path)
    resume_checkpoint: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    if resume:
        if not target.is_file():
            raise FileNotFoundError(f"Cannot resume: checkpoint does not exist: {target}")
        resume_checkpoint = torch.load(target, map_location="cpu", weights_only=True)
        if int(resume_checkpoint.get("format_version", 0)) != 2:
            raise ValueError(
                "This checkpoint does not include resumable optimizer state. "
                "Start a new 600-step run with the current Brain 1.4 code first."
            )
        resume_state = resume_checkpoint.get("resume_state")
        if not isinstance(resume_state, dict):
            raise ValueError("Checkpoint is missing its resumable training state.")
        if int(resume_state.get("seed", -1)) != seed:
            raise ValueError("Resume seed differs from the saved run. Use the same --seed (default: 1337).")
        if int(resume_state.get("schedule_steps", -1)) != schedule_total:
            raise ValueError(
                "The learning-rate schedule does not match the saved run. "
                f"Use --schedule-steps {resume_state.get('schedule_steps')}."
            )

    configure_runtime_threads()
    random.seed(seed)
    torch.manual_seed(seed)
    corpus_file = Path(corpus_path)
    if not corpus_file.is_file():
        raise FileNotFoundError(f"Training corpus not found: {corpus_file}")
    corpus_sha256 = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    if resume and resume_state is not None:
        if resume_state.get("corpus_sha256") != corpus_sha256:
            raise ValueError(
                "The corpus has changed since the previous stage. Restore the exact original "
                "data/fross_corpus.jsonl before resuming."
            )

    records = _load_corpus(corpus_file)
    random.shuffle(records)
    val_count = max(2, int(len(records) * 0.1))
    validation_records, train_records = records[:val_count], records[val_count:]

    if resume and resume_checkpoint is not None:
        tokenizer = _load_tokenizer(resume_checkpoint["tokenizer"])
    else:
        tokenizer = SentencePieceUnigramTokenizer.train(
            [_format_record(row) for row in train_records],
            target_vocab_size=target_vocab_size or 20000,
        )

    train_stream = _encode_stream(train_records, tokenizer)
    validation_stream = _encode_stream(validation_records, tokenizer)
    actual_block = min(block_size, int(DEFAULT_CONFIG["block_size"]))
    if len(train_stream) <= actual_block + 1:
        raise ValueError("Training corpus is too small for the selected context length.")

    if resume and resume_checkpoint is not None:
        config = resume_checkpoint.get("config")
        if not isinstance(config, dict) or int(config.get("block_size", 0)) != actual_block:
            raise ValueError("The requested block size differs from the checkpoint. Use the original --block-size.")
        prior_metrics = resume_checkpoint.get("metrics", {})
        prior_vocab_target = prior_metrics.get("requested_vocab_size")
        if prior_vocab_target is not None and int(prior_vocab_target) != target_vocab_size:
            raise ValueError("The requested vocabulary size differs from the checkpoint.")
    else:
        config = {**DEFAULT_CONFIG, "block_size": actual_block}

    device = _preferred_device()
    model = FrossTransformer(tokenizer.vocab_size, config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.08)
    start_step = 0
    last_loss = float("nan")
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_step = 0
    best_train_loss = float("nan")

    if resume and resume_checkpoint is not None and resume_state is not None:
        prior_metrics = resume_checkpoint.get("metrics", {})
        start_step = int(resume_state.get("completed_steps", 0))
        if start_step < 1:
            raise ValueError("Checkpoint has an invalid completed-step count.")
        if steps <= start_step:
            raise ValueError(
                f"Checkpoint already has {start_step} steps; --steps must be a larger cumulative target."
            )
        if int(prior_metrics.get("training_steps", -1)) != start_step:
            raise ValueError("Checkpoint metrics and optimizer state have different step counts.")
        model.load_state_dict(resume_state["last_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        random.setstate(resume_state["python_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        saved_val = prior_metrics.get("validation_loss")
        if saved_val is not None and math.isfinite(float(saved_val)):
            best_validation_loss = float(saved_val)
        best_checkpoint_step = int(prior_metrics.get("best_checkpoint_step", start_step))
        saved_train_loss = prior_metrics.get("train_loss")
        if saved_train_loss is not None:
            best_train_loss = float(saved_train_loss)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in resume_checkpoint["state_dict"].items()
        }

    last_start = len(train_stream) - actual_block - 1
    evaluation_interval = max(50, min(300, schedule_total // 8))
    for step in range(start_step, steps):
        starts = [random.randint(0, last_start) for _ in range(batch_size)]
        x = torch.tensor([train_stream[i:i + actual_block] for i in starts], dtype=torch.long, device=device)
        y = torch.tensor([train_stream[i + 1:i + actual_block + 1] for i in starts], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = 3e-4 * (1.0 - 0.65 * (step / max(1, schedule_total - 1)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        last_loss = float(loss.detach())
        is_eval_step = (step + 1) % evaluation_interval == 0 or step + 1 == steps
        if is_eval_step:
            candidate_validation_loss = _evaluate(model, validation_stream, actual_block)
            if math.isfinite(candidate_validation_loss) and candidate_validation_loss < best_validation_loss:
                best_validation_loss = candidate_validation_loss
                best_checkpoint_step = step + 1
                best_train_loss = last_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        if (step + 1) % max(1, (steps - start_step) // 5) == 0 or step + 1 == steps:
            shown_val = (
                f"{best_validation_loss:.4f}" if math.isfinite(best_validation_loss) else "not-yet-measured"
            )
            print(
                f"step {step + 1}/{steps} · train loss {last_loss:.4f} · "
                f"best validation loss {shown_val}",
                flush=True,
            )

    # Keep the true last training state separately from the best-validation inference weights.
    last_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    def _cpu_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: _cpu_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_cpu_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_cpu_tree(item) for item in value)
        return value

    saved_optimizer_state = _cpu_tree(optimizer.state_dict())
    saved_python_rng_state = random.getstate()
    saved_torch_rng_state = torch.get_rng_state().clone()

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        validation_loss = best_validation_loss
        reported_train_loss = best_train_loss
    else:
        validation_loss = _evaluate(model, validation_stream, actual_block)
        reported_train_loss = last_loss
    model.eval()
    checkpoint = {
        "format_version": 2,
        "version": VERSION,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "state_dict": _cpu_tree(model.state_dict()),
        "resume_state": {
            "last_state_dict": last_state_dict,
            "optimizer_state_dict": saved_optimizer_state,
            "python_rng_state": saved_python_rng_state,
            "torch_rng_state": saved_torch_rng_state,
            "completed_steps": steps,
            "corpus_sha256": corpus_sha256,
            "seed": seed,
            "schedule_steps": schedule_total,
        },
        "metrics": {
            "train_loss": round(reported_train_loss, 5) if math.isfinite(reported_train_loss) else None,
            "validation_loss": round(validation_loss, 5) if math.isfinite(validation_loss) else None,
            "training_device": str(device),
            "training_examples": len(train_records),
            "validation_examples": len(validation_records),
            "training_steps": steps,
            "steps_this_run": steps - start_step,
            "schedule_steps": schedule_total,
            "best_checkpoint_step": best_checkpoint_step or steps,
            "vocab_size": tokenizer.vocab_size,
            "requested_vocab_size": target_vocab_size if merge_count is None else None,
            "vocab_target_reached": (
                tokenizer.vocab_size == target_vocab_size if merge_count is None else None
            ),
            "tokenizer_type": ("sentencepiece-unigram" if isinstance(tokenizer, SentencePieceUnigramTokenizer) else ("word-boundary-bpe" if tokenizer.pretokenized else "legacy-bpe")),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trained_from_random_weights": True,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return {
        "ok": True, "version": VERSION, "output": str(target),
        **checkpoint["metrics"]
    }


def load_model(path: str | Path = DEFAULT_CHECKPOINT) -> tuple[FrossTransformer | None, BPETokenizer | None, dict[str, Any]]:
    configure_runtime_threads()
    target = Path(path)
    if not target.is_file():
        return None, None, {"loaded": False, "reason": "checkpoint-missing", "path": str(target)}
    try:
        checkpoint = torch.load(target, map_location="cpu", weights_only=True)
        if int(checkpoint.get("format_version", 0)) not in {1, 2}:
            raise ValueError("Unsupported checkpoint format.")
        tokenizer = _load_tokenizer(checkpoint["tokenizer"])
        config = checkpoint["config"]
        model = FrossTransformer(tokenizer.vocab_size, config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        device = _preferred_device()
        model.to(device)
        model.eval()
        return model, tokenizer, {
            "loaded": True,
            "version": checkpoint.get("version", VERSION),
            "path": str(target),
            "device": str(device),
            **checkpoint.get("metrics", {}),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as exc:
        return None, None, {"loaded": False, "reason": type(exc).__name__, "path": str(target)}


def export_onnx_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export the exact trained weights to a fixed-shape ONNX graph for the C++ runtime.

    The graph always receives block_size token IDs. Positions after the real sequence
    are right-padded with PAD_ID; causal attention guarantees they cannot affect the
    logits at the final real token.
    """
    model, tokenizer, status = load_model(checkpoint_path)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Cannot export an unavailable model: {status}")
    target = Path(output_path) if output_path else Path(checkpoint_path).with_suffix(".onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    block_size = int(model.config["block_size"])
    example = torch.full((1, block_size), PAD_ID, dtype=torch.long)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example,),
            str(target),
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    try:
        import onnx
        graph = onnx.load(str(target))
        # Keep ONNX IR metadata on a widely supported version; this graph uses no newer IR features.
        graph.ir_version = min(int(graph.ir_version), 10)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(target))
    except ImportError as exc:
        raise RuntimeError("The 'onnx' package is required to export the C++ inference graph.") from exc
    return {
        "ok": True,
        "path": str(target),
        "version": VERSION,
        "block_size": block_size,
        "vocab_size": tokenizer.vocab_size,
        "onnx_ir_version": int(graph.ir_version),
    }



def _expand_bare_topic(message: str) -> str:
    """Make a one-word topic fit the explanatory question style used in training."""
    original = str(message or "").strip()
    normalized = re.sub(r"[.!?]+$", "", original.lower().replace("ё", "е")).strip()
    if not normalized or len(normalized.split()) != 1:
        return original
    excluded = {
        "да", "нет", "ага", "угу", "ок", "окей", "ладно", "хорошо", "хм",
        "привет", "приветик", "здравствуй", "здравствуйте", "спасибо", "благодарю",
        "пока", "стоп", "отмена", "помоги", "помогите", "сделай", "создай",
        "напиши", "исправь", "удали", "добавь", "продолжи", "продолжай",
        "переведи", "перевод", "почему", "зачем", "как", "что", "кто", "где",
        "когда", "куда", "ты", "вы", "я", "мы", "он", "она", "они", "это",
        "можно", "нельзя", "код", "тест",
    }
    if normalized in excluded or len(normalized) > 40:
        return original
    if not re.fullmatch(r"[a-zа-яё][a-zа-яё0-9_+#.-]*", normalized):
        return original
    return f"Объясни простыми словами, что такое {original.rstrip('.,!?')}."



def _social_reply(message: str) -> str | None:
    """Handle only basic social pleasantries deterministically; real questions use the model."""
    normalized = str(message or "").lower().replace("ё", "е").strip()
    normalized = re.sub(r"[!?,.;:]+", " ", normalized)
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"\s+(фросс|fross)$", "", normalized)
    if normalized in {"привет", "приветик", "здравствуй", "здравствуйте", "доброе утро", "добрый день", "добрый вечер", "hello", "hi"}:
        return "Привет! Я на связи. Что хочешь обсудить?"
    if normalized in {"спасибо", "спасибо большое", "благодарю", "thanks", "thank you"}:
        return "Пожалуйста! Чем ещё помочь?"
    if normalized in {"пока", "до встречи", "до свидания", "bye", "goodbye"}:
        return "Пока! Удачи!"
    return None


@torch.inference_mode()
def generate_reply(
    model: FrossTransformer,
    tokenizer: BPETokenizer,
    message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    knowledge: list[dict[str, Any]] | None = None,
    mode: str = "normal",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int = 12,
    native_runtime: Any = None,
) -> str:
    if not str(message).strip():
        raise ValueError("Message must not be empty.")
    social = _social_reply(message)
    if social is not None:
        return social
    deep = str(mode).lower() in {"reasoning", "deep"}

    # Keep the model input close to the exact sequence format used during training.
    # Long workflow instructions previously displaced useful user text from the
    # 320-token context window even though the model had never trained on those prompts.
    context_parts: list[str] = []
    for item in (knowledge if isinstance(knowledge, list) else [])[:2]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", item.get("topic", ""))).strip()[:90]
        summary = str(item.get("summary", item.get("snippet", ""))).strip()[:180]
        if title and summary:
            context_parts.append(f"Знание: {title}. {summary}")
    for item in (history if isinstance(history, list) else [])[-4:]:
        if not isinstance(item, dict):
            continue
        role = "FROSS" if item.get("role") == "assistant" else "Пользователь"
        content = " ".join(str(item.get("content", "")).split())[:240]
        if content:
            context_parts.append(f"{role}: {content}")

    block_size = int(model.config["block_size"])
    prompt_budget = max(1, block_size - 1)
    latest_prompt_tokens = tokenizer.encode(_format_generation_prefix(_expand_bare_topic(message)))
    if len(latest_prompt_tokens) > prompt_budget:
        # Preserve the beginning and ending (including the FROSS response prefix).
        prefix = tokenizer.encode("Пользователь: ")
        suffix = tokenizer.encode("\nFROSS: ")
        separator = tokenizer.encode(" … ")
        available = max(1, prompt_budget - len(prefix) - len(suffix) - len(separator))
        body = tokenizer.encode(str(message).strip())
        head_count = available // 2
        tail_count = available - head_count
        latest_prompt_tokens = prefix + body[:head_count] + separator + body[-tail_count:] + suffix
        latest_prompt_tokens = latest_prompt_tokens[-prompt_budget:]
    remaining = max(0, prompt_budget - len(latest_prompt_tokens))
    context_text = "\n".join(context_parts)
    context_tokens = tokenizer.encode(context_text + "\n") if context_text else []
    # The latest user message has priority; trim old context first.
    prompt_tokens = context_tokens[-remaining:] + latest_prompt_tokens if remaining else latest_prompt_tokens
    token_ids = ([BOS_ID] + prompt_tokens)[-block_size:]
    max_tokens = max_new_tokens if max_new_tokens is not None else MAX_GENERATION_TOKENS
    temp = float(temperature if temperature is not None else (0.30 if deep else 0.34))
    temp = max(0.1, min(1.5, temp))
    generated: list[int] = []
    recent_generated: list[int] = []
    for _ in range(max(1, min(int(max_tokens), MAX_GENERATION_TOKENS))):
        device = next(model.parameters()).device
        inputs = torch.tensor([token_ids[-block_size:]], dtype=torch.long, device=device)
        if native_runtime is None:
            logits = model(inputs)[0, -1, :].float()
        else:
            logits = torch.tensor(
                native_runtime.forward(token_ids[-block_size:]),
                dtype=torch.float32,
                device=device,
            )
        logits[PAD_ID] = float("-inf")
        logits[UNK_ID] = float("-inf")
        logits[BOS_ID] = float("-inf")
        # Avoid a zero-length answer: the first few tokens should establish a
        # continuation before EOS is allowed to end sampling.
        if len(generated) < 8:
            logits[EOS_ID] = float("-inf")
        logits = logits / temp
        if recent_generated:
            seen = set(recent_generated[-64:])
            for token_id in seen:
                if 4 <= token_id < logits.numel():
                    logits[token_id] -= 0.45
        if top_k > 0 and top_k < logits.numel():
            cutoff = torch.topk(logits, top_k).values[-1]
            logits[logits < cutoff] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            break
        next_id = int(torch.multinomial(probabilities, num_samples=1))
        if next_id == EOS_ID:
            break
        token_ids.append(next_id)
        generated.append(next_id)
        recent_generated.append(next_id)
        decoded = tokenizer.decode(generated)
        if "\nПользователь:" in decoded:
            decoded = decoded.split("\nПользователь:", 1)[0]
            return _clean_generated_text(decoded)
    answer = tokenizer.decode(generated).split("\nПользователь:", 1)[0]
    return _clean_generated_text(answer)


def _clean_generated_text(text: str) -> str:
    """Keep paragraph/code formatting while removing runaway blank lines."""
    answer = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("command", choices=("train", "test", "generate", "export"))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=600, help="Cumulative training-step target; use 5000 to continue a 600-step checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Continue from the last saved training state, including AdamW and RNG states.")
    parser.add_argument("--schedule-steps", type=int, default=None, help="Total learning-rate schedule horizon. Keep this at 5000 for both training stages.")
    parser.add_argument("--seed", type=int, default=1337, help="Training seed; it must match when resuming.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=320)
    parser.add_argument(
        "--vocab-size", type=int, default=20000,
        help="Target SentencePiece Unigram vocabulary size; actual size may be lower if the corpus is too small.",
    )
    parser.add_argument(
        "--merges", type=int, default=None,
        help="Legacy override: train a fixed number of BPE merges instead of targeting vocab size.",
    )
    parser.add_argument("--text", default="Объясни, как обучается нейросеть.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--onnx-output", default=None)
    args = parser.parse_args()
    if args.command == "export":
        try:
            result = export_onnx_model(args.output, args.onnx_output)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command in {"train", "test"}:
        metrics = train_model(
            args.corpus,
            args.output,
            steps=args.steps if args.command == "train" else min(args.steps, 12),
            batch_size=args.batch_size,
            block_size=args.block_size,
            merge_count=args.merges,
            target_vocab_size=args.vocab_size,
            seed=args.seed,
            resume=args.resume,
            schedule_steps=args.schedule_steps,
        )
        model, tokenizer, status = load_model(args.output)
        if model is None or tokenizer is None:
            print(json.dumps({"ok": False, "error": status}, ensure_ascii=False, indent=2))
            return 1
        sample = generate_reply(model, tokenizer, args.text, max_new_tokens=24)
        result = {**metrics, "model_loaded": True, "generation_test": bool(sample), "sample": sample}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Training success means the checkpoint was created and can be loaded.
        # A tiny/early model can still fail a language-generation smoke sample;
        # that must not invalidate a completed training/export build.
        if args.command == "train":
            return 0
        return 0 if sample else 1
    model, tokenizer, status = load_model(args.output)
    if model is None or tokenizer is None:
        print(json.dumps({"ok": False, "status": status}, ensure_ascii=False, indent=2))
        return 1
    print(generate_reply(model, tokenizer, args.text, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
