"""ctypes bridge to the small C++ ONNX Runtime inference engine."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable


class NativeRuntime:
    """Load the exported ONNX graph once and expose last-token logits to Python."""

    def __init__(self, library_path: str, model_path: str, *, block_size: int, vocab_size: int):
        self.library_path = str(Path(library_path))
        self.model_path = str(Path(model_path))
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        if self.block_size < 1 or self.vocab_size < 1:
            raise ValueError("block_size and vocab_size must be positive")
        if not Path(self.library_path).is_file():
            raise FileNotFoundError(f"C++ runtime library not found: {self.library_path}")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._library = ctypes.CDLL(self.library_path)
        char_ptr = ctypes.POINTER(ctypes.c_char)
        self._library.fross_runtime_create.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, char_ptr, ctypes.c_size_t
        ]
        self._library.fross_runtime_create.restype = ctypes.c_void_p
        self._library.fross_runtime_destroy.argtypes = [ctypes.c_void_p]
        self._library.fross_runtime_destroy.restype = None
        self._library.fross_runtime_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            char_ptr,
            ctypes.c_size_t,
        ]
        self._library.fross_runtime_forward.restype = ctypes.c_int

        error = ctypes.create_string_buffer(1024)
        self._handle = self._library.fross_runtime_create(
            self.model_path.encode("utf-8"),
            self.block_size,
            self.vocab_size,
            error,
            len(error),
        )
        if not self._handle:
            detail = error.value.decode("utf-8", errors="replace") or "unknown C++ runtime error"
            raise RuntimeError(detail)

    def forward(self, token_ids: Iterable[int]) -> list[float]:
        tokens = [int(token) for token in token_ids][-self.block_size:]
        if not tokens:
            raise ValueError("at least one token is required")
        padded = tokens + [0] * (self.block_size - len(tokens))
        input_array = (ctypes.c_int64 * self.block_size)(*padded)
        output_array = (ctypes.c_float * self.vocab_size)()
        error = ctypes.create_string_buffer(1024)
        result = self._library.fross_runtime_forward(
            self._handle,
            input_array,
            len(tokens),
            output_array,
            self.vocab_size,
            error,
            len(error),
        )
        if result < 0:
            detail = error.value.decode("utf-8", errors="replace") or "unknown C++ inference error"
            raise RuntimeError(detail)
        if result != self.vocab_size:
            raise RuntimeError(f"C++ runtime returned {result} logits, expected {self.vocab_size}")
        return list(output_array)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.fross_runtime_destroy(handle)
            self._handle = None

    def __enter__(self) -> "NativeRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
