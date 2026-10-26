FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1
WORKDIR /app

# Python is used for model training and orchestration. C++/ONNX Runtime executes inference.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl g++ \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir "onnx>=1.16,<2"

ARG ONNXRUNTIME_VERSION=1.31.0
RUN mkdir -p /opt/onnxruntime \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
       | tar -xz --strip-components=1 -C /opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

COPY fross_transformer.py /app/fross_transformer.py
COPY data/fross_corpus.jsonl /app/data/fross_corpus.jsonl
COPY data/fross_teacher_corpus.jsonl /app/data/fross_teacher_corpus.jsonl
COPY training/distillation/prepare_corpus.py /app/training/distillation/prepare_corpus.py
COPY services/python-brain/server.py /app/server.py
COPY services/python-brain/native_runtime.py /app/native_runtime.py
COPY services/python-brain/cpp_runtime.cpp /app/cpp_runtime.cpp

RUN g++ -std=c++17 -O3 -fPIC -shared /app/cpp_runtime.cpp \
      -I/opt/onnxruntime/include -L/opt/onnxruntime/lib -lonnxruntime \
      -Wl,-rpath,/opt/onnxruntime/lib -o /app/libfross_cpp.so

# Train the FROSS student from random initialization on validated original and
# teacher-generated text examples. Only text data is transferred: no teacher weights
# are imported, and the teacher is not called during this build or during inference.
# The default is 3000 cumulative training steps; override with a build argument if needed.
ARG FROSS_TRAIN_STEPS=3000
ARG FROSS_VOCAB_SIZE=5000
ARG FROSS_TRAIN_THREADS=2

# Prepare the corpus in its own layer, so a retry can reuse it instead of repeating work.
RUN python /app/training/distillation/prepare_corpus.py \
       --base /app/data/fross_corpus.jsonl \
       --teacher /app/data/fross_teacher_corpus.jsonl \
       --output /app/data/fross_training_corpus.jsonl

# For the production target, save an intermediate 600-step checkpoint with a 3000-step
# learning-rate schedule, then continue from its saved model, optimizer, and RNG state.
# Small CI smoke builds (for example FROSS_TRAIN_STEPS=2) train directly in one pass.
RUN if [ "${FROSS_TRAIN_STEPS}" -gt 600 ]; then \
      FROSS_TORCH_THREADS="${FROSS_TRAIN_THREADS}" OMP_NUM_THREADS="${FROSS_TRAIN_THREADS}" MKL_NUM_THREADS="${FROSS_TRAIN_THREADS}" \
      python /app/fross_transformer.py train --corpus /app/data/fross_training_corpus.jsonl \
        --output /app/fross-transformer.pt --steps 600 --schedule-steps "${FROSS_TRAIN_STEPS}" \
        --batch-size 4 --block-size 320 --vocab-size "${FROSS_VOCAB_SIZE}"; \
    else \
      FROSS_TORCH_THREADS="${FROSS_TRAIN_THREADS}" OMP_NUM_THREADS="${FROSS_TRAIN_THREADS}" MKL_NUM_THREADS="${FROSS_TRAIN_THREADS}" \
      python /app/fross_transformer.py train --corpus /app/data/fross_training_corpus.jsonl \
        --output /app/fross-transformer.pt --steps "${FROSS_TRAIN_STEPS}" --schedule-steps "${FROSS_TRAIN_STEPS}" \
        --batch-size 4 --block-size 320 --vocab-size "${FROSS_VOCAB_SIZE}"; \
    fi

RUN if [ "${FROSS_TRAIN_STEPS}" -gt 600 ]; then \
      FROSS_TORCH_THREADS="${FROSS_TRAIN_THREADS}" OMP_NUM_THREADS="${FROSS_TRAIN_THREADS}" MKL_NUM_THREADS="${FROSS_TRAIN_THREADS}" \
      python /app/fross_transformer.py train --corpus /app/data/fross_training_corpus.jsonl \
        --output /app/fross-transformer.pt --steps "${FROSS_TRAIN_STEPS}" --resume \
        --schedule-steps "${FROSS_TRAIN_STEPS}" --batch-size 4 --block-size 320 \
        --vocab-size "${FROSS_VOCAB_SIZE}"; \
    else \
      echo "Target is ${FROSS_TRAIN_STEPS} steps; staged resume is not needed."; \
    fi

RUN python /app/fross_transformer.py export --output /app/fross-transformer.pt --onnx-output /app/fross-transformer.onnx \
    && python -c "import torch; c=torch.load('/app/fross-transformer.pt', map_location='cpu', weights_only=True); m=c['metrics']; assert m['vocab_size']==${FROSS_VOCAB_SIZE}, m; assert m['training_steps']==${FROSS_TRAIN_STEPS}, m; assert m['vocab_target_reached'], m; print('Brain 1.4 checkpoint verified:', m['vocab_size'], 'tokens,', m['parameters'], 'parameters,', m['training_steps'], 'steps')" \
    && python -c "import onnx; onnx.checker.check_model(onnx.load('/app/fross-transformer.onnx'))" \
    && python -c "import math; import fross_transformer as t; from native_runtime import NativeRuntime; m, tok, status=t.load_model('/app/fross-transformer.pt'); assert m is not None and tok is not None, status; r=NativeRuntime('/app/libfross_cpp.so','/app/fross-transformer.onnx',block_size=m.config['block_size'],vocab_size=tok.vocab_size); logits=r.forward(tok.encode('Привет!')); assert len(logits)==tok.vocab_size and all(math.isfinite(x) for x in logits); print('C++/ONNX smoke test passed:', len(logits), 'logits'); r.close()"
ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
ENV FROSS_ONNX_MODEL_PATH=/app/fross-transformer.onnx
ENV FROSS_CPP_RUNTIME_LIBRARY=/app/libfross_cpp.so
ENV FROSS_CPP_RUNTIME_ENABLED=1
EXPOSE 10000
CMD ["python", "/app/server.py"]
