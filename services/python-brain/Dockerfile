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
COPY training/distillation/verify_checkpoint.py /app/training/distillation/verify_checkpoint.py
COPY services/python-brain/server.py /app/server.py
COPY services/python-brain/native_runtime.py /app/native_runtime.py
COPY services/python-brain/cpp_runtime.cpp /app/cpp_runtime.cpp

RUN g++ -std=c++17 -O3 -fPIC -shared /app/cpp_runtime.cpp \
      -I/opt/onnxruntime/include -L/opt/onnxruntime/lib -lonnxruntime \
      -Wl,-rpath,/opt/onnxruntime/lib -o /app/libfross_cpp.so

# Train the FROSS student from random initialization on validated original and
# teacher-generated text examples. Only text data is transferred: no teacher weights
# are imported, and the teacher is not called during this build or during inference.
# Prefer the hosted checkpoint when it is a compatible FROSS 1.4 model.
# If download/validation fails, the image still gets a bounded 600-step model.
ARG FROSS_MODEL_URL=https://deliverbit.com/f/uMSeXVFA5w/fross-transformer
# Accept a compatible uploaded Brain 1.4 checkpoint from 600 steps upward to avoid
# retraining during every deploy. The /health endpoint always reports its real step count.
ARG FROSS_REQUIRED_MODEL_STEPS=600
ARG FROSS_TRAIN_STEPS=600
ARG FROSS_VOCAB_SIZE=5000
ARG FROSS_TRAIN_THREADS=2

RUN python /app/training/distillation/prepare_corpus.py \
       --base /app/data/fross_corpus.jsonl \
       --teacher /app/data/fross_teacher_corpus.jsonl \
       --output /app/data/fross_training_corpus.jsonl

RUN if [ -n "${FROSS_MODEL_URL}" ]; then \
      echo "Attempting to download the hosted FROSS checkpoint..."; \
      if curl -fL --show-error --progress-bar --connect-timeout 8 --max-time 60 "${FROSS_MODEL_URL}" -o /app/fross-transformer.download; then \
        mv /app/fross-transformer.download /app/fross-transformer.pt; \
        if python -c "import torch, fross_transformer as t; c=torch.load('/app/fross-transformer.pt',map_location='cpu',weights_only=True); m=c.get('metrics',{}); assert c.get('version') == t.VERSION,c.get('version'); assert c.get('format_version') in (1,2); assert int(m.get('vocab_size',0)) == ${FROSS_VOCAB_SIZE},m; assert m.get('vocab_target_reached') is True,m; assert int(m.get('training_steps',0)) >= ${FROSS_REQUIRED_MODEL_STEPS},m; assert int(c.get('config',{}).get('block_size',0)) == 320; model,tok,status=t.load_model('/app/fross-transformer.pt'); assert model is not None and tok is not None,status; print('Compatible hosted checkpoint:',m.get('training_steps'),'steps')"; then \
          printf 'uploaded-checkpoint\n' > /app/fross-model-source.txt; \
        else \
          rm -f /app/fross-transformer.pt; \
          echo "Hosted checkpoint is incompatible or has fewer than ${FROSS_REQUIRED_MODEL_STEPS} steps; falling back to repository training."; \
        fi; \
      else \
        rm -f /app/fross-transformer.download; \
        echo "Hosted checkpoint download failed; falling back to repository training."; \
      fi; \
    else \
      echo "FROSS_MODEL_URL is empty; training from repository corpus."; \
    fi

# 3,000-step training remains available as a build argument, but is not forced
# into every deployment. For targets >600 the first stage is resumable.
RUN if [ -f /app/fross-model-source.txt ]; then \
      echo "A validated hosted checkpoint is ready; skipping retraining."; \
    elif [ "${FROSS_TRAIN_STEPS}" -gt 600 ]; then \
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

RUN if [ -f /app/fross-model-source.txt ]; then \
      echo "Hosted checkpoint already contains the trained weights; skipping resume."; \
    elif [ "${FROSS_TRAIN_STEPS}" -gt 600 ]; then \
      FROSS_TORCH_THREADS="${FROSS_TRAIN_THREADS}" OMP_NUM_THREADS="${FROSS_TRAIN_THREADS}" MKL_NUM_THREADS="${FROSS_TRAIN_THREADS}" \
      python /app/fross_transformer.py train --corpus /app/data/fross_training_corpus.jsonl \
        --output /app/fross-transformer.pt --steps "${FROSS_TRAIN_STEPS}" --resume \
        --schedule-steps "${FROSS_TRAIN_STEPS}" --batch-size 4 --block-size 320 \
        --vocab-size "${FROSS_VOCAB_SIZE}"; \
    else \
      echo "Target is ${FROSS_TRAIN_STEPS} steps; staged resume is not needed."; \
    fi

RUN if [ ! -f /app/fross-model-source.txt ]; then \
      printf 'repository-training steps=%s\n' "${FROSS_TRAIN_STEPS}" > /app/fross-model-source.txt; \
    fi

RUN python /app/training/distillation/verify_checkpoint.py \
      --checkpoint /app/fross-transformer.pt \
      --model-source /app/fross-model-source.txt \
      --required-model-steps "${FROSS_REQUIRED_MODEL_STEPS}" \
      --train-steps "${FROSS_TRAIN_STEPS}" \
      --vocab-size "${FROSS_VOCAB_SIZE}" \
    && python /app/fross_transformer.py export --output /app/fross-transformer.pt --onnx-output /app/fross-transformer.onnx \
    && python -c "import onnx; onnx.checker.check_model(onnx.load('/app/fross-transformer.onnx'))" \
    && python -c "import math; import fross_transformer as t; from native_runtime import NativeRuntime; m, tok, status=t.load_model('/app/fross-transformer.pt'); assert m is not None and tok is not None, status; r=NativeRuntime('/app/libfross_cpp.so','/app/fross-transformer.onnx',block_size=m.config['block_size'],vocab_size=tok.vocab_size); logits=r.forward(tok.encode('Привет!')); assert len(logits)==tok.vocab_size and all(math.isfinite(x) for x in logits); print('C++/ONNX smoke test passed:', len(logits), 'logits'); r.close()"

ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
ENV FROSS_ONNX_MODEL_PATH=/app/fross-transformer.onnx
ENV FROSS_CPP_RUNTIME_LIBRARY=/app/libfross_cpp.so
ENV FROSS_CPP_RUNTIME_ENABLED=1
EXPOSE 10000
CMD ["python", "/app/server.py"]
