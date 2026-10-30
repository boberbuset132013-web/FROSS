FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1 FROSS_MODEL_GITHUB_REPOSITORY=boberbuset132013-web/FROSS
WORKDIR /app

# Temporary troubleshooting build: run inference through PyTorch only.
# Keep the existing FROSS 1.5 checkpoint unchanged; training is offline.
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --retries 5 --timeout 120 --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir --retries 5 --timeout 120 "onnx>=1.16,<2" "sentencepiece>=0.2.0,<0.3"

COPY fross_transformer.py /app/fross_transformer.py
COPY data/ /app/data/
COPY training/distillation/verify_checkpoint.py /app/training/distillation/verify_checkpoint.py
COPY services/python-brain/server.py /app/server.py
COPY services/python-brain/download_model.py /app/download_model.py

ARG FROSS_BUILD_BUNDLE_MODEL=0
ARG FROSS_MODEL_URL=https://disk.yandex.ru/d/Ybh-XEUk7Xwt9w
ARG FROSS_REQUIRED_MODEL_STEPS=3000
ARG FROSS_VOCAB_SIZE=5000

# Normal Hostless builds skip checkpoint download. The running service loads the
# verified checkpoint from GitHub Releases after the HTTP listener starts.
# Keep runtime model files in /tmp because Hostless mounts /app read-only.
RUN set -eu; if [ "${FROSS_BUILD_BUNDLE_MODEL}" = "1" ]; then \
    echo "Bundling a pre-trained FROSS checkpoint for CI smoke testing..."; \
    python /app/download_model.py "${FROSS_MODEL_URL}" /tmp/fross-transformer.download; \
    mv /tmp/fross-transformer.download /tmp/fross-transformer.pt; \
    printf 'uploaded-checkpoint\n' > /app/fross-model-source.txt; \
    python /app/training/distillation/verify_checkpoint.py \
      --checkpoint /tmp/fross-transformer.pt \
      --model-source /app/fross-model-source.txt \
      --required-model-steps "${FROSS_REQUIRED_MODEL_STEPS}" \
      --train-steps 600 \
      --vocab-size "${FROSS_VOCAB_SIZE}"; \
  else \
    echo "Skipping model download during build; upload it from FROSS Settings."; \
  fi

ENV FROSS_TRANSFORMER_MODEL_PATH=/tmp/fross-transformer.pt
ENV FROSS_ONNX_MODEL_PATH=/tmp/fross-transformer.onnx
ENV FROSS_CPP_RUNTIME_ENABLED=0
EXPOSE 10000
CMD ["python", "/app/server.py"]
