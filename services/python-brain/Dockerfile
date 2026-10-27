FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1 FROSS_MODEL_GITHUB_REPOSITORY=boberbuset132013-web/FROSS
WORKDIR /app

# This image must never train a model. Training is offline; normal deployments
# start without bundling a checkpoint so an upload from Settings can bootstrap the service.
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates curl g++ \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --retries 5 --timeout 120 --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir --retries 5 --timeout 120 "onnx>=1.16,<2"

ARG ONNXRUNTIME_VERSION=1.31.0
RUN mkdir -p /opt/onnxruntime \
    && curl --fail --location --show-error --silent --retry 5 --retry-all-errors --connect-timeout 30 --max-time 300 \
       "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
       --output /tmp/onnxruntime.tgz \
    && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

COPY fross_transformer.py /app/fross_transformer.py
COPY training/distillation/verify_checkpoint.py /app/training/distillation/verify_checkpoint.py
COPY services/python-brain/server.py /app/server.py
COPY services/python-brain/download_model.py /app/download_model.py
COPY services/python-brain/native_runtime.py /app/native_runtime.py
COPY services/python-brain/cpp_runtime.cpp /app/cpp_runtime.cpp

RUN g++ -std=c++17 -O3 -fPIC -shared /app/cpp_runtime.cpp \
      -I/opt/onnxruntime/include -L/opt/onnxruntime/lib -lonnxruntime \
      -Wl,-rpath,/opt/onnxruntime/lib -o /app/libfross_cpp.so

ARG FROSS_BUILD_BUNDLE_MODEL=0
ARG FROSS_MODEL_URL=https://disk.yandex.ru/d/Ybh-XEUk7Xwt9w
ARG FROSS_REQUIRED_MODEL_STEPS=3000
ARG FROSS_VOCAB_SIZE=5000

# Normal Hostless builds do not download a checkpoint or spend time exporting ONNX.
# The running service accepts a model from Settings, saves it as a GitHub Release Asset,
# and reloads that same asset after future restarts. CI explicitly bundles a known
# checkpoint to exercise full production inference in the smoke test.
RUN set -eu; if [ "${FROSS_BUILD_BUNDLE_MODEL}" = "1" ]; then \
    echo "Bundling a pre-trained FROSS checkpoint for CI smoke testing..."; \
    python /app/download_model.py "${FROSS_MODEL_URL}" /app/fross-transformer.download; \
    mv /app/fross-transformer.download /app/fross-transformer.pt; \
    printf 'uploaded-checkpoint\n' > /app/fross-model-source.txt; \
    python /app/training/distillation/verify_checkpoint.py \
      --checkpoint /app/fross-transformer.pt \
      --model-source /app/fross-model-source.txt \
      --required-model-steps "${FROSS_REQUIRED_MODEL_STEPS}" \
      --train-steps 600 \
      --vocab-size "${FROSS_VOCAB_SIZE}"; \
    python /app/fross_transformer.py export \
      --output /app/fross-transformer.pt \
      --onnx-output /app/fross-transformer.onnx; \
    python -c "import onnx; onnx.checker.check_model(onnx.load('/app/fross-transformer.onnx'))"; \
    python -c "import math; import fross_transformer as t; from native_runtime import NativeRuntime; m, tok, status=t.load_model('/app/fross-transformer.pt'); assert m is not None and tok is not None, status; r=NativeRuntime('/app/libfross_cpp.so','/app/fross-transformer.onnx',block_size=m.config['block_size'],vocab_size=tok.vocab_size); logits=r.forward(tok.encode('Привет!')); assert len(logits)==tok.vocab_size and all(math.isfinite(x) for x in logits); print('C++/ONNX smoke test passed:', len(logits), 'logits'); r.close()"; \
  else \
    echo "Skipping model download during build; upload it from FROSS Settings."; \
  fi

ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
ENV FROSS_ONNX_MODEL_PATH=/app/fross-transformer.onnx
ENV FROSS_CPP_RUNTIME_LIBRARY=/app/libfross_cpp.so
ENV FROSS_CPP_RUNTIME_ENABLED=1
EXPOSE 10000
CMD ["python", "/app/server.py"]
