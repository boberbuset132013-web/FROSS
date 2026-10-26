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
COPY services/python-brain/server.py /app/server.py
COPY services/python-brain/native_runtime.py /app/native_runtime.py
COPY services/python-brain/cpp_runtime.cpp /app/cpp_runtime.cpp

RUN g++ -std=c++17 -O3 -fPIC -shared /app/cpp_runtime.cpp \
      -I/opt/onnxruntime/include -L/opt/onnxruntime/lib -lonnxruntime \
      -Wl,-rpath,/opt/onnxruntime/lib -o /app/libfross_cpp.so

# Weights are trained from random initialization on the local corpus, then exported
# to ONNX. No pretrained model or hosted LLM API is downloaded or called.
ARG FROSS_TRAIN_STEPS=3000
RUN python /app/fross_transformer.py train --corpus /app/data/fross_corpus.jsonl \
       --output /app/fross-transformer.pt --steps ${FROSS_TRAIN_STEPS} --batch-size 4 --block-size 320 --merges 384 \
    && python /app/fross_transformer.py export --output /app/fross-transformer.pt --onnx-output /app/fross-transformer.onnx \
    && test -n "$(python /app/fross_transformer.py generate --output /app/fross-transformer.pt --text 'Как работает attention?')" \
    && python -c "import onnx; onnx.checker.check_model(onnx.load('/app/fross-transformer.onnx'))"
ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
ENV FROSS_ONNX_MODEL_PATH=/app/fross-transformer.onnx
ENV FROSS_CPP_RUNTIME_LIBRARY=/app/libfross_cpp.so
ENV FROSS_CPP_RUNTIME_ENABLED=1
EXPOSE 10000
CMD ["python", "/app/server.py"]
