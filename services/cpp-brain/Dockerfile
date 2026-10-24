FROM gcc:14-bookworm AS builder
WORKDIR /src
COPY services/cpp-brain/main.cpp /src/main.cpp
RUN g++ -std=c++17 -O2 -pipe -Wall -Wextra -pedantic /src/main.cpp -o /fross-cpp-brain

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1
ENV FROSS_CPP_EXECUTABLE=/app/fross-cpp-brain
WORKDIR /app
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY --from=builder /fross-cpp-brain /app/fross-cpp-brain
COPY fross_transformer.py /app/fross_transformer.py
COPY data/fross_corpus.jsonl /app/data/fross_corpus.jsonl
COPY services/cpp-brain/server.py /app/server.py
# This service trains its own copy from random weights; nothing is downloaded as a model.
RUN python /app/fross_transformer.py train --corpus /app/data/fross_corpus.jsonl \
    --output /app/fross-transformer.pt --steps 600 --batch-size 4 --block-size 256 --merges 384 \
    && test -n "$(python /app/fross_transformer.py generate --output /app/fross-transformer.pt --text 'Как работает attention?')"
ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
EXPOSE 10000
CMD ["python", "/app/server.py"]
