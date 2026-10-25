FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1
WORKDIR /app
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY fross_transformer.py /app/fross_transformer.py
COPY data/fross_corpus.jsonl /app/data/fross_corpus.jsonl
COPY services/python-brain/server.py /app/server.py
# All weights are initialized randomly and trained on the repository's original corpus.
# No pretrained checkpoint or remote model provider is used.
RUN python /app/fross_transformer.py train --corpus /app/data/fross_corpus.jsonl \
       --output /app/fross-transformer.pt --steps 3000 --batch-size 4 --block-size 320 --merges 384 \
    && test -n "$(python /app/fross_transformer.py generate --output /app/fross-transformer.pt --text 'Привет, FROSS.')" \
    && test -n "$(python /app/fross_transformer.py generate --output /app/fross-transformer.pt --text 'Как работает attention?')"
ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
EXPOSE 10000
CMD ["python", "/app/server.py"]
