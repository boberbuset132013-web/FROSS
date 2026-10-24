FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FROSS_TORCH_THREADS=1
WORKDIR /app
RUN python -m pip install --no-cache-dir "numpy>=1.26,<3" \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY fross_brain_v05.py /app/fross_brain_v05.py
COPY fross_neural_brain.py /app/fross_neural_brain.py
COPY fross_transformer.py /app/fross_transformer.py
COPY data/fross_corpus.jsonl /app/data/fross_corpus.jsonl
COPY services/python-brain/server.py /app/server.py
# All weights are initialized randomly and trained on the repository's original corpus.
# No pretrained checkpoint or remote model provider is used.
RUN python /app/fross_neural_brain.py train --output /app/fross-neural-model.npz \
    && python /app/fross_transformer.py train --corpus /app/data/fross_corpus.jsonl \
       --output /app/fross-transformer.pt --steps 240 --batch-size 8 --block-size 256 --merges 384
ENV FROSS_NEURAL_MODEL_PATH=/app/fross-neural-model.npz
ENV FROSS_TRANSFORMER_MODEL_PATH=/app/fross-transformer.pt
EXPOSE 10000
CMD ["python", "/app/server.py"]
