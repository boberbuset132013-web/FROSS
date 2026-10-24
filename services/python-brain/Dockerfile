FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN python -m pip install --no-cache-dir "numpy>=1.26,<3"
COPY fross_brain_v05.py /app/fross_brain_v05.py
COPY fross_neural_brain.py /app/fross_neural_brain.py
COPY services/llm_client.py /app/llm_client.py
COPY services/python-brain/server.py /app/server.py
# Train and save the weights into the image so startup can load a ready model.
RUN python /app/fross_neural_brain.py train --output /app/fross-neural-model.npz \
    && python /app/fross_neural_brain.py test --output /app/fross-neural-model.npz
ENV FROSS_NEURAL_MODEL_PATH=/app/fross-neural-model.npz
EXPOSE 10000
CMD ["python", "/app/server.py"]
