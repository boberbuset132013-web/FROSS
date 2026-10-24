FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY fross_brain_v04.py /app/fross_brain_v04.py
COPY services/python-brain/server.py /app/server.py
EXPOSE 10000
CMD ["python", "/app/server.py"]
