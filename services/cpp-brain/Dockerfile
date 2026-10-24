FROM gcc:14-bookworm AS builder
WORKDIR /src
COPY services/cpp-brain/main.cpp /src/main.cpp
RUN g++ -std=c++17 -O2 -pipe -Wall -Wextra -pedantic /src/main.cpp -o /fross-cpp-brain

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
ENV FROSS_CPP_EXECUTABLE=/app/fross-cpp-brain
WORKDIR /app
COPY --from=builder /fross-cpp-brain /app/fross-cpp-brain
COPY fross_brain_v05.py /app/fross_brain_v05.py
COPY services/cpp-brain/server.py /app/server.py
EXPOSE 10000
CMD ["python", "/app/server.py"]
