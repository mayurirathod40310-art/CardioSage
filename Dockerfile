FROM node:22-bookworm-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/cardiosage /app/backend/cardiosage
COPY data /app/data
COPY models /app/models
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

ENV CARDIOSAGE_FRONTEND=/app/frontend/dist \
    CARDIOSAGE_DATA=/app/data/heart.csv \
    CARDIOSAGE_MODELS=/app/models \
    CARDIOSAGE_DB=/var/data/cardiosage.db

EXPOSE 10000
CMD ["sh", "-c", "uvicorn cardiosage.main:app --app-dir /app/backend --host 0.0.0.0 --port ${PORT:-10000}"]
