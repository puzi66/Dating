FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY matching ./matching
COPY app ./app

RUN useradd --create-home --uid 10001 shalosh && mkdir -p /var/data && chown shalosh /var/data
USER shalosh

ENV APP_ENV=production DATA_DIR=/var/data PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
