FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scheduler_app ./scheduler_app
COPY stages.json ./stages.json

EXPOSE 8080

CMD ["uvicorn", "scheduler_app.main:app", "--host", "0.0.0.0", "--port", "8080"]
