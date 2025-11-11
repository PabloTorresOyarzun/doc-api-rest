FROM python:3.11-slim

WORKDIR /app

COPY app/ /app/
COPY .env /app/.env

RUN pip install --no-cache-dir fastapi uvicorn python-multipart jinja2 \
    azure-ai-formrecognizer azure-core python-dotenv

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
