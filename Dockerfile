FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput
RUN python manage.py migrate --run-syncdb

# Importer les données du bucket au build
RUN python manage.py sync_bucket || true

EXPOSE 7860

CMD ["gunicorn", "dem_mcm.wsgi:application", "--bind", "0.0.0.0:7860", "--workers", "2"]