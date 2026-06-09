FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dash

COPY . .

ENV PYTHONPATH=/app
ENV RESULTS_DIR=/app/results

EXPOSE 8050

CMD ["python", "reporting/dashboard.py", "--results-dir", "/app/results", "--host", "0.0.0.0", "--port", "8050"]
