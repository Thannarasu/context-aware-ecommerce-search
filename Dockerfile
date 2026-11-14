FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 HF_HOME=/home/app/.cache/huggingface
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements-phase1.txt ./
RUN pip install --no-cache-dir -r requirements-phase1.txt
COPY --chown=app:app app.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
