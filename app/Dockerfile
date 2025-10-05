FROM python:3.11-slim
# ---------- Set Working Directory ----------
WORKDIR /app

COPY model.json .
COPY data.csv .
COPY label_encoder.joblib .
COPY static ./static
# ---------- Install Dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ---------- Copy Application Code ----------
COPY main.py .

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run FastAPI with Uvicorn ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]