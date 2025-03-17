#main.py file  
from fastapi import FastAPI
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from dotenv import load_dotenv
from loguru import logger
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.sessions import SessionMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_here")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load credentials from .env
VALID_USERNAME = os.getenv("API_USERNAME")
VALID_PASSWORD = os.getenv("API_PASSWORD")

# ✅ Initialize Loguru logging
logger.add("logs/api.log", rotation="1 day", level="INFO", format="{time} {level} {message}")

# ✅ Prometheus Monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Model paths
MODEL_DIR = "Models"
LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
NB_MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidff_vectorizer.pkl")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_sentiment_model.h5")

# Load models safely
try:
    lr_model = joblib.load(LR_MODEL_PATH)
    nb_model = joblib.load(NB_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

    logger.info("✅ All models loaded successfully!")

except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    raise RuntimeError("Model loading failed. Check model files.")

# Preprocess text
def preprocess_text(text):
    return text.lower().strip()

# Prediction for Logistic Regression & Naive Bayes
def predict_tfidf(text, model):
    try:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        return "Positive" if prediction == 1 else "Negative"
    except Exception as e:
        logger.error(f"❌ Error in predict_tfidf: {e}")
        return "Prediction Failed"

# Prediction for LSTM
def predict_lstm(text):
    try:
        text_seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=200, padding="post")
        prediction = lstm_model.predict(text_seq)
        return "Positive" if prediction[0][0] > 0.4 else "Negative"
    except Exception as e:
        logger.error(f"❌ Error in predict_lstm: {e}")
        return "Prediction Failed"

# ✅ MLflow Logging Function
def log_prediction(model_name, text, prediction):
    mlflow.set_tracking_uri("http://mlflow:5000")  # Use MLflow container name
    mlflow.set_experiment("Sentiment_Analysis")

    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("input_text", text)
        mlflow.log_metric("prediction", 1 if prediction == "Positive" else 0)
        logger.info(f"📊 MLflow Logged: Model={model_name}, Prediction={prediction}")

# ✅ Middleware for Logging API Requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📩 Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"📤 Response: {response.status_code}")
    return response

# Authentication Function
def is_authenticated(request: Request):
    return request.session.get("authenticated", False)

# Login Page
@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

# Logout Route
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

# Home Page - Render Input Form (Authenticated Users Only)
@app.get("/")
def home(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Handle Form Submission & Show Predictions
@app.post("/")
def predict(request: Request, text: str = Form(...)):
    if not is_authenticated(request):
        return RedirectResponse(url="/login")

    processed_text = preprocess_text(text)

    predictions = {
        "Logistic Regression": predict_tfidf(processed_text, lr_model),
        "Naive Bayes": predict_tfidf(processed_text, nb_model),
        "LSTM": predict_lstm(processed_text)
    }


    for model, pred in predictions.items():
        log_prediction(model, processed_text, pred)

    return templates.TemplateResponse("index.html", {"request": request, "text": text, "prediction": predictions})
