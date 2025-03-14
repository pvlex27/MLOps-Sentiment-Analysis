Sentiment Analysis API with MLOps Integration

This project automates ML workflow, model deployment, and performance monitoring using MLOps best practices. It trains and deploys a Sentiment Analysis model as a REST API using FastAPI and integrates MLflow, Prometheus, and Grafana for tracking and monitoring.

📌 Features

✅ Sentiment Analysis using the IMDB dataset

✅ Multiple Models: Logistic Regression, Naïve Bayes, LSTM (Logistic Regression performs best)

✅ FastAPI for API deployment with authentication

✅ MLflow for tracking experiments, model logging, and performance

✅ Prometheus for API & model metric monitoring

✅ Grafana for performance visualization

✅ Loguru for API request-response logging

✅ Dockerized setup for easy deployment

✅ Developed & tested on Windows


🛠 Tech Stack


Programming: Python3. 9

Libraries: Scikit-learn, TensorFlow/PyTorch, FastAPI, Pandas, NumPy

Model Deployment: FastAPI, Docker

Monitoring: MLflow, Prometheus, Grafana

Virtual Environment: ml

Download :

1. Download imdb dataset and add    aclImdb folder to the project
2. To run Prometheus and grafana download Prometheus and grafana

📁 Project Structure

ML_Model-Sentiment_Analysis/

│── aclImdb/ # Dataset (Train/Test folders)

│── Accuracy-Reports/ # Model accuracy reports & figures

│── Models/ # Trained models (Logistic Regression, Naïve Bayes, LSTM)

│── Templates/ # Frontend (login/index.html)

│── logs/ # API request/response logs

│── mlruns/ # MLflow data tracking

│── mlflow_metrics.py # MLflow tracking setup

│── prometheus.yml # Prometheus configuration

│── Dockerfile # Docker container setup

│── docker-compose.yml # Docker Compose configuration

│── requirements.txt # Required dependencies

│── train.py # Model training & evaluation

│── main.py # FastAPI main application (Prediction API)

│── .env # Authentication credentials


📌 Setup & Installation


1️⃣ Clone the Repository

git clone

cd ML_Model-Sentiment_Analysis

2️⃣ Create Virtual Environment & Install Dependencies

python -m venv ml

ml\Scripts\activate # For Windows

pip install -r requirements.txt

3️⃣ Download & Extract Dataset

Extract the IMDB dataset inside the aclImdb/ directory with train/ and test/ folders.

4️⃣ Train the Model

python train.py

🔹 Training Includes:

✔ Preprocessing (Cleaning, Tokenization, TF-IDF Vectorization)

✔ Training Models: Logistic Regression, Naïve Bayes, LSTM

✔ Evaluation Metrics: Accuracy, Precision, Recall, F1-score

✔ Model Storage: Saved in Models/ directory

✔ Accuracy Reports: Stored in Accuracy-Reports/

✔ Model Training Graphs: Stored in Graph-Figure/

5️⃣ Start FastAPI Server

uvicorn main:app --host 0.0.0.0 --port 8080

🔹 FastAPI Features

✔ Authentication: Login via /login

✔ Prediction API: /predict (Accepts JSON input, returns sentiment prediction)

✔ Frontend: Interactive UI (Login & Sentiment Prediction Page)

6️⃣ Access API Documentation

Swagger UI: http://localhost:8080/docs

ReDoc: http://localhost:8080/redoc

7️⃣ Test API Endpoints Using cURL

curl -X POST "http://127.0.0.1:8080/" -H "Content-Type: application/x-www-form-urlencoded" -u admin:password123 -d "text=I love this movie!"

📦 Deployment with Docker

1️⃣ Build & Run Docker Containers

docker-compose up --build -d

2️⃣ Access UI Services

Service URL

FastAPI http://localhost:8080

MLflow http://localhost:5000

Grafana http://localhost:3001

Prometheus http://localhost:9090

📝 Future Enhancements

✔ CI/CD Pipeline Integration

✔ Cloud Deployment (AWS/GCP/Azure)

✔ Auto-scaling with Kubernetes

🛠 Contributing Feel free to submit issues or feature requests. Contributions are always welcome!

🔥 Developed with passion using MLOps principles! 🚀
