**Sentiment Analysis API with MLOps Integration**

This project automates ML workflow, model deployment, and performance monitoring using MLOps best practices. It trains and deploys a Sentiment Analysis model as a REST API using FastAPI and integrates MLflow, Prometheus, and Grafana for tracking and monitoring.

1. Model Training and Evaluation (train.py): Uses traditional models (Logistic Regression, Naive Bayes) and deep learning models (LSTM) to classify movie reviews into positive or negative sentiments.

2. Web Application (main.py): A FastAPI-based web app that allows users to input text and get sentiment predictions.

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

Downloads :

1. Download IMDB dataset for training models.
   
2. Prometheus and Grafana to set up monitoring dashboards.

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

git clone https://github.com/pvlex27/MLOps-Sentiment-Analysis.git

cd ML_Model-Sentiment_Analysis

2️⃣ Create Virtual Environment & Install Dependencies

python -m venv ml

ml\Scripts\activate # For Windows

pip install -r requirements.txt

3️⃣ Download & Extract Dataset

Extract the IMDB dataset inside the aclImdb/ directory with train/ and test/ folders.

To run Prometheus and grafana for monitoring dashboards download Prometheus and grafana and add grafana-v11.5.2 , prometheus-3.2.1.windows-amd64 to the project folder.

4️⃣ Train the Model

Add imdb dataset into the project to train that.

Rub python train.py

🔹 Training Includes:

✔ Preprocessing (Cleaning, Tokenization, TF-IDF Vectorization)

✔ Training Models: Logistic Regression, Naïve Bayes, LSTM

✔ Evaluation Metrics: Accuracy, Precision, Recall, F1-score

✔ Model Storage: Saved in Models/ directory

✔ Accuracy Reports: Stored in Accuracy-Reports/

✔ Model Training Graphs: Stored in Graph-Figure/

✔ Result: Logistic Regression performed the best.

5️⃣ Run and Start FastAPI Server

Run main.py tp predict the analysis

uvicorn main:app --host 0.0.0.0 --port 8080 and check using http://127.0.0.1:8080/ or http://localhost:8080/

🔹 FastAPI Features

✔ Authentication: Login via /login

✔ Prediction API: /predict (Accepts JSON input, returns sentiment prediction)

✔ Frontend: Interactive UI (Login & Sentiment Prediction Page)

6️⃣ Access API Documentation

Swagger UI: http://localhost:8080/docs

ReDoc: http://localhost:8080/redoc

7️⃣ Test API Endpoints Using cURL(command promt-windows)

Main app - curl -X POST "http://127.0.0.1:8080/" -H "Content-Type: application/x-www-form-urlencoded" -u admin:password123 -d "text=I love this movie!"

mlflow - curl -i http://localhost:8080

grafana - curl -i http://localhost:3001

prometheus - curl -i http://localhost:9090

📦 Deployment with Docker

1️⃣ Build & Run Docker Containers

docker-compose up --build -d

2️⃣ Access UI Services

Service URL

FastAPI http://localhost:8080

MLflow http://localhost:5000

Grafana http://localhost:3001

Prometheus http://localhost:9090

Conclusion

This project successfully automates an end-to-end Sentiment Analysis workflow — from data preprocessing, model training, and deployment to monitoring using modern MLOps tools like FastAPI, Docker, MLflow, Prometheus, and Grafana. The Logistic Regression model showed the best performance, and the API is fully functional with user authentication and real-time monitoring.

📝 Future Enhancements

✔ CI/CD Pipeline Integration

✔ Cloud Deployment (AWS/GCP/Azure)

✔ Auto-scaling with Kubernetes

🛠 Contributing Feel free to submit issues or feature requests. Contributions are always welcome!

🔥 Developed with passion using MLOps principles! 🚀
