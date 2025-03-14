from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Custom metrics
metrics.info("mlflow_server", "MLflow Tracking Server Metrics")

@app.route("/")
def index():
    return "MLflow Prometheus Exporter Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)