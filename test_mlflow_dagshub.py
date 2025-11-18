from pathlib import Path
import os
from dotenv import load_dotenv
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("Tracking URI:", tracking_uri)
print("Username    :", username)
print("Password set:", bool(password))

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("test_connection")

with mlflow.start_run(run_name="hello_dagshub") as run:
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)
    print("Run ID:", run.info.run_id)
