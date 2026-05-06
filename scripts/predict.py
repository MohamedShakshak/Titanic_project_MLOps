# scripts/predict.py
"""
Example client for the MLflow inference server.
Run after the Docker container is up:
    uv run python scripts/predict.py
"""
import json
import requests
from dotenv import load_dotenv

load_dotenv()

INFERENCE_URL = "http://localhost:5001"


def check_health() -> bool:
    """Check if the inference server is healthy."""
    try:
        response = requests.get(f"{INFERENCE_URL}/ping", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def predict_single(passenger: dict) -> int:
    payload = {"dataframe_records": [passenger]}

    response = requests.post(
        f"{INFERENCE_URL}/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    
    # Print response body before raising so we can see the error
    if not response.ok:
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    response.raise_for_status()
    return response.json()["predictions"][0]


def predict_batch(passengers: list[dict]) -> list[int]:
    """Send multiple passengers for prediction."""
    if not passengers:
        return []

    columns = list(passengers[0].keys())
    data = [[p[col] for col in columns] for p in passengers]

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": data,
        }
    }

    response = requests.post(
        f"{INFERENCE_URL}/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["predictions"]


if __name__ == "__main__":
    # Check server health
    if not check_health():
        print("ERROR: Inference server is not running.")
        print("Start it with: docker compose -f docker-compose.inference.yaml up -d")
        exit(1)

    print("Server is healthy.")

    # Test passengers
    passengers = [
        {
            "Pclass": 1,
            "Sex": "female",
            "Age": 29.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 211.3,
            "Embarked": "S",
            "Name": "Cumings, Mrs. John Bradley",
            "Ticket": "PC 17599",
            "Cabin": "C85",
        },
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S",
            "Name": "Braund, Mr. Owen Harris",
            "Ticket": "A/5 21171",
            "Cabin": "",
        },
        {
            "Pclass": 2,
            "Sex": "female",
            "Age": 26.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 13.0,
            "Embarked": "S",
            "Name": "Hewlett, Mrs. (Mary D Kingcome)",
            "Ticket": "248706",
            "Cabin": "",
        },
    ]

    # Single prediction
    print("\n── Single Prediction ──")
    result = predict_single(passengers[0])
    status = "Survived" if result == 1 else "Did not survive"
    print(f"{passengers[0]['Name']}: {status}")

    # Batch prediction
    print("\n── Batch Prediction ──")
    results = predict_batch(passengers)
    for passenger, prediction in zip(passengers, results):
        status = "Survived" if prediction == 1 else "Did not survive"
        print(f"{passenger['Name']}: {status}")