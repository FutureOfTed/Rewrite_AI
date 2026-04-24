from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_inference_api():
    response = client.post(
        "/inference/",
        json={"features": [1.0, 2.0, 3.0]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
