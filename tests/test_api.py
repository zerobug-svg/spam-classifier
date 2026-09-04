from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


def test_spam_prediction():

    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You won a free prize!"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result

    assert "spam_probability" in result

    assert "model_version" in result

    assert result["prediction"] in [
        "spam",
        "ham"
    ]

    assert 0.0 <= result["spam_probability"] <= 1.0

    assert result["model_version"]


def test_normal_message():

    response = client.post(
        "/predict",
        json={
            "message": "Can you send me the project report?"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result

    assert "spam_probability" in result

    assert "model_version" in result

    assert result["prediction"] in [
        "spam",
        "ham"
    ]

    assert 0.0 <= result["spam_probability"] <= 1.0

    assert result["model_version"]