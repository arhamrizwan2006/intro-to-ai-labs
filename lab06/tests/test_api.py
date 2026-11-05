import pytest

from .conftest import GOOD


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@pytest.mark.parametrize("path", ["/predict", "/predict/pipeline"])
def test_valid_prediction(client, path):
    r = client.post(path, json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("SURVIVED", "NOT SURVIVED")
    assert body["predicted_label"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_first_class_woman_survives_third_class_man_does_not(client):
    a = client.post("/predict", json=GOOD).json()
    b = client.post("/predict", json={"PClass": "Third", "Gender": "Male",
                                      "Sibling": "Zero", "Embarked": "Southampton"}).json()
    assert a["prediction"] == "SURVIVED"
    assert b["prediction"] == "NOT SURVIVED"


def test_both_routes_agree(client):
    for p in ("First", "Second", "Third"):
        for g in ("Male", "Female"):
            body = {**GOOD, "PClass": p, "Gender": g}
            assert (client.post("/predict", json=body).json()["predicted_label"]
                    == client.post("/predict/pipeline", json=body).json()["predicted_label"])


def test_unknown_category_is_400(client):
    r = client.post("/predict", json={**GOOD, "Embarked": "Atlantis"})
    assert r.status_code == 400
    assert "Allowed" in r.json()["detail"]


def test_missing_field_is_422(client):
    bad = {k: v for k, v in GOOD.items() if k != "Gender"}
    assert client.post("/predict", json=bad).status_code == 422


def test_wrong_type_is_422(client):
    assert client.post("/predict", json={**GOOD, "Sibling": 1}).status_code == 422


def test_lenient_reports_unknown_fields(client):
    r = client.post("/predict/lenient", json={**GOOD, "Embarked": "Atlantis"})
    assert r.status_code == 200
    assert r.json()["unknown_fields"] == ["Embarked"]


def test_lenient_known_has_no_unknowns(client):
    assert client.post("/predict/lenient", json=GOOD).json()["unknown_fields"] == []


def test_metrics_counts_requests(client):
    client.post("/predict", json=GOOD)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and 'path="/predict"' in text
