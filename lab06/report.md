# Lab 06 - FastAPI model deployment

**Model:** SVC on four categorical features (PClass, Gender, Sibling, Embarked), trained on the Lab 05 Titanic data (889 rows, 80/20 stratified split).
Test accuracy **77.0%** vs **61.8%** for always predicting the majority class.

## Design
- `train_model.py` saves five pickles (SVC + four LabelEncoders) and one bundled `Pipeline` (OrdinalEncoder + SVC). Both routes agree on all 72 possible inputs.
- `app/main.py` loads artefacts once in `lifespan`. Endpoints: `/health`, `/predict`, `/predict/pipeline`, `/predict/lenient`, `/metrics`.
- Request fields are `str`, so a malformed body gives **422** while a well-formed body with an unknown category gives **400** listing the allowed values.
- `/predict/lenient` maps unknown categories to `-1` and reports them in `unknown_fields`.
- `/metrics` exposes a request counter and latency histogram (Prometheus).
- `Dockerfile` (python:3.12-slim + uvicorn). Not built here; sandbox has no daemon access.

## Example requests
```
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"PClass":"First","Gender":"Female","Sibling":"Zero","Embarked":"Cherbourg"}'
-> {"predicted_label":1,"prediction":"SURVIVED","confidence":0.8323}

curl -X POST localhost:8000/predict -d '{"PClass":"First","Gender":"Female","Sibling":"Zero","Embarked":"Atlantis"}' -H 'Content-Type: application/json'
-> 400 {"detail":"Unknown value 'Atlantis' for Embarked. Allowed: ['Cherbourg','Queenstown','Southampton']"}

curl -X POST localhost:8000/predict/lenient -H 'Content-Type: application/json' \
  -d '{"PClass":"First","Gender":"Female","Sibling":"Zero","Embarked":"Atlantis"}'
-> {"predicted_label":1,"prediction":"SURVIVED","confidence":0.6756,"unknown_fields":["Embarked"]}
```

## Results
- `pytest`: 11 passed (valid, 400, 422, route agreement, lenient, metrics).
- Latency over 200 requests: `/predict` mean 8.5 ms, p95 11.2 ms; `/predict/pipeline` mean 14.1 ms (DataFrame + pipeline overhead).
- Predicted survival by group: women ~100% (third class 75%), men ~0-17%. This mirrors the data, where sex and class dominate.

## Notes
- The teacher's notebook supplied no `main.py` or pkl files, so the service and model were built from its spec.
- scikit-learn 1.9 warns that `SVC(probability=True)` is deprecated; it still works, and the lab needs the probability for `confidence`.

## Run
```
pip install -r requirements.txt
python train_model.py
pytest
uvicorn app.main:app --port 8000     # docs at /docs
```
