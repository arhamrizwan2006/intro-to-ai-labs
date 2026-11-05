import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # the context manager runs the lifespan, so artefacts are loaded
    with TestClient(app) as c:
        yield c


GOOD = {"PClass": "First", "Gender": "Female", "Sibling": "Zero", "Embarked": "Cherbourg"}
