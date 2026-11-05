# app/schemas.py
"""Request and response models for the Titanic survival API.

The request fields are plain strings on purpose. Pydantic therefore rejects a missing
field or a wrong type with 422, while a well-formed body carrying a category the encoder
has never seen reaches the handler, which answers 400 with the list of allowed values.
(With typing.Literal the unknown category would be rejected earlier, with 422.)
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"PClass": "First", "Gender": "Female", "Sibling": "Zero", "Embarked": "Cherbourg"}
    ]})

    PClass: str       # First, Second, Third
    Gender: str       # Male, Female
    Sibling: str      # Zero, One, Two, Three  (Three means three or more)
    Embarked: str     # Cherbourg, Queenstown, Southampton


class PredictResponse(BaseModel):
    predicted_label: int
    prediction: Literal["SURVIVED", "NOT SURVIVED"]
    confidence: float     # model probability of the predicted class, 0 to 1


class LenientResponse(PredictResponse):
    unknown_fields: list[str]   # fields whose value was mapped to the reserved code -1
