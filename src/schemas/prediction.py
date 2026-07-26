from typing import Optional
from pydantic import BaseModel, Field


class PatientInputSchema(BaseModel):
    Gender: str
    Age: float
    BMI: float
    L1_4T: float = Field(..., alias="L1-4T")
    FNT: float
    TLT: float
    Calsium: float
    Calcitriol: float
    Bisphosphonate: float = 0.0
    Calcitonin: float = 0.0
    VT: float = 0.0
    VD: float = 1.0
    OP: float = 0.0
    Smoking: float = 0.0
    Drinking: float = 0.0

    class Config:
        populate_by_name = True


class PredictionOutputSchema(BaseModel):
    risk_score: float
    classification: str
    threshold_used: float = 0.8420
    decision: str
    model_version: str = "isolation_forest_v1.0"
