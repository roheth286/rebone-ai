from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from src.api.deps import get_current_user
from src.db.deps import get_db
from src.db.models import PredictionRecord, User
from src.ml.engine import run_inference
from src.schemas.prediction import PatientInputSchema, PredictionOutputSchema

router = APIRouter(tags=["ML Prediction"])


@router.post("/predict", response_model=PredictionOutputSchema, status_code=status.HTTP_200_OK)
def predict_fracture_risk(
    patient_in: PatientInputSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient_dict = patient_in.model_dump(by_alias=True)
    inference_result = run_inference(patient_dict)

    prediction_label = 1 if inference_result["classification"] == "High Risk" else 0

    record = PredictionRecord(
        user_id=current_user.id,
        clinical_inputs=patient_dict,
        risk_score=inference_result["risk_score"],
        prediction_label=prediction_label,
        model_version=inference_result["model_version"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return inference_result
