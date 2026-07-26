from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.api.deps import get_current_user
from src.db.deps import get_db
from src.db.models import PredictionRecord, User

router = APIRouter(prefix="/history", tags=["Medical History"])


@router.get("", status_code=status.HTTP_200_OK)
def get_user_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.user_id == current_user.id)
        .order_by(PredictionRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id": rec.id,
            "clinical_inputs": rec.clinical_inputs,
            "risk_score": rec.risk_score,
            "prediction_label": rec.prediction_label,
            "classification": "High Risk" if rec.prediction_label == 1 else "Low Risk",
            "model_version": rec.model_version,
            "created_at": rec.created_at,
        }
        for rec in records
    ]


@router.get("/{record_id}", status_code=status.HTTP_200_OK)
def get_prediction_record_by_id(
    record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = (
        db.query(PredictionRecord)
        .filter(PredictionRecord.id == record_id, PredictionRecord.user_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction record not found",
        )
    return {
        "id": record.id,
        "clinical_inputs": record.clinical_inputs,
        "risk_score": record.risk_score,
        "prediction_label": record.prediction_label,
        "classification": "High Risk" if record.prediction_label == 1 else "Low Risk",
        "model_version": record.model_version,
        "created_at": record.created_at,
    }
