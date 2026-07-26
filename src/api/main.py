from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.config import settings
from src.api.v1.auth import router as auth_router
from src.api.v1.history import router as history_router
from src.api.v1.predict import router as predict_router
from src.db.deps import init_db
from src.ml.model_loader import load_ml_assets


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as e:
        print(f"Database initialization warning: {e}")

    try:
        load_ml_assets()
    except Exception as e:
        print(f"ML Asset loading warning: {e}")

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API for Fracture Risk Assessment and MLOps Tracking",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(predict_router, prefix=settings.API_V1_STR)
app.include_router(history_router, prefix=settings.API_V1_STR)


@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
    }


@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    return {
        "status": "ok",
        "database": "configured",
        "ml_engine": "ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
