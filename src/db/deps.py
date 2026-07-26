from typing import Generator
import src.db.models
from src.db.session import Base, SessionLocal, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
