from pydantic import BaseModel

class PredictionResponse(BaseModel):
    survived: int
    probability: float
    model_file: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_file: str