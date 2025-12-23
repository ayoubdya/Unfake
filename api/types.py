from pydantic import BaseModel


class StatementRequest(BaseModel):
  statement: str

  class Config:
    json_schema_extra = {
      "example": {
        "statement": "Scientists confirm that regular exercise improves cardiovascular health."
      }
    }


class ArticleRequest(BaseModel):
  article: str

  class Config:
    json_schema_extra = {
      "example": {
        "article": "A new study published in the Journal of Medicine reveals that drinking water daily can improve overall health. Researchers at Harvard University conducted a 10-year study..."
      }
    }


class PredictionResponse(BaseModel):
  statement: str
  prediction: str
  confidence: float
  probabilities: dict

  class Config:
    json_schema_extra = {
      "example": {
        "statement": "Scientists confirm that regular exercise improves cardiovascular health.",
        "prediction": "Real",
        "confidence": 0.92,
        "probabilities": {"fake": 0.08, "real": 0.92},
      }
    }


class ArticlePredictionResponse(BaseModel):
  article: str
  prediction: str
  confidence: float
  probabilities: dict

  class Config:
    json_schema_extra = {
      "example": {
        "article": "A new study published in the Journal of Medicine...",
        "prediction": "Real",
        "confidence": 0.89,
        "probabilities": {"fake": 0.11, "real": 0.89},
      }
    }
