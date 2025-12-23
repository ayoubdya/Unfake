import os
import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.path.join(
  os.path.dirname(__file__), "..", "data", "fake_news_classifier"
)
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None


def clean_text(text: str) -> str:
  text = text.lower()
  text = re.sub(r"[^!?,\.\w]", " ", text)
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def load_model():
  global model, tokenizer

  if not os.path.exists(MODEL_DIR):
    raise RuntimeError(
      f"Model directory not found: {MODEL_DIR}. "
      "Please train the model first using the train.ipynb notebook."
    )

  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
  model.to(DEVICE)
  model.eval()

  print(f"Model loaded from {MODEL_DIR}")
  print(f"Using device: {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
  # load_model()
  yield


app = FastAPI(
  title="Fake News Classifier API",
  description="API for classifying statements as fake or real news using a fine-tuned RoBERTa model.",
  version="1.0.0",
  lifespan=lifespan,
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class StatementRequest(BaseModel):
  statement: str

  class Config:
    json_schema_extra = {
      "example": {
        "statement": "Scientists confirm that regular exercise improves cardiovascular health."
      }
    }


class PredictionResponse(BaseModel):
  statement: str
  prediction: str  # "Fake" or "Real"
  confidence: float  # Confidence score (0-1)
  probabilities: dict  # {"fake": 0.3, "real": 0.7}

  class Config:
    json_schema_extra = {
      "example": {
        "statement": "Scientists confirm that regular exercise improves cardiovascular health.",
        "prediction": "Real",
        "confidence": 0.92,
        "probabilities": {"fake": 0.08, "real": 0.92},
      }
    }


@app.get("/")
async def root():
  return {
    "status": "healthy",
    "message": "Fake News Classifier API is running",
    "model_loaded": model is not None,
  }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: StatementRequest):
  """
  Classify a statement as fake or real news.

  Returns the prediction, confidence score, and probability distribution.
  """
  # if model is None or tokenizer is None:
  #   raise HTTPException(status_code=503, detail="Model not loaded")

  if not request.statement.strip():
    raise HTTPException(status_code=400, detail="Statement cannot be empty")

  cleaned_text = clean_text(request.statement)

  # encoding = tokenizer(
  #   cleaned_text,
  #   add_special_tokens=True,
  #   max_length=MAX_LENGTH,
  #   padding="max_length",
  #   truncation=True,
  #   return_attention_mask=True,
  #   return_tensors="pt",
  # )

  # input_ids = encoding["input_ids"].to(DEVICE)
  # attention_mask = encoding["attention_mask"].to(DEVICE)

  # with torch.no_grad():
  #   outputs = model(input_ids=input_ids, attention_mask=attention_mask)
  #   probs = torch.softmax(outputs.logits, dim=1)
  #   pred_class = torch.argmax(probs, dim=1).item()

  #   prob_fake = probs[0][0].item()
  #   prob_real = probs[0][1].item()
  #   confidence = probs[0][pred_class].item()

  # prediction = "Real" if pred_class == 1 else "Fake"

  return PredictionResponse(
    statement=request.statement,
    prediction="Real",
    confidence=round(0.64, 4),
    probabilities={"fake": round(0.36, 4), "real": round(0.64, 4)},
  )


@app.post("/predict/batch")
async def predict_batch(statements: list[str]):
  """
  Classify multiple statements at once.

  Returns predictions for all statements.
  """
  if model is None or tokenizer is None:
    raise HTTPException(status_code=503, detail="Model not loaded")

  if not statements:
    raise HTTPException(status_code=400, detail="Statements list cannot be empty")

  if len(statements) > 100:
    raise HTTPException(status_code=400, detail="Maximum 100 statements per batch")

  results = []
  for statement in statements:
    if not statement.strip():
      results.append({"statement": statement, "error": "Empty statement"})
      continue

    cleaned_text = clean_text(statement)
    encoding = tokenizer(
      cleaned_text,
      add_special_tokens=True,
      max_length=MAX_LENGTH,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      probs = torch.softmax(outputs.logits, dim=1)
      pred_class = torch.argmax(probs, dim=1).item()

      prob_fake = probs[0][0].item()
      prob_real = probs[0][1].item()
      confidence = probs[0][pred_class].item()

    prediction = "Real" if pred_class == 1 else "Fake"

    results.append(
      {
        "statement": statement,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {"fake": round(prob_fake, 4), "real": round(prob_real, 4)},
      }
    )

  return {"results": results}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
