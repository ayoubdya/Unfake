import os
from contextlib import asynccontextmanager
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from api.types import (
  ArticleRequest,
  ArticlePredictionResponse,
  PredictionResponse,
  StatementRequest,
)
from api.text_processing import clean_text, clean_article

# HEADLINE
HEADLINE_MODEL_DIR = os.path.join(
  os.path.dirname(__file__), "..", "data", "RoBERTa_Classifier"
)
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

headline_model = None
tokenizer = None

# ARTICLE
ARTICLE_MODEL_DIR = os.path.join(
  os.path.dirname(__file__), "..", "data", "gradient_boosting_classifier.pkl"
)
VECTORIZER_DIR = os.path.join(
  os.path.dirname(__file__), "..", "data", "tfidf_vectorizer.pkl"
)

article_model: GradientBoostingClassifier | None = None
vectorizer: TfidfVectorizer | None = None


def load_article_model():
  global article_model, vectorizer

  if not os.path.exists(ARTICLE_MODEL_DIR) or not os.path.exists(VECTORIZER_DIR):
    raise RuntimeError(
      f"Model or vectorizer file not found: {ARTICLE_MODEL_DIR} or {VECTORIZER_DIR}. "
      "Please train the model first using the train.ipynb notebook."
    )

  with open(ARTICLE_MODEL_DIR, "rb") as f:
    article_model = pickle.load(f)

  with open(VECTORIZER_DIR, "rb") as f:
    vectorizer = pickle.load(f)

  print(f"Article model loaded from {ARTICLE_MODEL_DIR}")
  print(f"Vectorizer loaded from {VECTORIZER_DIR}")


def load_headline_model():
  global headline_model, tokenizer

  if not os.path.exists(HEADLINE_MODEL_DIR):
    raise RuntimeError(
      f"Model directory not found: {HEADLINE_MODEL_DIR}. "
      "Please train the model first using the train.ipynb notebook."
    )

  tokenizer = AutoTokenizer.from_pretrained(HEADLINE_MODEL_DIR)
  headline_model = AutoModelForSequenceClassification.from_pretrained(
    HEADLINE_MODEL_DIR
  )
  headline_model.to(DEVICE)
  headline_model.eval()

  print(f"Model loaded from {HEADLINE_MODEL_DIR}")
  print(f"Using device: {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
  load_headline_model()
  load_article_model()
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


@app.get("/")
async def root():
  return {
    "status": "healthy",
    "message": "Fake News Classifier API is running",
    "model_loaded": headline_model is not None,
  }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: StatementRequest):
  if headline_model is None or tokenizer is None:
    raise HTTPException(status_code=503, detail="Model not loaded")

  if not request.statement.strip():
    raise HTTPException(status_code=400, detail="Statement cannot be empty")

  cleaned_text = clean_text(request.statement)

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
    outputs = headline_model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    prob_fake = probs[0][0].item()
    prob_real = probs[0][1].item()
    confidence = probs[0][pred_class].item()  # type: ignore

  prediction = "Real" if pred_class == 1 else "Fake"

  return PredictionResponse(
    statement=request.statement,
    prediction=prediction,
    confidence=round(confidence, 2),
    probabilities={"fake": round(prob_fake, 2), "real": round(prob_real, 2)},
  )


@app.post("/predict/batch")
async def predict_batch(statements: list[str]):
  if headline_model is None or tokenizer is None:
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
      outputs = headline_model(input_ids=input_ids, attention_mask=attention_mask)
      probs = torch.softmax(outputs.logits, dim=1)
      pred_class = torch.argmax(probs, dim=1).item()

      prob_fake = probs[0][0].item()
      prob_real = probs[0][1].item()
      confidence = probs[0][pred_class].item()  # type: ignore

    prediction = "Real" if pred_class == 1 else "Fake"

    results.append(
      {
        "statement": statement,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {"fake": round(prob_fake, 2), "real": round(prob_real, 2)},
      }
    )

  return {"results": results}


@app.post("/predict/article", response_model=ArticlePredictionResponse)
async def predict_article(request: ArticleRequest):
  if article_model is None or vectorizer is None:
    raise HTTPException(status_code=503, detail="Article model not loaded")

  if not request.article.strip():
    raise HTTPException(status_code=400, detail="Article cannot be empty")

  cleaned_text = clean_article(request.article)

  text_vectorized = vectorizer.transform([cleaned_text])

  probabilities = article_model.predict_proba(text_vectorized)[0]
  pred_class = article_model.predict(text_vectorized)[0]

  prob_fake = probabilities[0]
  prob_real = probabilities[1]
  confidence = probabilities[pred_class]

  prediction = "Real" if pred_class == 1 else "Fake"

  return ArticlePredictionResponse(
    article=request.article[:500] + "..."
    if len(request.article) > 500
    else request.article,
    prediction=prediction,
    confidence=round(float(confidence), 2),
    probabilities={
      "fake": round(float(prob_fake), 2),
      "real": round(float(prob_real), 2),
    },
  )


@app.post("/predict/article/batch")
async def predict_article_batch(articles: list[str]):
  if article_model is None or vectorizer is None:
    raise HTTPException(status_code=503, detail="Article model not loaded")

  if not articles:
    raise HTTPException(status_code=400, detail="Articles list cannot be empty")

  if len(articles) > 50:
    raise HTTPException(status_code=400, detail="Maximum 50 articles per batch")

  results = []
  for article in articles:
    if not article.strip():
      results.append({"article": article, "error": "Empty article"})
      continue

    cleaned_text = clean_article(article)
    text_vectorized = vectorizer.transform([cleaned_text])

    probabilities = article_model.predict_proba(text_vectorized)[0]
    pred_class = article_model.predict(text_vectorized)[0]

    prob_fake = probabilities[0]
    prob_real = probabilities[1]
    confidence = probabilities[pred_class]

    prediction = "Real" if pred_class == 1 else "Fake"

    results.append(
      {
        "article": article[:500] + "..." if len(article) > 500 else article,
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "probabilities": {
          "fake": round(float(prob_fake), 2),
          "real": round(float(prob_real), 2),
        },
      }
    )

  return {"results": results}


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
