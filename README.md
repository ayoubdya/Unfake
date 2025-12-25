# UnFake - Fake News Detection System

An AI-powered fake news detection system that classifies news content as **Real** or **Fake** using machine learning.

## Features

- **Dual-model approach**: RoBERTa for headlines (74.69% accuracy), Gradient Boosting for articles (99.57% accuracy)
- **REST API**: FastAPI backend with single and batch prediction endpoints
- **Web Interface**: Simple frontend for easy news verification

## Quick Start

```bash
# Install dependencies
uv sync

# Run the API
cd api
uvicorn main:app --reload --port 8000

# Open frontend (in another terminal)
cd frontend
python -m http.server 3000
```

## API Endpoints

| Endpoint                 | Method | Description                    |
| ------------------------ | ------ | ------------------------------ |
| `/predict`               | POST   | Classify a statement           |
| `/predict/article`       | POST   | Classify an article            |
| `/predict/batch`         | POST   | Batch statement classification |
| `/predict/article/batch` | POST   | Batch article classification   |

## Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Models**: RoBERTa (headlines), Gradient Boosting + TF-IDF (articles)

## Author

Ayoub DYA - ENSAM Casablanca
