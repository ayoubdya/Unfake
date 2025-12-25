# UnFake - Fake News Detection System

An AI-powered fake news detection system that classifies news content as **Real** or **Fake** using machine learning.

## Features

- **Dual-model approach**: RoBERTa for headlines (74.69% accuracy), Gradient Boosting for articles (99.57% accuracy)
- **REST API**: FastAPI backend with single and batch prediction endpoints
- **Web Interface**: Simple frontend for easy news verification
- **Data Collection**: PolitiFact scraper for gathering fact-checked statements

## Project Structure

```
unfake/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── text_processing.py # Text preprocessing
│   └── types.py           # Pydantic models
├── data/                   # Datasets and trained models
│   ├── RoBERTa_Classifier/ # Headline model
│   └── True_Fake/         # Article dataset
├── frontend/              # Web interface
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── notebooks/             # Training notebooks
│   ├── train_headline.ipynb
│   └── train_article.ipynb
└── scraper/               # Data collection
    └── politifact.py
```

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

## Example Usage

```python
import requests

# Classify a statement
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Some news headline or statement here"}
)
print(response.json())
# {"text": "...", "prediction": "real", "confidence": 0.87}
```

## Datasets

- **Headlines**: 25,999 PolitiFact fact-checked statements
- **Articles**: 44,898 news articles (True.csv + Fake.csv)

## Tech Stack

- **Backend**: FastAPI, PyTorch, Transformers, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Models**: RoBERTa (headlines), Gradient Boosting + TF-IDF (articles)
- **Other**: NLTK, Pandas, NumPy, BeautifulSoup
