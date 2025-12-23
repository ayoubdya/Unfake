import re


def clean_text(text: str) -> str:
  text = text.lower()
  text = re.sub(r"[^a-z0-9\s,.!?]", " ", text)
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def clean_article(text: str) -> str:
  text = text.lower().strip()
  text = re.sub(r"https?://\S+|www\.\S+", "", text)
  text = re.sub(r"\[.*?\]", "", text)
  text = re.sub(r"\w*\d\w*", "", text)
  text = re.sub(r"<.*?>+", "", text)
  text = re.sub(r"\n", " ", text)
  text = re.sub(r"\s+", " ", text)
  text = re.sub(r"\W", " ", text)
  return text
