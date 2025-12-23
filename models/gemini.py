import os
import sys
import time
from typing import List, AsyncGenerator

from google import genai


class Gemini:
  EMBEDDING_DIMENSION = 768

  def __init__(self):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
      print("GEMINI_API_KEY is not set", file=sys.stderr)
      sys.exit(1)
    self.client = genai.Client(api_key=GEMINI_API_KEY)

  def embed_content(self, contents: List[str]) -> List[List[float]]:
    BATCH_SIZE = 100
    MAX_RETRIES = 5
    INITIAL_DELAY_MS = 1000  # 1 second
    DELAY_MULTIPLIER = 2  # Double delay on each retry

    embeddings = []

    for i in range(0, len(contents), BATCH_SIZE):
      batch = contents[i : i + BATCH_SIZE]
      if not batch:
        continue

      retries = 0
      delay = INITIAL_DELAY_MS
      success = False

      while not success and retries < MAX_RETRIES:
        try:
          responses = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=batch,  # type: ignore
            config={"task_type": "RETRIEVAL_QUERY"},
          )

          if not responses.embeddings:
            raise ValueError("No embeddings found in response")

          embedding_batch = [embedding.values for embedding in responses.embeddings]
          embeddings.extend(embedding_batch)
          success = True

        except Exception as error:
          retries += 1
          if retries >= MAX_RETRIES:
            print(f"Failed after {MAX_RETRIES} retries:", error)
            raise error

          print(
            f"Embedding API call failed (attempt {retries}/{MAX_RETRIES}). Retrying in {delay}ms..."
          )
          time.sleep(delay / 1000)
          delay *= DELAY_MULTIPLIER  # Increase delay for next retry

    return embeddings

  async def prompt(self, contents) -> AsyncGenerator[str, None]:
    async for chunk in await self.client.aio.models.generate_content_stream(
      model="gemini-2.5-pro-exp-03-25", contents=contents
    ):
      if chunk.text:
        yield chunk.text

  def count_tokens(self, contents) -> int:
    response = self.client.models.count_tokens(
      model="gemini-2.0-flash", contents=contents
    )
    if not response.total_tokens:
      raise ValueError("No token count found in response")
    return response.total_tokens
