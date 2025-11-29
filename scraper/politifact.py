#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import re
import csv
import time
from datetime import datetime

BASE = "https://www.politifact.com"
LIST_URL = BASE + "/factchecks/list/"

HEADERS = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:145.0) Gecko/20100101 Firefox/145.0",
  "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.5",
  "Sec-GPC": "1",
  "Alt-Used": "www.politifact.com",
  "Connection": "keep-alive",
  "Upgrade-Insecure-Requests": "1",
  "Sec-Fetch-Dest": "document",
  "Sec-Fetch-Mode": "navigate",
  "Sec-Fetch-Site": "none",
  "Sec-Fetch-User": "?1",
  "Priority": "u=0, i",
}

VERDICT_MAP = {
  "meter-true": "True",
  "meter-mostly-true": "Mostly True",
  "meter-half-true": "Half True",
  "meter-barely-true": "Mostly False",
  "meter-false": "False",
  "meter-pants-fire": "Pants on Fire",
}

BATCH_SIZE = 1000

_month_day_year_re = re.compile(r"([A-Z][a-z]+ \d{1,2}, \s*\d{4})")


def extract_date_iso(text):
  if not text:
    return ""
  m = _month_day_year_re.search(text)
  if not m:
    m2 = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if not m2:
      return ""
    s = m2.group(1)
  else:
    s = m.group(1)
  try:
    dt = datetime.strptime(s.strip(), "%B %d, %Y")
    return dt.strftime("%Y-%m-%d")
  except Exception:
    return s.strip()


def extract_verdict_from_url(url_or_text):
  """Extract verdict from meter image URL or alt text."""
  url_lower = url_or_text.lower()
  for key, verdict in VERDICT_MAP.items():
    if key in url_lower:
      return verdict
  return ""


def parse_list_page_statements(session, url):
  """Parse fact-check statements directly from a list page."""
  r = session.get(url, timeout=20)
  r.raise_for_status()
  soup = BeautifulSoup(r.text, "html.parser")

  statements = []

  for article in soup.find_all("article", class_="m-statement"):
    try:
      verdict = ""
      meter_img = article.find("img", alt=True)
      if meter_img:
        alt_text = meter_img.get("alt", "").strip()  # type: ignore
        if alt_text:
          verdict = extract_verdict_from_url(alt_text)
        if not verdict:
          src = meter_img.get("src", "")
          if src:
            verdict = extract_verdict_from_url(src)

      statement_text = ""
      fact_url = ""
      quote_div = article.find("div", class_="m-statement__quote")
      if quote_div:
        statement_link = quote_div.find("a")
        if statement_link:
          statement_text = statement_link.get_text(strip=True)
          fact_url = statement_link.get("href", "")
          if fact_url and not fact_url.startswith("http"):  # type: ignore
            fact_url = BASE + fact_url  # type: ignore
        else:
          statement_text = quote_div.get_text(strip=True)

      statement_date = ""
      meta_div = article.find("div", class_="m-statement__desc")
      if meta_div:
        meta_text = meta_div.get_text(strip=True)
        date_match = re.search(r"stated on\s+([A-Z][a-z]+ \d{1,2}, \d{4})", meta_text)
        if date_match:
          statement_date = extract_date_iso(date_match.group(1))

      factcheck_date = ""
      footer = article.find("footer", class_="m-statement__footer")
      if footer:
        footer_text = footer.get_text(strip=True)
        date_match = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", footer_text)
        if date_match:
          factcheck_date = extract_date_iso(date_match.group(1))

      statement_source = ""
      source_link = article.find("a", class_="m-statement__name")
      if source_link:
        statement_source = source_link.get_text(strip=True)

      statement_text = statement_text.replace("“", '"').replace("”", '"')
      statement_text = statement_text.replace("‘", "'").replace("’", "'")
      statement_text = statement_text.replace("\u2013", "-").replace("\u2014", "-")
      statement_text = statement_text.replace("…", "...")

      if statement_text and verdict:
        statements.append(
          {
            "verdict": verdict,
            "statement": statement_text,
            "statement_date": statement_date,
            "statement_source": statement_source,
            "factcheck_date": factcheck_date,
            "url": fact_url,
          }
        )
    except Exception as e:
      print(f"Error parsing statement: {e}")
      continue

  return statements


def write_statements_to_csv(statements, output_csv):
  fieldnames = [
    "verdict",
    "statement",
    "statement_date",
    "statement_source",
    "factcheck_date",
    "url",
  ]

  with open(output_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if f.tell() == 0:
      writer.writeheader()
    for statement in statements:
      writer.writerow(statement)


def main(output_csv="politifact_statements.csv", max_pages=877):
  session = requests.Session()
  session.headers.update(HEADERS)

  all_statements = []

  print("Crawling PolitiFact list pages and extracting statements...")
  page = 1
  while page <= max_pages:
    try:
      list_url = LIST_URL if page == 1 else f"{LIST_URL}?page={page}"
      statements = parse_list_page_statements(session, list_url)

      if not statements:
        print(f"No statements on page {page} — stopping.")
        break

      all_statements.extend(statements)
      print(
        f"Page {page}: found {len(statements)} statements (total {len(all_statements)})"
      )
    except Exception as e:
      print(f"Error fetching list page {page}: {e}")
      break

    if len(all_statements) >= BATCH_SIZE:
      write_statements_to_csv(all_statements, output_csv)
      print(f"Saved {len(all_statements)} statements to {output_csv} so far.")
      all_statements = []

    page += 1
    time.sleep(1.0)

  if all_statements:
    write_statements_to_csv(all_statements, output_csv)

  print(f"Done — saved {len(all_statements)} statements to {output_csv}")


if __name__ == "__main__":
  main()
