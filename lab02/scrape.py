"""
Lab 02 - Web scraping helper
Introduction to Artificial Intelligence (AI-101L)

Fetches blog articles (URL, title, body text) and saves them to a CSV file.
Run it directly (python scrape.py) or import scrape_articles() from a notebook.
"""

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ARTICLE_URLS = [
    "https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html",
    "https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html",
    "https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html",
    "https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html",
    "https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html",
]

BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MIN_TEXT_LENGTH = 100


def fetch_article(url):
    """Download one page and return {'url', 'title', 'text'}, or None if unusable."""
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
    response.raise_for_status()
    page = BeautifulSoup(response.content, "lxml")

    # Blogger puts the headline in <h3 class="post-title">; fall back to <title>.
    heading = page.select_one("h3.post-title")
    title = heading.get_text(strip=True) if heading else page.title.get_text(strip=True)

    # The article body lives in <div class="post-body entry-content">.
    body = page.select_one("div.post-body.entry-content")
    if body is None:
        return None

    text = " ".join(p.get_text(strip=True) for p in body.find_all("p"))
    if len(text) < MIN_TEXT_LENGTH:
        return None
    return {"url": url, "title": title, "text": text}


def scrape_articles(urls=ARTICLE_URLS, pause=1.0):
    """Scrape every URL politely and return the results as a DataFrame."""
    rows = []
    for url in urls:
        try:
            row = fetch_article(url)
        except requests.RequestException as error:
            print("[ERROR]", url, "->", error)
            continue
        if row is None:
            print("[SKIP] no usable article body:", url)
        else:
            rows.append(row)
        time.sleep(pause)
    return pd.DataFrame(rows, columns=["url", "title", "text"])


if __name__ == "__main__":
    articles = scrape_articles()
    articles.to_csv("articles_raw.csv", index=False)
    print("Saved", len(articles), "articles to articles_raw.csv")
