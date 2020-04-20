from fastapi import FastAPI
from base import NewsContent, NewsUrl
import categorizer
from loguru import logger

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/get_category/url/", response_model=NewsUrl)
async def get_category_from_url(news_url: NewsUrl):
    logger.info('Extracting category: {}'.format(news_url.url))
    news_url.category = categorizer.get_category_from_url(news_url.url)
    return news_url


@app.get("/get_category/content/{content}", response_model=NewsContent)
async def get_category_from_content(content: str):
    logger.info('Extracting category: {}'.format(content))
    return categorizer.get_category_from_content(content)
