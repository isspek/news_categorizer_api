from pathlib import Path

from fastapi import FastAPI

from base import NewsContent, NewsUrl
from categorizer import get_category_from_content, get_category_from_url

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/url/get_category/{url}", response_model=NewsUrl)
async def get_category_from_url(url: str = Path(..., title="Url of the news for categorization")):
    return get_category_from_url(url)


@app.get("/content/get_category/{content}", response_model=NewsContent)
async def get_category_from_content(content: str = Path(..., content="Content of the news for categorization")):
    return get_category_from_content(content)
