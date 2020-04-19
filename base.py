from pydantic import BaseModel


class NewsUrl(BaseModel):
    url: str
    category: str


class NewsContent(BaseModel):
    content: str
    category: str
