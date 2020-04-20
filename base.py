from pydantic import BaseModel


class NewsUrl(BaseModel):
    url: str
    category: str = None


class NewsContent(BaseModel):
    content: str
    category: str = None
