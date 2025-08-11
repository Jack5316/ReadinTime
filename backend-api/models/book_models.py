from pydantic import BaseModel
from typing import Optional, Any, Union

class BookInfo(BaseModel):
    title: str
    author: str
    description: str
    folder: str
    cover: str

class BookUploadData(BaseModel):
    name: str
    pdfData: Optional[bytes] = None
    title: str
    author: str
    description: str
    bookPath: str

class TextSegment(BaseModel):
    text: str
    start: float
    end: float

class BookData(BaseModel):
    textMappings: list[TextSegment]

class Result(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None 