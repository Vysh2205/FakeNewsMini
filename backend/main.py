from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

class AnalysisResponse(BaseModel):
    is_fake: bool
    confidence: float
    keywords: list[str]
    explanation: str

@app.get("/")
def read_root():
    return {"status": "Fake News API is running"}

@app.post("/api/analyze/text", response_model=AnalysisResponse)
def analyze_text(request: TextRequest):
    # Mock ML implementation for MVP
    # In a real scenario, this would call DistilBERT
    is_fake = random.choice([True, False])
    confidence = round(random.uniform(0.7, 0.99), 2)
    
    keywords = ["sensational", "unverified", "shocking"] if is_fake else ["verified", "factual", "reported"]
    explanation = (
        f"The text exhibits patterns commonly associated with {'fake' if is_fake else 'reliable'} news. "
        "Specific emotional keywords and lack of cited sources contributed to this score."
    ) if is_fake else "The text appears objective and aligns with reliable reporting structures."

    return AnalysisResponse(
        is_fake=is_fake,
        confidence=confidence,
        keywords=keywords,
        explanation=explanation
    )

@app.post("/api/analyze/url", response_model=AnalysisResponse)
def analyze_url(request: UrlRequest):
    # Mock URL scraping
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    is_fake = random.choice([True, False])
    confidence = round(random.uniform(0.7, 0.99), 2)
    
    return AnalysisResponse(
        is_fake=is_fake,
        confidence=confidence,
        keywords=["clickbait", "ad-heavy"] if is_fake else ["trusted domain"],
        explanation="Extracted article content analyzed."
    )
