from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from scraper import scrape_article
from ml_service import analyze_text_with_ai
from database import get_db
import models

app = FastAPI(title="Verity AI Platform API")

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
    overall_risk: int
    clickbait: int
    source_reliability: int
    emotional_language: int
    evidence_quality: int
    verdict: str

@app.post("/api/analyze/text", response_model=AnalysisResponse)
def analyze_text(request: TextRequest, db: Session = Depends(get_db)):
    result = analyze_text_with_ai(request.text)
    # Log to DB (dummy user id 1 for MVP)
    history = models.AnalysisHistory(
        user_id=1,
        content_type="text",
        content=request.text,
        is_fake=result["is_fake"],
        confidence_score=result["confidence"],
        explanation=result["explanation"]
    )
    db.add(history)
    db.commit()
    return result

@app.post("/api/analyze/url", response_model=AnalysisResponse)
def analyze_url(request: UrlRequest, db: Session = Depends(get_db)):
    article_data = scrape_article(request.url)
    if "error" in article_data:
        raise HTTPException(status_code=400, detail="Could not scrape URL")
    
    result = analyze_text_with_ai(article_data["text"])
    
    history = models.AnalysisHistory(
        user_id=1,
        content_type="url",
        content=request.url,
        is_fake=result["is_fake"],
        confidence_score=result["confidence"],
        explanation=result["explanation"]
    )
    db.add(history)
    db.commit()
    return result

@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    total = db.query(models.AnalysisHistory).count()
    fake = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.is_fake == True).count()
    return {
        "total_analyzed": total,
        "fake_detected": fake,
        "real_detected": total - fake
    }
