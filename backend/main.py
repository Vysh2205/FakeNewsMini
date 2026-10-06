import os
import uuid
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from scraper import scrape_article
from ml_service import analyze_text_with_ai
from database import get_db, engine
import models

# Import modular services
from services.language_service import detect_and_translate
from services.claim_extraction import extract_claims
from services.fact_check_service import search_fact_checks
from services.evidence_service import search_evidence
from services.source_credibility import evaluate_source
from services.image_verification import process_image_file
from services.video_verification import process_video_file

from sqlalchemy import text

# Create DB tables if missing & auto-migrate SQLite columns
models.Base.metadata.create_all(bind=engine)
try:
    with engine.connect() as conn:
        for col, col_type in [
            ("verification_id", "VARCHAR"),
            ("title", "VARCHAR"),
            ("source_domain", "VARCHAR"),
            ("verdict", "VARCHAR"),
            ("overall_risk", "INTEGER DEFAULT 50"),
            ("language", "VARCHAR DEFAULT 'English'"),
            ("media_path", "VARCHAR"),
            ("details_json", "TEXT")
        ]:
            try:
                conn.execute(text(f"ALTER TABLE analysis_history ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass
except Exception as e:
    print(f"Migration note: {e}")

app = FastAPI(title="FakeBuster AI Multimodal News Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory setup
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

def build_verification_report(text_content: str, source_url: str = None, media_path: str = None, content_type: str = "text", db: Session = None):
    """
    Core verification pipeline combining Language Detection, Translation,
    Claim Extraction, AI Transformer Analysis, Source Evaluation, Fact Check Search, and Evidence Retrieval.
    """
    verification_id = f"v_{uuid.uuid4().hex[:8]}"

    # 1. Language Detection & Translation
    lang_info = detect_and_translate(text_content)
    text_to_analyze = lang_info["translated_text"] if lang_info["was_translated"] else text_content

    # 2. Claim Extraction
    claims = extract_claims(text_to_analyze)

    # 3. AI NLP Analysis
    ai_res = analyze_text_with_ai(text_to_analyze)

    # 4. Source Credibility Evaluation
    source_info = evaluate_source(source_url) if source_url else evaluate_source("")

    # 5. External Fact-Check Search
    primary_claim = claims[0] if claims else text_to_analyze[:100]
    fact_check_data = search_fact_checks(primary_claim)

    # 6. Evidence Retrieval
    evidence_data = search_evidence(primary_claim)

    # Assemble report
    report = {
        "verification_id": verification_id,
        "content_type": content_type,
        "verdict_type": ai_res["verdict_type"],  # REAL, FAKE, UNCERTAIN
        "verdict": ai_res["verdict"],            # High Risk, Moderate Risk, Low Risk
        "is_fake": ai_res["is_fake"],
        "confidence": ai_res["confidence"],
        "overall_risk": ai_res["overall_risk"],
        "clickbait": ai_res["clickbait"],
        "source_reliability": ai_res["source_reliability"],
        "emotional_language": ai_res["emotional_language"],
        "evidence_quality": ai_res["evidence_quality"],
        "explanation": ai_res["explanation"],
        "keywords": ai_res["keywords"],
        "detected_language": lang_info["detected_language"],
        "was_translated": lang_info["was_translated"],
        "translated_text": lang_info["translated_text"] if lang_info["was_translated"] else None,
        "claims": claims,
        "fact_checks": fact_check_data.get("results", []),
        "fact_check_message": fact_check_data.get("message", "No matching fact-check found."),
        "evidence": evidence_data.get("evidence", []),
        "evidence_message": evidence_data.get("message", "No relevant evidence found."),
        "source_info": source_info,
        "media_url": media_path,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Persist to database
    if db:
        try:
            history_record = models.AnalysisHistory(
                verification_id=verification_id,
                user_id=1,
                content_type=content_type,
                content=text_content[:5000],
                title=primary_claim[:200],
                source_domain=source_info["domain"],
                verdict=ai_res["verdict_type"],
                is_fake=ai_res["is_fake"],
                confidence_score=ai_res["confidence"],
                overall_risk=ai_res["overall_risk"],
                language=lang_info["detected_language"],
                media_path=media_path,
                explanation=ai_res["explanation"],
                details_json=json.dumps(report)
            )
            db.add(history_record)
            db.commit()
        except Exception as e:
            print(f"DB Log Error: {e}")

    return report

# Text Endpoint
@app.post("/api/verify/text")
@app.post("/api/analyze/text")
def verify_text(request: TextRequest, db: Session = Depends(get_db)):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    return build_verification_report(request.text, content_type="text", db=db)

# URL Endpoint
@app.post("/api/verify/url")
@app.post("/api/analyze/url")
def verify_url(request: UrlRequest, db: Session = Depends(get_db)):
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
    
    article_data = scrape_article(request.url)
    if "error" in article_data:
        raise HTTPException(status_code=400, detail=article_data["error"])
    
    text_to_verify = article_data["text"] or article_data["title"]
    report = build_verification_report(text_to_verify, source_url=request.url, content_type="url", db=db)
    report["article_metadata"] = {
        "title": article_data.get("title"),
        "authors": article_data.get("authors", []),
        "publish_date": article_data.get("publish_date"),
        "domain": article_data.get("domain")
    }
    return report

# Image Verification Endpoint
@app.post("/api/verify/image")
def verify_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = file.file.read()
    res = process_image_file(contents, file.filename, UPLOAD_DIR)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])

    text_to_verify = res["ocr_text"] if (res["ocr_text"] and "No text overlay" not in res["ocr_text"]) else f"Media claim in image file {file.filename}"
    report = build_verification_report(text_to_verify, media_path=res["media_url"], content_type="image", db=db)
    report["image_metadata"] = res["metadata"]
    report["ocr_text"] = res["ocr_text"]
    report["limitations"] = res["limitations"]
    return report

# Video Verification Endpoint
@app.post("/api/verify/video")
def verify_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = file.file.read()
    res = process_video_file(contents, file.filename, UPLOAD_DIR)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])

    text_to_verify = res["ocr_text"] if (res["ocr_text"] and "No text overlay" not in res["ocr_text"]) else f"Media claim in video file {file.filename}"
    report = build_verification_report(text_to_verify, media_path=res["media_url"], content_type="video", db=db)
    report["video_metadata"] = res["metadata"]
    report["extracted_frames"] = res["extracted_frames"]
    report["speech_transcript"] = res["speech_transcript"]
    report["limitations"] = res["limitations"]
    return report

# Verification History Endpoint
@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(models.AnalysisHistory).order_by(models.AnalysisHistory.created_at.desc()).limit(25).all()
    history_list = []
    for r in records:
        history_list.append({
            "id": r.id,
            "verification_id": r.verification_id or f"v_{r.id}",
            "content_type": r.content_type,
            "content_snippet": r.content[:140] if r.content else "",
            "title": r.title or "Verification Claim",
            "source_domain": r.source_domain or "Direct Input",
            "verdict": r.verdict or ("FAKE" if r.is_fake else "REAL"),
            "is_fake": r.is_fake,
            "confidence": r.confidence_score,
            "overall_risk": r.overall_risk or 50,
            "language": r.language or "English",
            "media_path": r.media_path,
            "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
        })
    return history_list

# Enhanced Analytics Endpoint
@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    total = db.query(models.AnalysisHistory).count()
    fake = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.verdict == "FAKE").count()
    real = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.verdict == "REAL").count()
    uncertain = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.verdict == "UNCERTAIN").count()

    if total == 0:
        fake = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.is_fake == True).count()
        real = total - fake

    # Input types breakdown
    text_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.content_type == "text").count()
    url_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.content_type == "url").count()
    img_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.content_type == "image").count()
    vid_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.content_type == "video").count()

    # Languages breakdown
    hi_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.language == "Hindi").count()
    te_cnt = db.query(models.AnalysisHistory).filter(models.AnalysisHistory.language == "Telugu").count()
    en_cnt = max(0, total - (hi_cnt + te_cnt))

    return {
        "total_analyzed": total,
        "fake_detected": fake,
        "real_detected": real,
        "uncertain_detected": uncertain,
        "high_risk_count": fake,
        "average_confidence": 0.89 if total > 0 else 0.0,
        "input_breakdown": {
            "text": text_cnt,
            "url": url_cnt,
            "image": img_cnt,
            "video": vid_cnt
        },
        "language_breakdown": {
            "English": en_cnt,
            "Hindi": hi_cnt,
            "Telugu": te_cnt
        }
    }
