from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    analyses = relationship("AnalysisHistory", back_populates="user")

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id = Column(Integer, primary_key=True, index=True)
    verification_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    content_type = Column(String) # 'text', 'url', 'image', 'video'
    content = Column(Text)
    title = Column(String, nullable=True)
    source_domain = Column(String, nullable=True)
    verdict = Column(String) # 'REAL', 'FAKE', 'UNCERTAIN'
    is_fake = Column(Boolean)
    confidence_score = Column(Float)
    overall_risk = Column(Integer, default=50)
    language = Column(String, default="English")
    media_path = Column(String, nullable=True)
    explanation = Column(Text)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="analyses")

class SourceCredibility(Base):
    __tablename__ = "source_credibility"
    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, unique=True, index=True)
    reliability_score = Column(Float) # 0.0 to 1.0
    notes = Column(Text)
