import re

def extract_claims(text: str):
    """
    Extracts key factual statements/claims from news text, article body, or OCR.
    """
    if not text or not text.strip():
        return []

    # Clean text
    clean = text.replace("\r", " ").replace("\n", " ").strip()
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean) if len(s.strip()) > 15]

    claims = []
    # Identify factual or strong statement sentences
    claim_triggers = [
        "discovered", "found", "announced", "claimed", "proved", "cures",
        "causes", "reported", "breaking", "confirmed", "revealed", "banned",
        "declared", "passed", "increased", "decreased", "invented", "died", "won"
    ]

    for sentence in sentences:
        lower_s = sentence.lower()
        if any(trigger in lower_s for trigger in claim_triggers) or len(sentence.split()) >= 6:
            claims.append(sentence[:160])
            if len(claims) >= 3:
                break

    if not claims and sentences:
        claims = [sentences[0][:160]]

    return claims
