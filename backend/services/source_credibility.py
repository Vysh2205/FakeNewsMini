import urllib.parse

KNOWN_SOURCES = {
    "reuters.com": {"name": "Reuters", "rating": "High Credibility", "score": 95, "category": "International News Agency"},
    "apnews.com": {"name": "Associated Press", "rating": "High Credibility", "score": 96, "category": "Wire Service"},
    "bbc.com": {"name": "BBC News", "rating": "High Credibility", "score": 92, "category": "Public Broadcaster"},
    "nytimes.com": {"name": "The New York Times", "rating": "High Credibility", "score": 88, "category": "Major Newspaper"},
    "theonion.com": {"name": "The Onion", "rating": "Satire / Parody", "score": 10, "category": "Satirical Publication"},
    "infowars.com": {"name": "InfoWars", "rating": "Low Credibility", "score": 15, "category": "Conspiracy / Unverified"}
}

def evaluate_source(url_or_domain: str):
    """
    Evaluates source credibility based on domain analysis.
    For unknown sources, clearly returns 'Source credibility information unavailable'.
    Never automatically marks unknown sources as unreliable.
    """
    if not url_or_domain:
        return {
            "domain": "Unknown",
            "source_name": "Unknown Source",
            "credibility_status": "Source credibility information unavailable.",
            "reliability_score": 50,
            "category": "Unindexed Domain"
        }

    # Extract domain
    clean = url_or_domain.lower().replace("http://", "").replace("https://", "").replace("www.", "")
    domain = clean.split("/")[0]

    for known_domain, info in KNOWN_SOURCES.items():
        if known_domain in domain:
            return {
                "domain": domain,
                "source_name": info["name"],
                "credibility_status": f"{info['rating']} ({info['category']})",
                "reliability_score": info["score"],
                "category": info["category"]
            }

    return {
        "domain": domain if domain else "Unknown",
        "source_name": domain.capitalize() if domain else "External Source",
        "credibility_status": "Source credibility information unavailable.",
        "reliability_score": 50,
        "category": "General Web Domain"
    }
