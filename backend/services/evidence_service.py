import urllib.parse
import requests
from bs4 import BeautifulSoup

def search_evidence(query: str):
    """
    Searches web news sources for relevant evidence supporting or contradicting claims.
    Categorizes evidence into Supporting, Contradicting, or Related.
    Fails gracefully if no evidence can be retrieved.
    """
    if not query or len(query.strip()) < 5:
        return {
            "status": "no_evidence",
            "message": "No relevant evidence found.",
            "evidence": []
        }

    evidence_items = []
    try:
        # Search via DuckDuckGo HTML endpoint with safe headers
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(search_url, headers=headers, timeout=4)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = soup.find_all('div', class_='result')
            
            for res in results[:4]:
                title_tag = res.find('a', class_='result__a')
                snippet_tag = res.find('a', class_='result__snippet')
                url_tag = res.find('a', class_='result__url')
                
                if title_tag:
                    title = title_tag.text.strip()
                    snippet = snippet_tag.text.strip() if snippet_tag else "No snippet available."
                    url = title_tag.get('href', '#')
                    domain = url_tag.text.strip() if url_tag else "Web Source"
                    
                    # Categorize based on stance wording
                    snip_lower = (title + " " + snippet).lower()
                    if any(w in snip_lower for w in ["false", "fake", "debunked", "misleading", "myth"]):
                        category = "Contradicting"
                    elif any(w in snip_lower for w in ["confirmed", "official", "verified", "true", "report"]):
                        category = "Supporting"
                    else:
                        category = "Related"

                    evidence_items.append({
                        "source": domain,
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "category": category,
                        "relevance": "High" if category != "Related" else "Moderate"
                    })
    except Exception as e:
        print(f"Evidence search error: {e}")

    if evidence_items:
        return {
            "status": "success",
            "message": f"Retrieved {len(evidence_items)} relevant evidence source(s)",
            "evidence": evidence_items
        }
    
    return {
        "status": "no_evidence",
        "message": "No relevant evidence found.",
        "evidence": []
    }
