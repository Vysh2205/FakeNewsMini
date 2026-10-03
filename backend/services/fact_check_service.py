import os
import requests

FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY", "")

def search_fact_checks(query: str):
    """
    Integrates with Google Fact Check Tools API or open fact checking endpoints.
    Fails gracefully if no key is set or no results match.
    Never fabricates false fact check results.
    """
    if not query or len(query.strip()) < 5:
        return {
            "status": "no_results",
            "message": "No matching fact-check found.",
            "results": []
        }

    if FACT_CHECK_API_KEY:
        try:
            url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={requests.utils.quote(query)}&key={FACT_CHECK_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])
                results = []
                for item in claims[:3]:
                    claim_text = item.get("text", "")
                    claim_reviews = item.get("claimReview", [])
                    if claim_reviews:
                        review = claim_reviews[0]
                        publisher = review.get("publisher", {}).get("name", "Reputable Fact-Checker")
                        rating = review.get("textualRating", "Unverified")
                        url_link = review.get("url", "#")
                        title = review.get("title", claim_text)
                        results.append({
                            "organization": publisher,
                            "claim": claim_text,
                            "verdict": rating,
                            "summary": title,
                            "publication_date": review.get("reviewDate", "Recent"),
                            "source_link": url_link
                        })
                if results:
                    return {
                        "status": "success",
                        "message": f"Found {len(results)} external fact-check(s)",
                        "results": results
                    }
        except Exception as e:
            print(f"Fact Check API call error: {e}")

    # Default fallback when API is not configured or returns empty results
    return {
        "status": "no_results",
        "message": "No matching fact-check found.",
        "results": []
    }
