import urllib.parse
import re
import requests
from bs4 import BeautifulSoup
import newspaper

BLOCKED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "::1"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def extract_fallback_from_slug(url: str, parsed_url):
    domain = parsed_url.netloc or "Unknown Domain"
    path_parts = [p for p in parsed_url.path.split('/') if p]
    slug = path_parts[-1] if path_parts else ""
    # Clean up trailing numbers or extension like .html, -12144457
    slug = re.sub(r'(\.html|\.htm|-\d+)$', '', slug, flags=re.IGNORECASE)
    words = [w.capitalize() for w in slug.split('-') if w and not w.isdigit()]
    title = ' '.join(words) if words else f"News Article from {domain}"
    
    return {
        "title": title,
        "text": f"{title}. Article headline extracted from URL. (Note: Direct scraping was protected by {domain}).",
        "authors": [domain],
        "publish_date": None,
        "domain": domain,
        "is_fallback": True
    }

def scrape_article(url: str):
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return {"error": "Invalid URL scheme. Only http and https URLs are allowed."}

        hostname = (parsed.hostname or "").lower()
        if hostname in BLOCKED_HOSTS or hostname.startswith("192.168.") or hostname.startswith("10.") or hostname.startswith("172.16."):
            return {"error": "Access to internal or private IP addresses is blocked for security reasons (SSRF Protection)."}

        # --- TIER 1: newspaper3k with Config ---
        try:
            config = newspaper.Config()
            config.browser_user_agent = USER_AGENTS[0]
            config.request_timeout = 8
            
            article = newspaper.Article(url, config=config)
            article.download()
            article.parse()

            if article.title and (article.text and len(article.text) > 30):
                return {
                    "title": article.title,
                    "text": article.text,
                    "authors": article.authors if hasattr(article, "authors") else [],
                    "publish_date": str(article.publish_date) if article.publish_date else None,
                    "domain": parsed.netloc
                }
        except Exception:
            pass  # Proceed to Tier 2

        # --- TIER 2: Requests + BeautifulSoup ---
        try:
            headers = {
                "User-Agent": USER_AGENTS[0],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }
            resp = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Extract Title
                title = None
                og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
                if og_title and og_title.get("content"):
                    title = og_title["content"]
                elif soup.title and soup.title.string:
                    title = soup.title.string.strip()
                elif soup.find("h1"):
                    title = soup.find("h1").get_text().strip()

                # Extract Text from Paragraphs
                paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 25]
                text = " ".join(paragraphs[:10])

                if title and len(text) > 30:
                    return {
                        "title": title,
                        "text": text,
                        "authors": [parsed.netloc],
                        "publish_date": None,
                        "domain": parsed.netloc
                    }
        except Exception:
            pass  # Proceed to Tier 3

        # --- TIER 3: URL Slug Extraction Fallback ---
        return extract_fallback_from_slug(url, parsed)

    except Exception as e:
        # Ultimate fallback rather than breaking UI
        parsed_err = urllib.parse.urlparse(url)
        return extract_fallback_from_slug(url, parsed_err)
