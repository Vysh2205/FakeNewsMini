import urllib.parse
import newspaper

BLOCKED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "::1"]

def scrape_article(url: str):
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return {"error": "Invalid URL scheme. Only http and https URLs are allowed."}

        hostname = (parsed.hostname or "").lower()
        if hostname in BLOCKED_HOSTS or hostname.startswith("192.168.") or hostname.startswith("10.") or hostname.startswith("172.16."):
            return {"error": "Access to internal or private IP addresses is blocked for security reasons (SSRF Protection)."}

        article = newspaper.Article(url)
        article.download()
        article.parse()
        
        return {
            "title": article.title or parsed.netloc,
            "text": article.text if article.text else article.title,
            "authors": article.authors if hasattr(article, "authors") else [],
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "domain": parsed.netloc
        }
    except Exception as e:
        return {"error": f"Failed to retrieve article from URL: {str(e)}"}
