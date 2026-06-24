import newspaper

def scrape_article(url: str):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date
        }
    except Exception as e:
        return {"error": str(e)}
