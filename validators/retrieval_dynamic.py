import wikipedia
import feedparser

def retrieve_from_wikipedia(query, top_k=3):
    """Retrieve top-k Wikipedia summaries."""
    retrieved = []
    try:
        page_titles = wikipedia.search(query, results=top_k)
        for title in page_titles:
            try:
                summary = wikipedia.summary(title, sentences=2)
                retrieved.append(summary)
            except Exception:
                continue
    except Exception:
        pass
    return retrieved

def retrieve_from_arxiv(query, top_k=3):
    """Retrieve top-k Arxiv abstracts."""
    query = query.replace('\n', ' ').replace(' ', '+')
    base_url = "http://export.arxiv.org/api/query?search_query=all:"
    query_url = f"{base_url}{query}&start=0&max_results={top_k}"
    parsed = feedparser.parse(query_url)
    abstracts = [entry.summary for entry in parsed.entries]
    return abstracts
