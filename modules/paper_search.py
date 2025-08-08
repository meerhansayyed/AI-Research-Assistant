import requests
import xml.etree.ElementTree as ET

def search_arxiv(query, max_results=5):
    """
    Search arXiv for papers related to the given query.
    Returns a list of dictionaries with title, summary, and link.
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return [{"error": f"API request failed with status {response.status_code}"}]
    
    root = ET.fromstring(response.text)
    namespace = {"arxiv": "http://www.w3.org/2005/Atom"}
    papers = []
    
    for entry in root.findall("arxiv:entry", namespace):
        title = entry.find("arxiv:title", namespace).text.strip()
        summary = entry.find("arxiv:summary", namespace).text.strip()
        link = entry.find("arxiv:id", namespace).text.strip()
        
        papers.append({
            "title": title,
            "summary": summary,
            "link": link
        })
    
    return papers
