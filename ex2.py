from serpapi import GoogleSearch
import csv
import time

AUTHOR_ID = "geX_RLEAAAAJ"  # M Mallegowda's author ID from the image
API_KEY = "14c9957c14272f34073d41bd7a4820e7bd21d251fa178b4eb320275ce2a0a7cd"
LANG = "en"
PAGE_SIZE = 100  # Request maximum items per page

def fetch_all_articles():
    all_articles = []
    next_cursor = None

    while True:
        params = {
            "engine": "google_scholar_author",
            "author_id": AUTHOR_ID,
            "api_key": API_KEY,
            "hl": LANG,
            "num": 100  # Request maximum number of results per page to get beyond 20
        }
        if next_cursor:
            params["cstart"] = next_cursor['cstart']
            params["pagesize"] = next_cursor['pagesize']

        search = GoogleSearch(params)
        results = search.get_dict()

        # Get author articles
        articles = results.get("articles", [])
        all_articles.extend(articles)

        # Check for next page
        next_cursor = results.get("next")
        if not next_cursor:
            break
        time.sleep(1)  # Polite delay to avoid any rate issue

    return all_articles

def fetch_article_details(article_id):
    params = {
        "engine": "google_scholar_cite",  # Use cite engine to get full citation details
        "q": article_id,  # Citation ID
        "api_key": API_KEY,
        "hl": LANG
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Get citation data from the first available format (if any)
    citations = results.get("citations", [])
    if not citations:
        return {}
    
    # Try to get the most detailed citation format (bibtex or MLA)
    citation = None
    for fmt in citations:
        if fmt.get("type") in ["bibtex", "mla"]:
            citation = fmt
            break
    if not citation and citations:
        citation = citations[0]
    
    # Extract clean text from citation
    citation_text = citation.get("snippet", "") if citation else ""
    
    # Parse citation details
    details = {}
    if "author" in citation_text.lower() or "authors" in citation_text.lower():
        details["authors"] = citation_text.split('"')[1].strip() if '"' in citation_text else ""
    if "pages" in citation_text.lower():
        try:
            details["pages"] = citation_text.split("pages")[1].split(",")[0].strip()
        except:
            details["pages"] = ""
    if "publisher" in citation_text.lower():
        try:
            details["publisher"] = citation_text.split("publisher")[1].split(",")[0].strip()
        except:
            details["publisher"] = ""
    if "journal" in citation_text.lower():
        try:
            details["journal"] = citation_text.split("journal")[1].split(",")[0].strip()
        except:
            details["journal"] = ""
    if "conference" in citation_text.lower():
        try:
            details["conference"] = citation_text.split("conference")[1].split(",")[0].strip()
        except:
            details["conference"] = ""
            
    return details

def main():
    articles = fetch_all_articles()
    print(f"Found {len(articles)} articles for author.")
    
    # Create a unique filename using timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scholar_articles_{timestamp}.csv"
    
    print(f"\nSaving results to: {filename}")
    try:
        with open(filename, "w", newline='', encoding="utf-8") as file:
            fieldnames = [
                "title", "authors", "journal", "year", "description", 
                "pages", "publisher", "conference", "total_citations"
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for i, art in enumerate(articles, 1):
                print(f"\nProcessing article {i}/{len(articles)}:")
                print(f"Title: {art.get('title')}")
                
                # Get basic article info
                result_id = art.get("citation_id")
                if not result_id:
                    # Try to extract from cite link
                    cite_link = art.get("inline_links", {}).get("serpapi_cite_link", "")
                    if cite_link and "q=" in cite_link:
                        result_id = cite_link.split("q=")[-1]
                
                # Fetch detailed citation
                details = fetch_article_details(result_id) if result_id else {}
                
                # Combine article data with citation details
                row = {
                    "title": art.get("title", ""),
                    "authors": details.get("authors", art.get("authors", "")),
                    "journal": details.get("journal", art.get("publication", "")),
                    "year": art.get("year", ""),
                    "description": art.get("snippet", ""),
                    "pages": details.get("pages", ""),
                    "publisher": details.get("publisher", ""),
                    "conference": details.get("conference", ""),
                    "total_citations": art.get("cited_by", {}).get("value", "")
                }
                
                # Clean up any None values
                for key in row:
                    if row[key] is None:
                        row[key] = ""
                    elif isinstance(row[key], dict):
                        row[key] = str(row[key])
                    elif isinstance(row[key], list):
                        row[key] = ", ".join(str(x) for x in row[key])
                
                writer.writerow(row)
                print(f"✓ Saved with {sum(1 for v in row.values() if v)} fields populated")
                time.sleep(1)  # Polite delay
                
    except PermissionError:
        print(f"\n❌ Error: Cannot write to {filename}")
        print("Please ensure:")
        print("1. The file is not open in another program (like Excel)")
        print("2. You have write permissions in this folder")
        print(f"3. Try closing any programs that might have {filename} open")
        raise
    
    print(f"\n✅ Successfully saved {len(articles)} articles to {filename}")


if __name__ == "__main__":
    main()
