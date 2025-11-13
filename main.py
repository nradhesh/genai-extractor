import json
import sys
from scraper.pdf_scraper import ScholarScraper

if __name__ == "__main__":
    # Option 1: Use API URL
    API_URL = "https://serpapi.com/search.json?engine=google_scholar&q=%22Jamuna%20S%20Murthy%22&hl=en&api_key=14c9957c14272f34073d41bd7a4820e7bd21d251fa178b4eb320275ce2a0a7cd"
    
    # Option 2: Load from JSON file (if provided as argument)
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        print(f"📂 Loading JSON from file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scraper = ScholarScraper("")  # No API URL needed when loading from file
    else:
        # Fetch from API and follow pagination to cover all pages
        scraper = ScholarScraper(API_URL)
        try:
            data = scraper.fetch_all_json()
        except Exception as e:
            print(f"⚠️  Error fetching paginated results: {e}\nFalling back to single-page fetch.")
            data = scraper.fetch_json()
    
    scraper.download_pdfs(data)

    print("\n🎯 All available PDFs saved in the 'data/' folder.")
