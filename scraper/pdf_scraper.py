import os
import time
import requests
from urllib.parse import urlparse
import re

class ScholarScraper:
    def __init__(self, api_url, save_dir="data"):
        self.api_url = api_url
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # Headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://scholar.google.com/'
        }

    def fetch_json(self):
        """Fetch a single JSON page from self.api_url.

        This keeps backward compatibility with existing code.
        """
        if not self.api_url:
            raise ValueError("API URL not provided")

        response = requests.get(self.api_url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def fetch_all_json(self, start_url=None, max_pages=None, sleep=1.0):
        """Fetch and aggregate results across paginated JSON pages.

        Behavior:
        - If `start_url` is provided it will be used as the first request URL. Otherwise `self.api_url` is used.
        - The method looks for `serpapi_pagination.next` or `pagination.next` in the returned JSON to follow pages.
        - Aggregates `organic_results` and removes duplicates by `result_id` if present.
        - `max_pages` can limit the number of pages to fetch (None = unlimited until no next link).
        - `sleep` is seconds to wait between requests to avoid aggressive scraping.

        Returns a dict similar to a single-page response but with combined `organic_results`.
        """
        url = start_url or self.api_url
        if not url:
            raise ValueError("No start URL or api_url provided to fetch pages from")

        combined = None
        seen_ids = set()
        pages_fetched = 0

        while url:
            if max_pages is not None and pages_fetched >= max_pages:
                break

            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

            if combined is None:
                # Start with the first page metadata
                combined = dict(data)
                combined['organic_results'] = []
            # Merge organic results, avoid duplicates
            for item in data.get('organic_results', []):
                rid = item.get('result_id') or item.get('link') or item.get('title')
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                combined['organic_results'].append(item)

            pages_fetched += 1

            # Determine next link from serpapi_pagination or pagination
            next_url = None
            serpapi_pag = data.get('serpapi_pagination') or {}
            if serpapi_pag.get('next'):
                next_url = serpapi_pag.get('next')
            else:
                pag = data.get('pagination') or {}
                next_url = pag.get('next')

            # If next_url is present but looks like a relative or HTML page, stop
            if not next_url:
                break

            # Respect a short delay to be polite
            time.sleep(sleep)
            url = next_url

        if combined is None:
            return {}
        return combined

    def download_pdfs(self, data):
        papers = data.get("organic_results", [])
        total_papers = len(papers)
        pdf_count = 0
        
        print(f"📚 Found {total_papers} papers. Checking for PDFs...\n")
        
        for idx, paper in enumerate(papers, 1):
            title = paper.get("title", f"Untitled_{idx}")
            # Clean title for filename
            title = re.sub(r'[<>:"/\\|?*]', '_', title)
            title = title[:100]  # Limit filename length
            
            resources = paper.get("resources", [])
            if not resources:
                print(f"⏭️  [{idx}/{total_papers}] {title[:60]}... - No PDF resources found")
                continue
            
            pdf_found = False
            for res in resources:
                if res.get("file_format", "").lower() == "pdf":
                    pdf_link = res.get("link", "")
                    if pdf_link:
                        pdf_found = True
                        if self._save_pdf(pdf_link, title, idx, total_papers):
                            pdf_count += 1
                        break
            
            if not pdf_found:
                print(f"⏭️  [{idx}/{total_papers}] {title[:60]}... - No PDF link found")
        
        print(f"\n📊 Summary: Successfully downloaded {pdf_count} PDF(s) out of {total_papers} papers")

    def _save_pdf(self, link, title, paper_num, total):
        try:
            print(f"⬇️  [{paper_num}/{total}] Downloading: {title[:60]}...")
            print(f"    🔗 Link: {link[:80]}...")
            
            # Make request with headers and allow redirects
            session = requests.Session()
            session.headers.update(self.headers)
            
            r = session.get(link, timeout=30, allow_redirects=True, stream=True)
            r.raise_for_status()
            
            # Check if we got redirected to a login/access page (common for academic publishers)
            final_url = r.url
            if final_url != link:
                print(f"    ⚠️  Redirected to: {final_url[:80]}...")
            
            # Check content type
            content_type = r.headers.get("content-type", "").lower()
            
            # Read first chunk to check PDF signature and detect HTML
            first_chunk = next(r.iter_content(chunk_size=1024), None)
            if first_chunk is None:
                print(f"    ❌ Failed: Empty response\n")
                return False
            
            # Check if response is HTML (likely a login/access page)
            if "text/html" in content_type:
                # Check if it's actually HTML by looking at content
                content_start = first_chunk[:100].decode('utf-8', errors='ignore').lower()
                if '<html' in content_start or '<!doctype' in content_start:
                    print(f"    ❌ Failed: Redirected to HTML page (may require authentication/access)\n")
                    return False
            
            # Check PDF magic bytes
            is_pdf = first_chunk[:4] == b'%PDF'
            
            if "application/pdf" in content_type or is_pdf:
                path = os.path.join(self.save_dir, f"{title}.pdf")
                with open(path, "wb") as f:
                    # Write the first chunk we already read
                    f.write(first_chunk)
                    # Write the rest
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(path) / 1024  # Size in KB
                detection_method = "Content-Type" if "application/pdf" in content_type else "Magic bytes"
                print(f"    ✅ Saved: {title}.pdf ({file_size:.1f} KB) [{detection_method}]\n")
                return True
            else:
                preview = first_chunk[:50].decode('utf-8', errors='ignore').replace('\n', ' ').replace('\r', ' ')[:50]
                print(f"    ❌ Failed: Not a PDF (Content-Type: {content_type}, Preview: {preview}...)\n")
                return False
                    
        except requests.exceptions.Timeout:
            print(f"    ❌ Download failed: Request timeout\n")
            return False
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Download failed: {str(e)}\n")
            return False
        except Exception as e:
            print(f"    ❌ Error: {str(e)}\n")
            return False
