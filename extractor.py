import time
import random
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def get_all_paper_links(profile_url):
    options = Options()
    # For debugging, keep headless commented:
    # options.add_argument("--headless")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.get(profile_url)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "gsc_a_at")))
    while True:
        try:
            show_more = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "gsc_bpf_more")))
            if show_more.is_enabled() and show_more.is_displayed():
                driver.execute_script("arguments[0].click();", show_more)
                time.sleep(random.uniform(2, 4))
            else:
                break
        except Exception:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = []
    for a in soup.select('.gsc_a_at'):
        href = a.get('href')
        if href:
            links.append("https://scholar.google.com" + href)
    driver.quit()
    return links

def extract_paper_details(article_url):
    options = Options()
    # For debugging, keep headless commented:
    # options.add_argument("--headless")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.get(article_url)
    try:
        # Wait for up to 15 seconds for the title or for a CAPTCHA
        WebDriverWait(driver, 15).until(
            lambda d: d.find_elements(By.ID, "gsc_oci_title") or "robot" in d.page_source.lower() or "captcha" in d.page_source.lower()
        )
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # CAPTCHA Detection
        if "robot" in html.lower() or "captcha" in html.lower():
            print("CAPTCHA detected! Complete CAPTCHA in browser and press Enter to continue...")
            input("Press Enter when done...")
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

        def get(label):
            divs = soup.find_all('div', class_='gsc_oci_field')
            for div in divs:
                if div.text.strip() == label:
                    sib = div.find_next_sibling('div', class_='gsc_oci_value')
                    return sib.text.strip() if sib else ""
            return ""
        article_title = soup.find('div', id='gsc_oci_title')
        article_title = article_title.text.strip() if article_title else ''
        citation_link = soup.find('a', string=lambda s: s and 'Cited by' in s)
        total_citations = citation_link.text if citation_link else ""
        data = {
            "Scholar articles": article_title,
            "Total citations": total_citations,
            "Description": get('Description'),
            "Pages": get('Pages'),
            "Publisher": get('Publisher'),
            "Conference": get('Conference'),
            "Publication date": get('Publication date'),
            "Authors": get('Authors')
        }
    except Exception as e:
        print(f"Failed to extract: {article_url}\nError: {e}")
        data = {
            "Scholar articles": "ERROR",
            "Total citations": "ERROR",
            "Description": "",
            "Pages": "",
            "Publisher": "",
            "Conference": "",
            "Publication date": "",
            "Authors": ""
        }
    driver.quit()
    # Sleep a random interval (to reduce bot detection)
    time.sleep(random.uniform(4, 8))
    return data

author_profile = "https://scholar.google.com/citations?user=geX_RLEAAAAJ&hl=en"

print(f"Getting paper links from: {author_profile}")
paper_urls = get_all_paper_links(author_profile)
print(f"Found {len(paper_urls)} papers.")

with open('scholar_articles.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ["Scholar articles", "Total citations", "Description", "Pages", "Publisher", "Conference", "Publication date", "Authors"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, url in enumerate(paper_urls):
        details = extract_paper_details(url)
        writer.writerow(details)
        print(f"Processed {i+1}/{len(paper_urls)}: {details['Scholar articles']}")
print("Extraction complete. Check scholar_articles.csv for results.")
