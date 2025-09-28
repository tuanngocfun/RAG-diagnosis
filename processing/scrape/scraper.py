import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import quote
import pandas as pd
from datetime import datetime
import sys
import os
import subprocess
import random

# --- Helper functions and setup (no changes needed here) ---

def install_package(package):
    """Checks if a package is installed and installs it if not."""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)

try:
    install_package('selenium')
    install_package('chromedriver_autoinstaller')
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    import chromedriver_autoinstaller
    SELENIUM_AVAILABLE = True
    print("✓ Selenium is available and configured.")
except (ImportError, Exception) as e:
    SELENIUM_AVAILABLE = False
    print(f"✗ Selenium not available. Error: {e}")

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
]

class MultiSourceAcademicScraper:
    # ... __init__, setup_selenium, setup_requests ... (no changes needed)
    def __init__(self, use_selenium=True):
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.papers = []
        if self.use_selenium:
            self.setup_selenium()
        self.setup_requests()

    def setup_selenium(self):
        try:
            chromedriver_autoinstaller.install()
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✓ Selenium WebDriver initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Selenium: {e}")
            self.use_selenium = False

    def setup_requests(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

    # <<< NEW FUNCTION FOR PAPERS.COOL >>>
    def scrape_papers_cool(self, query, max_results=1000):
        """
        Scrapes paper results from papers.cool using requests and BeautifulSoup.
        """
        print(f"🔍 Scraping Papers.cool for: '{query}'")
        papers = []
        # The site uses 'show' parameter to control the number of results.
        # We can set it to a high number to get all results on one page.
        encoded_query = quote(query)
        url = f"https://papers.cool/arxiv/search?query={encoded_query}&show={max_results}"

        try:
            time.sleep(random.uniform(1, 3)) # Be respectful
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all paper containers
            paper_containers = soup.select('div.panel.paper')
            
            if not paper_containers:
                print("⚠ No paper containers found on Papers.cool.")
                return []

            print(f"✓ Found {len(paper_containers)} paper elements on Papers.cool.")

            for container in paper_containers:
                try:
                    # Extract title
                    title_elem = container.select_one('a.title-link')
                    title = title_elem.get_text(strip=True) if title_elem else 'N/A'
                    
                    # Extract authors
                    author_tags = container.select('p.authors a.author')
                    authors = ', '.join([a.get_text(strip=True) for a in author_tags])

                    # Extract abstract
                    summary_elem = container.select_one('p.summary')
                    abstract = summary_elem.get_text(strip=True) if summary_elem else 'N/A'
                    
                    # Extract publish date
                    date_elem = container.select_one('p.date')
                    date_text = date_elem.get_text(strip=True) if date_elem else ''
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
                    year = year_match.group(0) if year_match else 'N/A'
                    
                    # Extract URL
                    paper_url = 'N/A'
                    if title_elem and 'href' in title_elem.attrs:
                        paper_url = "https://papers.cool" + title_elem['href']
                    
                    paper_data = {
                        'title': title,
                        'authors': authors,
                        'year': year,
                        'abstract': abstract,
                        'url': paper_url,
                        'source': 'Papers.cool'
                    }
                    papers.append(paper_data)
                
                except Exception as e:
                    print(f"⚠ Error extracting single paper from Papers.cool: {e}")
                    continue
            
            print(f"✓ Finished scraping Papers.cool. Total found: {len(papers)}")
            return papers

        except requests.exceptions.RequestException as e:
            print(f"✗ Error during request to Papers.cool: {e}")
            return []
    
    # ... other scraping functions remain the same ...
    def scrape_all_sources(self, query):
        all_papers = []
        print(f"\n🚀 Starting multi-source scraping for: {query}\n" + "=" * 60)
        
        # <<< ADD THE NEW SCRAPER TO THE LIST >>>
        pc_papers = self.scrape_papers_cool(query)
        all_papers.extend(pc_papers)
        
        # You can still run the others if you want
        # if self.use_selenium:
        #     cp_papers = self.scrape_connected_papers(query, page_limit=2)
        #     all_papers.extend(cp_papers)
            
        # ss_papers = self.scrape_semantic_scholar(query, limit=20)
        # all_papers.extend(ss_papers)
        
        # gs_papers = self.scrape_google_scholar_basic(query, num_results=10)
        # all_papers.extend(gs_papers)
        
        unique_papers = self.remove_duplicates(all_papers)
        print("=" * 60 + f"\n🎯 Total unique papers found: {len(unique_papers)}")
        return unique_papers
    
    def remove_duplicates(self, papers):
        # ... (no changes needed)
        unique_papers, seen_titles = [], set()
        for paper in papers:
            title = (paper.get('title') or '').lower().strip()
            if title:
                title_key = ' '.join(re.findall(r'\w+', title)[:8])
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_papers.append(paper)
        return unique_papers
    
    def save_results(self, papers, query):
        # ... (no changes needed)
        if not papers:
            print("📭 No papers to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:30]
        
        df = pd.DataFrame(papers)
        cols = ['title', 'authors', 'year', 'citations', 'references', 'source', 'abstract', 'venue', 'url']
        df = df.reindex(columns=[c for c in cols if c in df.columns])
        
        csv_filename = f"academic_papers_{safe_query}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"💾 Saved {len(papers)} papers to {csv_filename}")
        
        json_filename = f"academic_papers_{safe_query}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(papers)} papers to {json_filename}")
        
        print("\n📊 SUMMARY BY SOURCE:")
        print(df['source'].value_counts())
        
        print("\n📄 SAMPLE PAPERS (WITH ABSTRACTS):")
        for i, paper in enumerate(papers[:2]):
            print(f"\n--- PAPER {i+1} ---")
            print(f"Title: {paper.get('title', 'N/A')}")
            print(f"Authors: {paper.get('authors', 'N/A')}")
            print(f"Year: {paper.get('year', 'N/A')}, Citations: {paper.get('citations', 'N/A')}")
            print(f"Source: {paper.get('source', 'N/A')}")
            abstract = (paper.get('abstract') or "N/A")
            print(f"Abstract: {abstract[:300]}...")

    def close(self):
        # ... (no changes needed)
        if self.use_selenium and hasattr(self, 'driver'):
            self.driver.quit()
            print("🔌 Selenium driver closed")

def main():
    print("🔬 Multi-Source Academic Paper Scraper\n" + "=" * 50)
    
    # We don't need Selenium for papers.cool, so we can set use_selenium=False
    # But let's keep it True in case you want to uncomment the other scrapers
    scraper = MultiSourceAcademicScraper(use_selenium=True) 
    
    try:
        # Update the query for the new site
        query = "benchmark dataset for leishmaniasis diagnosis"
        
        # Call the new function directly, or use scrape_all_sources
        # papers = scraper.scrape_papers_cool(query)
        papers = scraper.scrape_all_sources(query)
        
        if papers:
            scraper.save_results(papers, query)
        else:
            print("\n❌ No papers found.")
    except KeyboardInterrupt:
        print("\n⏹ Scraping interrupted by user")
    except Exception as e:
        print(f"❌ An unexpected error occurred in main: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
