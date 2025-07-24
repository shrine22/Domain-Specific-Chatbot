# scraper.py
import requests
from bs4 import BeautifulSoup
import re
import json

def fetch_html(url):
    """Fetches HTML content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_text(html_content):
    """Cleans HTML content to extract readable text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style tags
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Get text
    text = soup.get_text()

    # Remove multiple newlines and spaces
    text = re.sub(r'\n\s*\n', '\n', text) # Remove excessive newlines
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

    return text

def scrape_website(url, output_file="scraped_data.json"):
    """Scrapes content from a website and saves cleaned text to a JSON file."""
    print(f"Scraping: {url}")
    html_content = fetch_html(url)
    if html_content:
        cleaned_data = {
            "url": url,
            "content": clean_text(html_content)
        }
        # For simplicity, we're just scraping the main page.
        # For a real application, you'd want to crawl links on the page.
        return cleaned_data
    return None

if __name__ == "__main__":
    changi_airport_url = "https://www.changiairport.com/" # Replace with actual main URL if different
    jewel_changi_airport_url = "https://www.jewelchangiairport.com/" # Replace with actual main URL if different

    all_scraped_data = []

    # Scrape Changi Airport
    changi_data = scrape_website(changi_airport_url)
    if changi_data:
        all_scraped_data.append(changi_data)

    # Scrape Jewel Changi Airport
    jewel_data = scrape_website(jewel_changi_airport_url)
    if jewel_data:
        all_scraped_data.append(jewel_data)

    # You might want to break down large content into smaller chunks here
    # For now, we'll keep it simple and process large chunks.
    # In a real scenario, you'd apply chunking before vectorization.
    processed_chunks = []
    for item in all_scraped_data:
        # Simple chunking by paragraph or fixed size (e.g., 500 characters)
        # For demonstration, let's split by double newline as a rough paragraph separator
        paragraphs = item['content'].split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50: # Only add meaningful paragraphs
                processed_chunks.append({
                    "id": f"{item['url']}_{i}", # Unique ID for each chunk
                    "text": para.strip(),
                    "source_url": item['url']
                })


    output_filename = "cleaned_website_content.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)
    print(f"Scraping and cleaning complete. Data saved to {output_filename}")
    print(f"Total chunks processed: {len(processed_chunks)}")