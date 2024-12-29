import requests
from bs4 import BeautifulSoup
import re
import os
from time import sleep
import argparse

def get_random_wikipedia_article():
    """
    Fetches a random Wikipedia article and returns its title and text content.
    """
    # Wikipedia's random article URL
    random_url = "https://en.wikipedia.org/wiki/Special:Random"
    
    # Add headers to avoid potential blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Get the random article
    response = requests.get(random_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get the article title
    title = soup.find(id="firstHeading").text.strip()
    
    # Get the main content
    content_div = soup.find(id="mw-content-text")
    
    # Remove unwanted elements but keep more content
    for element in content_div.find_all(['script', 'style', 'sup', 'span.mw-editsection']):
        element.decompose()
    
    # Get all text content including headers and lists
    content_parts = []
    
    # Add section headers
    for header in content_div.find_all(['h2', 'h3', 'h4']):
        header_text = header.get_text().strip()
        if header_text and not header_text.lower() == 'contents':
            content_parts.append(f"\n== {header_text} ==\n")
    
    # Add paragraphs
    for p in content_div.find_all('p'):
        text = p.get_text().strip()
        if text:
            content_parts.append(text)
    
    # Add lists
    for list_elem in content_div.find_all(['ul', 'ol']):
        for item in list_elem.find_all('li'):
            text = item.get_text().strip()
            if text:
                content_parts.append(f"• {text}")
    
    # Join all content
    text_content = '\n\n'.join(content_parts)
    
    # Clean up the text
    text_content = re.sub(r'\[\d+\]', '', text_content)  # Remove reference numbers
    text_content = re.sub(r'\s+', ' ', text_content)     # Normalize whitespace
    text_content = re.sub(r'\n\s*\n', '\n\n', text_content)  # Normalize paragraph breaks
    
    # Get the URL
    url = response.url
    
    return title, url, text_content.strip()

def save_articles(num_articles, output_dir='articles'):
    """
    Downloads and saves the specified number of random Wikipedia articles.
    
    Args:
        num_articles (int): Number of articles to download
        output_dir (str): Directory to save the articles in
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_articles):
        try:
            # Get random article
            title, url, content = get_random_wikipedia_article()
            
            # Create a valid filename from the title
            filename = re.sub(r'[<>:"/\\|?*]', '_', title)
            filepath = os.path.join(output_dir, f"{filename}.txt")
            
            # Save the article
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n")
                f.write(f"URL: {url}\n")
                f.write("\nContent:\n")
                f.write(content)
            
            print(f"Saved article {i+1}/{num_articles}: {title}")
            
            # Be nice to Wikipedia's servers
            sleep(1)
            
        except Exception as e:
            print(f"Error saving article {i+1}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Download random Wikipedia articles as text files')
    parser.add_argument('num_articles', type=int, help='Number of articles to download')
    parser.add_argument('--output-dir', type=str, default='articles',
                      help='Directory to save articles in (default: articles)')
    
    args = parser.parse_args()
    
    if args.num_articles < 1:
        print("Please specify a positive number of articles to download")
        return
    
    save_articles(args.num_articles, args.output_dir)

if __name__ == "__main__":
    main()