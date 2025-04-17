import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
from fuzzywuzzy import fuzz, process
from urllib.parse import quote

# Function to search for a model on OpenRouter and extract its release date
def get_release_date_from_openrouter(model_name, organization):
    try:
        # Get the main models page first to find all available models
        models_url = "https://openrouter.ai/models"
        print(f"Fetching models from: {models_url}")
        response = requests.get(models_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            model_links = soup.find_all('a', href=True)
            
            # Extract all model URLs from the page
            potential_model_urls = []
            
            for link in model_links:
                href = link.get('href', '')
                # Filter for model links which usually have this pattern
                if href.startswith('/') and '/' in href[1:] and not href.startswith('//') and not '#' in href and not '?' in href:
                    # Check if the model name is in the URL using fuzzy matching
                    model_text = href.split('/')[-1].replace('-', ' ').replace(':', ' ').lower()
                    similarity = fuzz.token_set_ratio(model_name.lower(), model_text)
                    if similarity > 60:  # Use a lower threshold to catch more potential matches
                        full_url = f"https://openrouter.ai{href}"
                        potential_model_urls.append((full_url, similarity))
            
            # Sort by similarity score
            potential_model_urls.sort(key=lambda x: x[1], reverse=True)
            
            # Try top 5 most likely matches
            for model_url, score in potential_model_urls[:5]:
                print(f"Checking potential match: {model_url} (similarity score: {score})")
                model_response = requests.get(model_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if model_response.status_code == 200:
                    model_soup = BeautifulSoup(model_response.text, 'html.parser')
                    page_text = model_soup.get_text()
                    
                    # Check if the model name is actually found in the page content
                    if model_name.lower() in page_text.lower() or score > 85:
                        print(f"Found model page for {model_name}")
                        
                        # Look for date patterns
                        date_patterns = [
                            r'Created\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Published\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Available since\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released on\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released date:?\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Launch(?:ed)? date:?\s+([A-Za-z]+\s+\d+,\s+\d{4})'
                        ]
                        
                        for pattern in date_patterns:
                            date_match = re.search(pattern, page_text, re.IGNORECASE)
                            if date_match:
                                date_text = date_match.group(1)
                                try:
                                    # Convert to YYYY/MM/DD format
                                    parsed_date = datetime.strptime(date_text, '%b %d, %Y')
                                    return parsed_date.strftime('%Y/%m/%d'), model_url
                                except ValueError:
                                    try:
                                        parsed_date = datetime.strptime(date_text, '%B %d, %Y')
                                        return parsed_date.strftime('%Y/%m/%d'), model_url
                                    except ValueError:
                                        continue
        
        # Try organization-specific direct URLs as a fallback
        if organization and not pd.isna(organization):
            # Generate several potential URL patterns for the model
            org_clean = organization.lower()
            model_variations = [
                model_name.replace(' ', '-').lower(),
                model_name.replace(' ', '').lower(),
                model_name.split(' ')[0].lower()  # Just the first part of the name
            ]
            
            # For Gemini models, try specific patterns known to work
            if 'gemini' in model_name.lower():
                if 'experimental' in model_name.lower() or 'exp' in model_name.lower():
                    month = datetime.now().month
                    day = datetime.now().day
                    model_variations.append(f"gemini-2.5-pro-exp-{month:02d}-{day:02d}:free")
                    model_variations.append(f"gemini-2.5-pro-exp")
                    model_variations.append(f"gemini-2.0-pro-exp")
                    model_variations.append(f"gemini-1.5-pro-exp")
            
            # For GPT models
            if 'gpt' in model_name.lower():
                if 'turbo' in model_name.lower():
                    model_variations.append(f"gpt-4-turbo")
                if 'preview' in model_name.lower():
                    model_variations.append(f"gpt-4-preview")
            
            for model_var in model_variations:
                direct_url = f"https://openrouter.ai/{org_clean}/{model_var}"
                print(f"Trying direct URL: {direct_url}")
                response = requests.get(direct_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_text = soup.get_text()
                    
                    # Check if the model name is in the page content
                    if model_name.lower() in page_text.lower() or fuzz.partial_ratio(model_name.lower(), page_text.lower()) > 80:
                        print(f"Found model page via direct URL")
                        
                        date_patterns = [
                            r'Created\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Published\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Available since\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released on\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Released date:?\s+([A-Za-z]+\s+\d+,\s+\d{4})',
                            r'Launch(?:ed)? date:?\s+([A-Za-z]+\s+\d+,\s+\d{4})'
                        ]
                        
                        for pattern in date_patterns:
                            date_match = re.search(pattern, page_text, re.IGNORECASE)
                            if date_match:
                                date_text = date_match.group(1)
                                try:
                                    # Convert to YYYY/MM/DD format
                                    parsed_date = datetime.strptime(date_text, '%b %d, %Y')
                                    return parsed_date.strftime('%Y/%m/%d'), direct_url
                                except ValueError:
                                    try:
                                        parsed_date = datetime.strptime(date_text, '%B %d, %Y')
                                        return parsed_date.strftime('%Y/%m/%d'), direct_url
                                    except ValueError:
                                        continue
        
        print(f"Could not find release date for {model_name} on OpenRouter")
        return None, None
        
    except Exception as e:
        print(f"Error searching for {model_name} on OpenRouter: {e}")
        return None, None

# Function to search for model release date via Google search
def get_release_date_from_google(model_name, organization):
    try:
        # Skip if model name is empty
        if not model_name:
            return None, None
        
        # Try multiple search patterns with different specificity
        search_patterns = []
        
        # Organization-specific searches
        if organization and not pd.isna(organization):
            search_patterns.extend([
                f"{model_name} {organization} release date site:wikipedia.org",  # Try Wikipedia first
                f"{model_name} {organization} release date",
                f"{model_name} {organization} launched",
                f"{model_name} {organization} announced"
            ])
        
        # Generic searches
        search_patterns.extend([
            f"{model_name} language model release date site:wikipedia.org",  # Wikipedia
            f"{model_name} language model release date",
            f"{model_name} AI model release",
            f"{model_name} LLM release"
        ])
        
        # Try each search pattern until we find a date
        for search_query in search_patterns:
            # URL encode the search query
            encoded_query = quote(search_query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            print(f"Searching Google: {search_url}")
            
            # Use a realistic browser user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            try:
                response = requests.get(search_url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # More sophisticated web scraping - extract specific elements
                    
                    # 1. Get search result items (Google wraps results in specific div elements)
                    search_results = []
                    # Try multiple common selectors for Google results
                    selectors = ['div.g', '.g', '.rc', '.yuRUbf', 'div[data-hveid]', 'div.tF2Cxc']
                    
                    for selector in selectors:
                        results = soup.select(selector)
                        if results:
                            search_results.extend(results)
                            break
                    
                    # If standard selectors didn't work, go with a more general approach
                    if not search_results:
                        # Look for any divs that contain both a link and text
                        for div in soup.find_all('div'):
                            if div.find('a') and len(div.get_text()) > 100:
                                search_results.append(div)
                    
                    # 2. Extract structured data from each result
                    extracted_data = []
                    
                    for result in search_results:
                        # Get title, url and snippet
                        title_elem = result.find(['h3', 'h2', 'h4'])
                        link_elem = result.find('a', href=True)
                        snippet_elems = result.find_all(['span', 'div'], attrs={'style': None, 'class': None}, limit=3)
                        
                        title = title_elem.get_text() if title_elem else ''
                        url = link_elem['href'] if link_elem else ''
                        if url.startswith('/url?'):
                            # Extract actual URL from Google's redirect URL
                            url = re.search(r'url=([^&]+)', url).group(1) if re.search(r'url=([^&]+)', url) else url
                        
                        # Get the longest text as the snippet
                        snippet = ''
                        for elem in snippet_elems:
                            if elem and len(elem.get_text()) > len(snippet):
                                snippet = elem.get_text()
                        
                        # If we didn't find a good snippet, try to get any text content
                        if not snippet:
                            for elem in result.find_all(['span', 'div']):
                                text = elem.get_text().strip()
                                if len(text) > 50 and len(text) > len(snippet):
                                    snippet = text
                        
                        # Get any date mentioned in the result title or snippet
                        result_text = f"{title} {snippet}"
                        # Check if this result is relevant to our model using fuzzy matching
                        relevance_score = fuzz.partial_ratio(model_name.lower(), result_text.lower())
                        
                        if relevance_score > 75 or model_name.lower() in result_text.lower():
                            extracted_data.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet,
                                'combined_text': result_text,
                                'relevance_score': relevance_score
                            })
                    
                    # Sort by relevance
                    extracted_data.sort(key=lambda x: x['relevance_score'], reverse=True)
                    
                    # 3. Process the most relevant results
                    for result in extracted_data[:5]:  # Check top 5 most relevant results
                        result_text = result['combined_text']
                        source_url = result['url']
                        
                        # Look for dates in the result text
                        date = extract_date_from_text(result_text, model_name)
                        if date:
                            return date, source_url
                        
                        # If the result URL points to Wikipedia or organization site, follow it for deeper analysis
                        if result['url'] and ('wikipedia.org' in result['url'] or 
                                         (organization and not pd.isna(organization) and 
                                          organization.lower() in result['url'].lower())):
                            try:
                                print(f"Following result URL: {result['url']}")
                                page_response = requests.get(result['url'], headers=headers, timeout=10)
                                if page_response.status_code == 200:
                                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                                    
                                    # For Wikipedia, look in specific sections
                                    if 'wikipedia.org' in result['url']:
                                        # Check infobox first (common location for release dates)
                                        infobox = page_soup.find('table', class_='infobox')
                                        if infobox:
                                            infobox_text = infobox.get_text()
                                            date = extract_date_from_text(infobox_text, model_name)
                                            if date:
                                                return date, source_url
                                    
                                    # Check page content
                                    main_content = page_soup.find(['div', 'article', 'main'])
                                    content_text = main_content.get_text() if main_content else page_soup.get_text()
                                    
                                    # Look for sections/paragraphs mentioning the model
                                    paragraphs = page_soup.find_all(['p', 'div', 'section'])
                                    relevant_paragraphs = []
                                    
                                    for para in paragraphs:
                                        para_text = para.get_text()
                                        if model_name.lower() in para_text.lower() or fuzz.partial_ratio(model_name.lower(), para_text.lower()) > 80:
                                            relevant_paragraphs.append(para_text)
                                    
                                    # Check relevant paragraphs first, then fall back to full content
                                    for para in relevant_paragraphs:
                                        date = extract_date_from_text(para, model_name)
                                        if date:
                                            return date, source_url
                                    
                                    # If no date found in specific paragraphs, try the whole content
                                    date = extract_date_from_text(content_text, model_name)
                                    if date:
                                        return date, source_url
                            except Exception as e:
                                print(f"Error following URL {result['url']}: {e}")
                    
                    # If we've processed all results without finding a date, check the full page text
                    full_page_text = soup.get_text()
                    date = extract_date_from_text(full_page_text, model_name)
                    if date:
                        return date, search_url
            
            except Exception as e:
                print(f"Error during Google search for query '{search_query}': {e}")
                # Continue to next search pattern
                continue
        
        print(f"Could not find release date for {model_name} via Google searches")
        return None, None
    
    except Exception as e:
        print(f"Error in Google search function: {e}")
        return None, None

# Helper function to extract a date from text
def extract_date_from_text(text, model_name):
    # Patterns for date formats near mentions of release or similar words
    release_date_patterns = [
        # Find dates near release keywords with different formats
        r'(?:released|launched|announced|unveiled|published|introduced)\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'(?:released|launched|announced|unveiled|published|introduced)\s+in\s+([A-Za-z]+\s+\d{4})',  # Month Year
        r'(?:released|launched|announced|unveiled|published|introduced)\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})', # DD Month YYYY
        r'(?:released|launched|announced|unveiled|published|introduced)\s+on\s+(\d{4}-\d{2}-\d{2})', # YYYY-MM-DD
        
        # Release date with various formats
        r'release\s+date\s*:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'release\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'release\s+date\s*:?\s*(\d{4}-\d{2}-\d{2})',
        
        # Introduction/availability wording
        r'became\s+available\s+on\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'became\s+available\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        r'became\s+available\s+on\s+(\d{4}-\d{2}-\d{2})'
    ]
    
    for pattern in release_date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for date_text in matches:
                try:
                    # Process different date formats
                    return parse_date_to_yyyy_mm_dd(date_text)
                except ValueError as e:
                    print(f"Error parsing date: {e} for text: {date_text}")
                    continue
    
    # If no dates found with specific release patterns, look for any dates near model mentions
    # Find all instances of the model name
    model_indices = [m.start() for m in re.finditer(re.escape(model_name), text, re.IGNORECASE)]
    if not model_indices and ' ' in model_name:
        # Try with shortened model name (first part only)
        short_name = model_name.split(' ')[0]
        model_indices = [m.start() for m in re.finditer(re.escape(short_name), text, re.IGNORECASE)]
    
    if model_indices:
        # Generic date patterns
        date_patterns = [
            r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY or Month DD YYYY
            r'(\d{4}-\d{2}-\d{2})',                # YYYY-MM-DD
            r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'      # DD Month YYYY
        ]
        
        # For each model mention, find the closest date
        closest_date = None
        closest_distance = float('inf')
        
        for model_idx in model_indices:
            # Get context window around model mention
            start_pos = max(0, model_idx - 200)
            end_pos = min(len(text), model_idx + 200)
            context = text[start_pos:end_pos]
            
            for pattern in date_patterns:
                date_matches = re.findall(pattern, context)
                for date_text in date_matches:
                    date_idx = context.find(date_text)
                    if date_idx >= 0:
                        # Calculate distance from model mention
                        absolute_date_idx = start_pos + date_idx
                        distance = abs(absolute_date_idx - model_idx)
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_date = date_text
        
        # Use the closest date if it's reasonably close (within 200 chars)
        if closest_date and closest_distance < 200:
            try:
                return parse_date_to_yyyy_mm_dd(closest_date)
            except ValueError:
                pass
    
    return None

# Helper function to parse various date formats to YYYY/MM/DD
def parse_date_to_yyyy_mm_dd(date_text):
    # Remove extra whitespace
    date_text = re.sub(r'\s+', ' ', date_text).strip()
    
    # Check for specific patterns and parse accordingly
    if re.match(r'\d{4}-\d{2}-\d{2}', date_text):  # YYYY-MM-DD
        parsed_date = datetime.strptime(date_text, '%Y-%m-%d')
        return parsed_date.strftime('%Y/%m/%d')
    
    elif re.match(r'[A-Za-z]+\s+\d{4}$', date_text):  # Month YYYY (no day)
        # Set day to 1st of the month
        try:
            parsed_date = datetime.strptime(date_text + " 1", '%B %Y %d')
        except ValueError:
            parsed_date = datetime.strptime(date_text + " 1", '%b %Y %d')
        return parsed_date.strftime('%Y/%m/%d')
    
    elif re.match(r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}', date_text):  # Month DD, YYYY or Month DD YYYY
        if ',' in date_text:
            try:
                parsed_date = datetime.strptime(date_text, '%B %d, %Y')
            except ValueError:
                parsed_date = datetime.strptime(date_text, '%b %d, %Y')
        else:
            # Add comma for parsing
            parts = date_text.split()
            if len(parts) == 3:  # Month DD YYYY
                modified_date = f"{parts[0]} {parts[1]}, {parts[2]}"
                try:
                    parsed_date = datetime.strptime(modified_date, '%B %d, %Y')
                except ValueError:
                    parsed_date = datetime.strptime(modified_date, '%b %d, %Y')
            else:
                raise ValueError(f"Unexpected date format: {date_text}")
    
    elif re.match(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', date_text):  # DD Month YYYY
        try:
            parsed_date = datetime.strptime(date_text, '%d %B %Y')
        except ValueError:
            parsed_date = datetime.strptime(date_text, '%d %b %Y')
    
    else:
        raise ValueError(f"Unrecognized date format: {date_text}")
    
    return parsed_date.strftime('%Y/%m/%d')

# Function to search for model release date on organization's blog
def get_release_date_from_org_blog(model_name, organization):
    try:
        # Skip if organization is empty
        if not organization or pd.isna(organization):
            return None, None
            
        # Map organization names to their blog URLs
        org_blogs = {
            "OpenAI": "https://openai.com/blog",
            "Google": "https://blog.google/technology/ai/",
            "Anthropic": "https://www.anthropic.com/news",
            "Meta": "https://ai.meta.com/blog/",
            "DeepSeek": "https://blog.deepseek.com/",
            "xAI": "https://x.ai/blog/",
            "Mistral AI": "https://mistral.ai/news/",
            "Mistral": "https://mistral.ai/news/",
            "Microsoft": "https://blogs.microsoft.com/ai/",
            "Cohere": "https://txt.cohere.com/",
            "Alibaba": "https://www.alibabacloud.com/blog",
            "Amazon": "https://aws.amazon.com/blogs/machine-learning/",
            "StepFun": "https://www.stepfun.ai/blog",
            "Tencent": "https://www.tencentresearch.com/news",
            "SoundAI": "https://www.soundai.com/news",
            "AbacusAI": "https://abacus.ai/blog"
        }
        
        # Clean organization name
        org_clean = organization.strip()
        
        # Check if we have a blog URL for this organization
        if org_clean not in org_blogs:
            return None, None
            
        blog_url = org_blogs[org_clean]
        search_query = model_name.replace(' ', '+')
        
        # For some sites, add a search parameter
        if "?" in blog_url:
            search_url = f"{blog_url}&q={search_query}"
        else:
            search_url = f"{blog_url}?q={search_query}"
            
        print(f"Searching organization blog: {search_url}")
        response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article titles or headings that might contain the model name
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'article-title', 'title'])
            relevant_articles = []
            
            for heading in headings:
                heading_text = heading.get_text()
                # Check if the model name is in the heading (using fuzzy matching)
                if fuzz.partial_ratio(model_name.lower(), heading_text.lower()) > 75:
                    # If we find the model name in a heading, store its parent article
                    parent = heading.parent
                    if parent not in relevant_articles:
                        relevant_articles.append(parent)
            
            # Look for dates in the relevant articles first, then fallback to the whole page
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'([A-Za-z]+\s+\d{1,2},\s+\d{4})',  # Month DD, YYYY
                r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'   # DD Month YYYY
            ]
            
            # First look in relevant articles
            for article in relevant_articles:
                article_text = article.get_text()
                
                for pattern in date_patterns:
                    date_matches = re.findall(pattern, article_text)
                    if date_matches:
                        # Try to parse the first match
                        date_text = date_matches[0]
                        try:
                            # Try different date formats
                            if re.match(r'\d{4}-\d{2}-\d{2}', date_text):
                                parsed_date = datetime.strptime(date_text, '%Y-%m-%d')
                            elif re.match(r'[A-Za-z]+\s+\d{1,2},\s+\d{4}', date_text):
                                try:
                                    parsed_date = datetime.strptime(date_text, '%B %d, %Y')
                                except ValueError:
                                    parsed_date = datetime.strptime(date_text, '%b %d, %Y')
                            elif re.match(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', date_text):
                                try:
                                    parsed_date = datetime.strptime(date_text, '%d %B %Y')
                                except ValueError:
                                    parsed_date = datetime.strptime(date_text, '%d %b %Y')
                            return parsed_date.strftime('%Y/%m/%d'), search_url
                        except ValueError:
                            continue
            
            # If we didn't find dates in relevant articles, check date elements across the page
            date_elements = soup.select("time, .date, .published, .post-date, [itemprop='datePublished']")
            
            if date_elements:
                # Get the first date, as it's likely to be from the most relevant result
                date_text = date_elements[0].get_text().strip()
                
                # Try various date formats
                date_formats = ['%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%d %B %Y', '%d %b %Y']
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_text, date_format)
                        return parsed_date.strftime('%Y/%m/%d'), search_url
                    except ValueError:
                        continue
            
            # As a last resort, look for any dates in the page text
            page_text = soup.get_text()
            for pattern in date_patterns:
                date_matches = re.findall(pattern, page_text)
                if date_matches:
                    # Try to find a date that's close to the model name in the text
                    model_index = page_text.lower().find(model_name.lower())
                    if model_index >= 0:
                        # Find the closest date to the model mention
                        closest_date = None
                        closest_distance = float('inf')
                        
                        for date_text in date_matches:
                            date_index = page_text.find(date_text)
                            if date_index >= 0:
                                distance = abs(date_index - model_index)
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_date = date_text
                        
                        if closest_date and closest_distance < 500:  # Only use if within 500 chars
                            try:
                                # Try different date formats
                                if re.match(r'\d{4}-\d{2}-\d{2}', closest_date):
                                    parsed_date = datetime.strptime(closest_date, '%Y-%m-%d')
                                elif re.match(r'[A-Za-z]+\s+\d{1,2},\s+\d{4}', closest_date):
                                    try:
                                        parsed_date = datetime.strptime(closest_date, '%B %d, %Y')
                                    except ValueError:
                                        parsed_date = datetime.strptime(closest_date, '%b %d, %Y')
                                elif re.match(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', closest_date):
                                    try:
                                        parsed_date = datetime.strptime(closest_date, '%d %B %Y')
                                    except ValueError:
                                        parsed_date = datetime.strptime(closest_date, '%d %b %Y')
                                return parsed_date.strftime('%Y/%m/%d'), search_url
                            except ValueError:
                                continue
        
        print(f"Could not find release date for {model_name} on organization blog")
        return None, None
    
    except Exception as e:
        print(f"Error searching for {model_name} on organization blog: {e}")
        return None, None

# Function to search for a model release date using multiple sources
def search_model_release_date(model_name, organization):
    # Make sure organization is a string, not NaN
    organization_str = str(organization) if not pd.isna(organization) else ""
    
    # Try OpenRouter first with improved matching
    date, source_url = get_release_date_from_openrouter(model_name, organization_str)
    if date:
        return date, source_url
    
    # Try Google search
    date, source_url = get_release_date_from_google(model_name, organization_str)
    if date:
        return date, source_url
    
    # Try organization's blog as a backup
    date, source_url = get_release_date_from_org_blog(model_name, organization_str)
    if date:
        return date, source_url
    
    print(f"Could not find release date for {model_name} from any source")
    return None, None

def main():
    # Read the CSV file
    csv_path = 'llm_perf_chart.csv'
    df = pd.read_csv(csv_path)
    
    # Add source URL column if it doesn't exist
    if 'SourceURL' not in df.columns:
        df['SourceURL'] = ""
    
    # Filter for models with missing release dates
    missing_dates_df = df[df['Releasedate'].isna() | (df['Releasedate'] == '')]
    missing_dates_count = len(missing_dates_df)
    print(f"Found {missing_dates_count} models with missing release dates")
    
    # Process each row with a missing release date
    for index, row in missing_dates_df.iterrows():
        model_name = row['Model']
        organization = row['Organization']
        
        print(f"\nSearching for release date of {model_name} by {organization}...")
        
        # Get the release date and source URL
        release_date, source_url = search_model_release_date(model_name, organization)
        
        if release_date:
            print(f"Found release date for {model_name}: {release_date}")
            print(f"Source URL: {source_url}")
            df.at[index, 'Releasedate'] = release_date
            df.at[index, 'SourceURL'] = source_url
            # Save after each successful find to preserve progress
            df.to_csv(csv_path, index=False)
            print(f"Updated CSV file with release date for {model_name}")
        
        # Be nice to the server - add a delay between requests
        time.sleep(2)
    
    # Final save of the CSV (though we've been saving after each successful find)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV file update complete")

if __name__ == "__main__":
    main()
