import os
import json
import time
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Output file path
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sankaku_tags.json')

def fetch_sankaku_tags():
    """
    Fetches tags from Sankaku API
    
    Returns:
        dict: Dictionary containing tags data
    """
    print("Fetching tags from Sankaku API...")
    
    # API base URL
    base_url = "https://sankakuapi.com/tags"
    
    # Headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'application/json',
        'Connection': 'keep-alive'
    }
    
    # Setup session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    all_tags = []
    page = 1
    last_successful_page = 0
    max_attempts = 3
    
    try:
        # Loop through pages to get all tags until failure
        while True:
            # Construct URL with page parameter
            url = f"{base_url}?lang=en&page={page}&limit=1000"
            print(f"Fetching page {page} with limit=1000...")
            
            attempt = 0
            success = False
            
            while attempt < max_attempts and not success:
                try:
                    # Make the request
                    response = session.get(url, headers=headers, timeout=120)
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        try:
                            # Parse JSON response
                            data = response.json()
                            
                            # Debug information
                            print(f"Page {page} response length: {len(str(data))} characters")
                            
                            # Check if we have tag data
                            if isinstance(data, list):
                                if len(data) == 0:
                                    # No more tags, exit the loop
                                    print(f"No more tags found after page {page-1}")
                                    success = True
                                    break
                                
                                # Add tags to our collection
                                all_tags.extend(data)
                                print(f"Added {len(data)} tags from page {page}")
                                last_successful_page = page
                                success = True
                            elif isinstance(data, dict):
                                # Handle case where API returns a different structure
                                if 'tags' in data:
                                    all_tags.extend(data['tags'])
                                    print(f"Added {len(data['tags'])} tags from page {page}")
                                    last_successful_page = page
                                    success = True
                                else:
                                    print(f"Unexpected response format on page {page}: {data.keys()}")
                                    break
                            else:
                                print(f"Unexpected response type on page {page}: {type(data)}")
                                break
                            
                            # Go to next page
                            page += 1
                            
                            # Add a short delay to avoid hitting rate limits
                            time.sleep(1)
                            
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON from page {page}")
                            attempt += 1
                            if attempt < max_attempts:
                                print(f"Retrying... (attempt {attempt+1}/{max_attempts})")
                                time.sleep(2)
                            else:
                                break
                    else:
                        print(f"Failed to fetch page {page}: HTTP {response.status_code}")
                        attempt += 1
                        if attempt < max_attempts:
                            print(f"Retrying... (attempt {attempt+1}/{max_attempts})")
                            time.sleep(2)
                        else:
                            break
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for page {page}: {e}")
                    attempt += 1
                    if attempt < max_attempts:
                        print(f"Retrying... (attempt {attempt+1}/{max_attempts})")
                        time.sleep(2)
                    else:
                        break
            
            if not success:
                break
        
        # Process and organize the tags
        tags_data = process_tags(all_tags)
        
        # Add metadata
        tags_data['_metadata'] = {
            'source': base_url,
            'fetched_at': datetime.now().isoformat(),
            'total_tags': len(all_tags),
            'parsing_method': 'api',
            'pages_fetched': last_successful_page,
            'last_successful_page': last_successful_page
        }
        
        return tags_data
        
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return {
            '_metadata': {
                'source': base_url,
                'fetched_at': datetime.now().isoformat(),
                'error': str(e),
                'last_successful_page': last_successful_page
            }
        }

def process_tags(tags_list):
    """
    Process the raw tags from API into a structured format
    
    Args:
        tags_list (list): Raw tags from the API
        
    Returns:
        dict: Processed tags data
    """
    tags_data = {}
    
    # Extract tag names into a list
    tag_names = []
    # Create dictionary mapping for tagName -> type
    tag_dict = {}
    
    for tag in tags_list:
        # Handle different possible API response formats
        if isinstance(tag, dict):
            tag_name = None
            tag_type = None
            
            if 'name_en' in tag:
                tag_name = tag['name_en']
            elif 'name' in tag:
                tag_name = tag['name']
            elif 'tag' in tag:
                tag_name = tag['tag']
                
            if 'type' in tag:
                tag_type = tag['type']
                
            if tag_name:
                tag_names.append(tag_name)
                if tag_type is not None:
                    tag_dict[tag_name] = {
                        'type': tag_type,
                        'id': tag.get('id', '')
                    }
                    
                    # Add Japanese name if available
                    if 'name_ja' in tag and tag['name_ja']:
                        tag_dict[tag_name]['name_ja'] = tag['name_ja']
        elif isinstance(tag, str):
            tag_names.append(tag)
    
    # Add to tags_data if we found any tags
    if tag_names:
        tags_data['tags'] = tag_names
    
    # Add tag dictionary
    if tag_dict:
        tags_data['tag_dict'] = tag_dict
    
    # For debugging, also store the raw data structure of a few sample tags
    debug_dir = os.path.join(os.path.dirname(OUTPUT_FILE), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, 'sankaku_api_sample.json'), 'w', encoding='utf-8') as f:
        # Save the first 10 tags or all if less than 10
        sample = tags_list[:min(10, len(tags_list))]
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    return tags_data

def save_tags(tags_data):
    """Save the tags data to a JSON file."""
    categories_count = len(tags_data) - 1  # Subtract one for metadata
    print(f"Saving {categories_count} tag categories to {OUTPUT_FILE}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save to JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(tags_data, f, ensure_ascii=False, indent=2)
    
    print(f"Tags saved successfully to {OUTPUT_FILE}")

def main():
    """Main function to fetch and save tags."""
    print(f"Starting Sankaku Tags Fetcher (API) at {datetime.now().isoformat()}")
    
    # Fetch tags
    tags_data = fetch_sankaku_tags()
    
    # Save tags
    save_tags(tags_data)
    
    print("Done!")

if __name__ == "__main__":
    main()
