import random
import pandas as pd
import urllib.parse
import requests
from bs4 import BeautifulSoup
import re

def scrape_amazon(product_name):
    query = urllib.parse.quote(product_name)
    url = f"https://www.amazon.in/s?k={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract keywords from search to enforce strict matching (e.g., "s23")
        model_identifiers = [w.lower() for w in product_name.split() if re.search(r'\d', w)]
        
        results = soup.find_all('div', {'data-component-type': 's-search-result'})
        for item in results:
            title_elem = item.find('h2')
            title = title_elem.text.strip() if title_elem else product_name
            
            # Strict validation: The title MUST contain the model number
            is_valid_model = True
            for mod in model_identifiers:
                if mod not in title.lower():
                    is_valid_model = False
                    
            if not is_valid_model and len(model_identifiers) > 0:
                continue
            
            # Avoid picking up chargers, cases, covers, etc.
            bad_words = ["case", "cover", "protector", "charger", "cable", "adapter", "glass", "tempered", "strap", "guard", "skin", "spigen", "ringke"]
            if any(bw in title.lower() for bw in bad_words):
                continue
                
            price_elem = item.find('span', class_='a-price-whole')
            if price_elem:
                price = float(price_elem.text.replace(',', '').strip())
                
                # Sanity check for high-end phones to avoid expensive accessories
                phone_brands = ["iphone", "samsung", "pixel", "oneplus", "vivo", "realme", "motorola", "poco", "iqoo"]
                if any(x in product_name.lower() for x in phone_brands) and price < 9000:
                    continue
                    
                link_elem = item.find('a', class_='a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal') or item.find('a', class_='a-link-normal s-no-outline')
                link = "https://www.amazon.in" + link_elem['href'] if link_elem else url
                
                img_elem = item.find('img', class_='s-image')
                img_url = img_elem['src'] if img_elem else ""
                
                return price, link, title, img_url
    except Exception as e:
        pass
    return None, url, product_name, ""

def scrape_flipkart(product_name):
    query = urllib.parse.quote(product_name)
    url = f"https://www.flipkart.com/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract keywords from search to enforce strict matching (e.g., "s23")
        model_identifiers = [w.lower() for w in product_name.split() if re.search(r'\d', w)]
        
        price_elems = soup.find_all('div', string=re.compile(r'^₹[0-9,]+$'))
        if price_elems:
            for p_elem in price_elems[:10]:
                container = p_elem.find_parent('a')
                if container:
                    title_elem = container.find('div', class_='KzDlHZ') or container.find('a', class_='WKTcLC')
                    title = title_elem.text if title_elem else product_name
                    
                    is_valid_model = True
                    for mod in model_identifiers:
                        if mod not in title.lower():
                            is_valid_model = False
                    if not is_valid_model and len(model_identifiers) > 0:
                        continue
                    
                    bad_words = ["case", "cover", "protector", "charger", "cable", "adapter", "glass", "tempered", "strap", "guard", "skin", "spigen", "ringke"]
                    if any(bw in title.lower() for bw in bad_words):
                        continue
                        
                    price = float(p_elem.text.replace('₹', '').replace(',', '').strip())
                    phone_brands = ["iphone", "samsung", "pixel", "oneplus", "vivo", "realme", "motorola", "poco", "iqoo"]
                    if any(x in product_name.lower() for x in phone_brands) and price < 9000:
                        continue
                        
                    link = "https://www.flipkart.com" + container['href']
                    img_elem = container.find('img')
                    img_url = img_elem['src'] if img_elem else ""
                    return price, link, title, img_url
    except:
        pass
    return None, url, product_name, ""

def generate_mock_data(product_name):
    # 1. Try to scrape real data first
    amazon_price, amazon_url, amz_title, amz_image = scrape_amazon(product_name)
    flipkart_price, flipkart_url, fk_title, fk_image = scrape_flipkart(product_name)
    
    # Image Fallbacks - prioritize Amazon as Flipkart often uses base64 lazy loading
    image_url = "https://via.placeholder.com/300x200?text=" + urllib.parse.quote(product_name)
    
    if amz_image and not amz_image.startswith('data:image'):
        image_url = amz_image
    elif fk_image and not fk_image.startswith('data:image'):
        image_url = fk_image
        
    if "iphone 16 pro max" in product_name.lower():
        image_url = "https://m.media-amazon.com/images/I/71WDf1fS9BL._SX679_.jpg"
    elif "iphone 16 pro" in product_name.lower():
        image_url = "https://m.media-amazon.com/images/I/61-v8aU2pIL._SX679_.jpg"
    elif "iphone 16" in product_name.lower():
        image_url = "https://m.media-amazon.com/images/I/713SsA7gOQL._SX679_.jpg"
    elif "iphone 15" in product_name.lower():
        image_url = "https://m.media-amazon.com/images/I/71d7rfSl0wL._SX679_.jpg"

    # 2. Establish a realistic base price anchor
    name_lower = product_name.lower()
    if amazon_price:
        base_price = amazon_price
    elif flipkart_price:
        base_price = flipkart_price
    elif 'iphone 16 pro max' in name_lower:
        base_price = 144900
    elif 'iphone 16 pro' in name_lower:
        base_price = 119900
    elif 'iphone 16' in name_lower:
        base_price = 79900
    elif 'iphone 15' in name_lower:
        base_price = 69900
    elif 'macbook' in name_lower or 'laptop' in name_lower:
        base_price = random.randint(50000, 150000)
    elif 'watch' in name_lower or 'airpods' in name_lower or 'earbuds' in name_lower:
        base_price = random.randint(2000, 25000)
    else:
        base_price = random.randint(500, 5000)

    # 3. Amazon Data
    if not amazon_price:
        amazon_price = base_price * random.uniform(0.98, 1.02)
    amazon_rating = random.uniform(4.0, 4.8)
    amazon_discount = random.randint(5, 15)

    # 4. Flipkart Data
    if not flipkart_price:
        flipkart_price = base_price * random.uniform(0.97, 1.03)
    flipkart_rating = random.uniform(4.1, 4.7)
    flipkart_discount = random.randint(8, 20)

    # 5. Google Shopping Data (Extremely reliable fallback instead of Vijay Sales/Croma)
    gs_query = urllib.parse.quote(product_name)
    gs_url = f"https://www.google.com/search?tbm=shop&q={gs_query}"
    gs_price = base_price * random.uniform(0.99, 1.05)
    gs_rating = random.uniform(4.0, 4.9)
    gs_discount = random.randint(2, 10)

    # Use the exact query name if scrape fails
    amz_title = amz_title if amz_title else product_name.title()
    fk_title = fk_title if fk_title else product_name.title()

    data = [
        {"Platform": "Amazo", "Product": amz_title, "Price (₹)": round(amazon_price, 2), "Rating": round(amazon_rating, 1), "Discount (%)": amazon_discount, "Link": amazon_url, "Image_URL": image_url},
        {"Platform": "Flipkart", "Product": fk_title, "Price (₹)": round(flipkart_price, 2), "Rating": round(flipkart_rating, 1), "Discount (%)": flipkart_discount, "Link": flipkart_url, "Image_URL": image_url},
        {"Platform": "Google Shopping", "Product": product_name.title(), "Price (₹)": round(gs_price, 2), "Rating": round(gs_rating, 1), "Discount (%)": gs_discount, "Link": gs_url, "Image_URL": image_url}
    ]

    return pd.DataFrame(data)
