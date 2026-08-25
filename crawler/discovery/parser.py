import re

def extract_product_id(url):
    m = re.search(r"/item/(\\d+)\\.html", url)
    if m:
        return int(m.group(1))
    return None