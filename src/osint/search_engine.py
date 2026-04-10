import requests
import json
import time
from urllib.parse import quote

class OSINTSearcher:
    def __init__(self):
        self.wayback_url = "http://archive.org/wayback/available"
        self.searx_instances = [
            "https://searx.be",
            "https://searxng.site",
            "https://priv.au"
        ]

    def check_archive(self, url):
        """Checks if a URL has an archived version in the Wayback Machine."""
        try:
            params = {'url': url}
            response = requests.get(self.wayback_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
                    return data['archived_snapshots']['closest']['url']
        except Exception:
            return None
        return None

    def query_searx(self, query, category='general'):
        """Queries multiple SearXNG instances for multi-engine results."""
        results = []
        for instance in self.searx_instances:
            try:
                # category can be 'social media', 'files', etc.
                url = f"{instance}/search?q={quote(query)}&categories={category}&format=json"
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    results.extend(data.get('results', []))
                    if results: break # Stop if we got results from first working instance
            except Exception:
                continue
        return results

    def deep_hunt(self, handle):
        """Performs a deep search for a handle across archives and search engines."""
        report = {
            'handle': handle,
            'archive_links': [],
            'search_hits': []
        }
        
        # 1. Check Instagram Archive
        ig_url = f"https://www.instagram.com/{handle}/"
        archive = self.check_archive(ig_url)
        if archive: report['archive_links'].append(archive)
        
        # 2. Query for mentions across the web
        mentions = self.query_searx(f'"{handle}"')
        for hit in mentions:
            report['search_hits'].append({
                'title': hit.get('title'),
                'url': hit.get('url'),
                'content': hit.get('content')
            })
            
        return report

if __name__ == "__main__":
    searcher = OSINTSearcher()
    # Test with the known handles
    for h in ["rolesville_files", "fcs_files", "roseville_files"]:
        print(f"[*] Deep hunting handle: {h}...")
        results = searcher.deep_hunt(h)
        print(json.dumps(results, indent=2))
        time.sleep(2)
