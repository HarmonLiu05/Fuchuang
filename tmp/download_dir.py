import urllib.request
import json
import os
import base64

def download_github_dir(repo, branch, path, output_dir):
    """Download a single directory from GitHub using the API."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    print(f"Fetching: {api_url}")
    
    req = urllib.request.Request(api_url)
    req.add_header('User-Agent', 'Python-Downloader')
    
    with urllib.request.urlopen(req) as response:
        files = json.loads(response.read().decode())
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    for item in files:
        if item['type'] == 'file':
            name = item['name']
            size = item['size']
            download_url = item['download_url']
            print(f"Downloading: {name} ({size} bytes) from {download_url}")
            
            req2 = urllib.request.Request(download_url)
            req2.add_header('User-Agent', 'Python-Downloader')
            with urllib.request.urlopen(req2) as resp:
                content = resp.read()
            
            filepath = os.path.join(output_dir, name)
            with open(filepath, 'wb') as f:
                f.write(content)
            downloaded += 1
            print(f"  -> Saved to {filepath}")
        elif item['type'] == 'dir':
            print(f"Subdirectory: {item['name']} (skipping)")
    
    print(f"\nDone! Downloaded {downloaded} files to {output_dir}")

if __name__ == '__main__':
    download_github_dir(
        repo='HarmonLiu05/Weight',
        branch='main',
        path='checkpoints_baseline_80',
        output_dir='/workspace/Fuchuang/tmp/checkpoints_baseline_80'
    )
