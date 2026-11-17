"""Download the public primary sources; record retrieval metadata and SHA-256."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'references/course-project.html': 'https://www.cis.upenn.edu/~myatskar/teaching/cis5300_fa26/term-project.html',
    'references/D16-1264.pdf': 'https://aclanthology.org/D16-1264.pdf',
    'references/D17-1215.pdf': 'https://aclanthology.org/D17-1215.pdf',
    'references/N18-2091.pdf': 'https://aclanthology.org/N18-2091.pdf',
    'references/adversarial-squad-readme.md': 'https://raw.githubusercontent.com/robinjia/adversarial-squad/master/README.md',
    'references/squad-adversarial-loader.py.txt': 'https://huggingface.co/datasets/stanfordnlp/squad_adversarial/raw/main/squad_adversarial.py',
    'data/raw/train-v1.1.json': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
    'data/raw/dev-v1.1.json': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
    'data/raw/addsent.json': 'https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/',
    'data/raw/addonesent.json': 'https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/',
}

def fetch(item):
    name, url = item
    result = {'file': name, 'url': url}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'CIS5300-course-project/1.0'})
        with urllib.request.urlopen(req, timeout=35) as response:
            data = response.read()
            if name.endswith('.json'):
                json.loads(data)
            if name.endswith('.pdf'):
                assert data.startswith(b'%PDF')
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            result.update(status=response.status, final_url=response.url, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(),
                          last_modified=response.headers.get('Last-Modified'),
                          etag=response.headers.get('ETag'),
                          fetched_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception as exc:
        result['error'] = str(exc)
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, SOURCES.items()))
    (ROOT / 'references/source_manifest.json').write_text(json.dumps(results, indent=2)+'\n')
