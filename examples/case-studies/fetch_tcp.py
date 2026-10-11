"""Download the TCP (OpenNeuro ds005237, CC0) files the case studies need into ./data.

Usage: python examples/case-studies/fetch_tcp.py [task ...]      (default: restAP restPA hammerAP)

Files come from OpenNeuro's public S3 bucket. Each download is checked against the size and MD5
in the bucket listing; files already present with the right size are kept. About 1.7 MB per run,
about 2 GB for every rest and face-task run of all subjects.
"""

import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

BUCKET = "https://s3.amazonaws.com/openneuro.org"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TARGET = Path(__file__).resolve().parent / "data" / "ds005237"


def fetch(url, attempts=5):
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except OSError:
            time.sleep(2**attempt)
    raise OSError(f"could not download {url}")


def listing(prefix):
    """Every (key, size, md5) under prefix; md5 is None for multipart uploads."""
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix} | ({"continuation-token": token} if token else {})
        root = ElementTree.fromstring(fetch(f"{BUCKET}?{urllib.parse.urlencode(query)}"))
        for item in root.findall("s3:Contents", NS):
            etag = item.find("s3:ETag", NS).text.strip('"')
            yield item.find("s3:Key", NS).text, int(item.find("s3:Size", NS).text), None if "-" in etag else etag
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token:
            return


def wanted(key, tasks):
    name = key.rsplit("/", 1)[-1]
    if key in ("ds005237/participants.tsv", "ds005237/dataset_description.json"):
        return True
    if key.startswith("ds005237/phenotype/"):
        return name.endswith(".tsv")
    task = re.search(r"task-([A-Za-z]+)_", name)
    if task is None or task.group(1) not in tasks:
        return False
    return name.endswith("_parcellated.h5") or name.endswith("_bold.json") or name.endswith("_events.tsv")


def download(item):
    key, size, md5 = item
    out = TARGET / key.removeprefix("ds005237/")
    if out.exists() and out.stat().st_size == size:
        return "kept"
    data = fetch(f"{BUCKET}/{urllib.parse.quote(key)}")
    if len(data) != size or (md5 and hashlib.md5(data).hexdigest() != md5):
        return f"MISMATCH {key}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.name + ".part").write_bytes(data)
    out.with_name(out.name + ".part").replace(out)
    return "fetched"


def main():
    tasks = set(sys.argv[1:] or ["restAP", "restPA", "hammerAP"])
    items = [i for prefix in ("ds005237/participants", "ds005237/dataset_description", "ds005237/phenotype/",
                              "ds005237/fMRI_timeseries_clean_denoised_GSR_parcellated/", "ds005237/sub-")
             for i in listing(prefix) if wanted(i[0], tasks)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(download, items))
    problems = [r for r in results if r not in ("kept", "fetched")]
    print(f"{len(items)} files: {results.count('fetched')} fetched, {results.count('kept')} kept, {len(problems)} problems")
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
