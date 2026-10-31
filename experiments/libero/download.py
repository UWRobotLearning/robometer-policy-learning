"""Download a public OpenPI checkpoint to /data with size and MD5 verification."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

BUCKET = "openpi-assets"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def objects(prefix):
    token = None
    while True:
        query = {"prefix": prefix, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        with urlopen(f"{API}?{urlencode(query)}", timeout=30) as response:
            page = json.load(response)
        yield from page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            break


def download(item, prefix, destination):
    name = item["name"]
    relative = name.removeprefix(prefix)
    if not name.startswith(prefix) or not relative or ".." in Path(relative).parts:
        raise ValueError(f"Unexpected checkpoint object: {name}")
    output = destination / relative
    expected = int(item["size"])
    if output.exists() and output.stat().st_size == expected:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    digest = hashlib.md5()  # GCS supplies MD5 for standard checkpoint objects.
    url = f"{API}/{quote(name, safe='')}?alt=media"
    try:
        with urlopen(url, timeout=300) as response, partial.open("wb") as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                digest.update(block)
        if partial.stat().st_size != expected:
            raise OSError(f"Incomplete checkpoint object: {name}")
        if item.get("md5Hash") and base64.b64encode(digest.digest()).decode() != item["md5Hash"]:
            raise OSError(f"Checkpoint checksum mismatch: {name}")
        partial.replace(output)
        return output
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pi0_libero", choices=["pi0_libero", "pi05_libero", "pi0_base"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prefix = f"checkpoints/{args.name}/"
    catalog = list(objects(prefix))
    if not any(item["name"].endswith("norm_stats.json") for item in catalog) or not any(
        "/params/" in item["name"] for item in catalog
    ):
        raise RuntimeError(f"Missing checkpoint weights or normalization assets under {prefix}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(catalog)} objects, {sum(int(item['size']) for item in catalog) / 2**30:.2f} GiB", flush=True
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, item, prefix, args.out) for item in catalog]
        for future in as_completed(futures):
            output = future.result()
            print(f"Verified {output.relative_to(args.out)}", flush=True)
    print(f"Checkpoint ready: {args.out}", flush=True)


if __name__ == "__main__":
    main()
