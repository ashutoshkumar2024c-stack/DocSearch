import os

# Tests use fake network clients: no spacing between calls and no on-disk OCR cache.
os.environ.setdefault("CLOUD_MIN_INTERVAL", "0")
os.environ.setdefault("OCR_CACHE_DIR", "off")