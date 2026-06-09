import json
import urllib.request

_URL = (
    "https://raw.githubusercontent.com/anishathalye/"
    "imagenet-simple-labels/master/imagenet-simple-labels.json"
)
_NUM_CLASSES = 1000
_cache: list[str] | None = None

def get_labels() -> list[str]:
    global _cache
    if _cache is None:
        try:
            with urllib.request.urlopen(_URL, timeout=10) as r:
                _cache = json.loads(r.read().decode())
        except OSError as e:
            print(f"[labels] Could not fetch ImageNet labels ({e}), "
                  f"falling back to class indices.")
            _cache = [f"class_{i}" for i in range(_NUM_CLASSES)]
    return _cache