import json
import urllib.request

_URL = (
    "https://raw.githubusercontent.com/anishathalye/"
    "imagenet-simple-labels/master/imagenet-simple-labels.json"
)
_cache: list[str] | None = None


def get_labels() -> list[str]:
    global _cache
    if _cache is None:
        with urllib.request.urlopen(_URL) as r:
            _cache = json.loads(r.read().decode())
    return _cache