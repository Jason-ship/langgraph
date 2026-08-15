import json
import urllib.request

BASE = "http://localhost:8123"

req = urllib.request.Request(
    f"{BASE}/threads/search",
    data=b'{"metadata": {}}',
    headers={"Content-Type": "application/json"},
    method="POST",
)
threads = json.load(urllib.request.urlopen(req))
print(f"found {len(threads)} threads")
for t in threads:
    tid = t.get("thread_id")
    if not tid:
        continue
    req2 = urllib.request.Request(f"{BASE}/threads/{tid}", method="DELETE")
    try:
        urllib.request.urlopen(req2)
        print(f"deleted {tid}")
    except Exception as e:  # noqa: BLE001
        print(f"failed {tid}: {e}")
