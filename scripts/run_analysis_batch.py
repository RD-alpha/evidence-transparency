"""Run the live analysis pipeline (/analyse/run) over several documents, sequentially, stopping on the first error.

Calls the Anthropic API (costs credits); run only with the user's authorisation (CLAUDE.md). No retries.
Usage: python scripts/run_analysis_batch.py alias1 alias2 ...   (from the repo root or web_app/)
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web_app"))
import app as appmod

client = appmod.app.test_client()
grand = 0.0
for alias in sys.argv[1:]:
    t0, cost, failed = time.time(), 0.0, None
    print(f"=== {alias}", flush=True)
    r = client.post("/analyse/run", json={"policy_name": alias})
    buf = b""
    for chunk in r.response:
        buf += chunk if isinstance(chunk, bytes) else chunk.encode()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("status") == "error":
                failed = ev.get("error", "unknown error"); print("   ERROR:", failed, flush=True)
            elif ev.get("status") == "complete":
                pass
            else:
                c = (ev.get("usage") or {}).get("cost_usd") or 0.0; cost += c
                print(f"   {ev['label'][:70]:70s} ${c:.3f}", flush=True)
    grand += cost
    scored = {a: (ROOT / "results" / alias / a / "scoring.json").exists() for a in ("diagnosis", "proposal", "implementation", "testing")}
    print(f"--- {alias}: ${cost:.2f} in {time.time()-t0:.0f}s | scoring files {scored} | running total ${grand:.2f}", flush=True)
    if failed or not all(scored.values()):
        print("STOPPING: error or missing scores; not retrying.", flush=True); sys.exit(1)
print(f"ALL DONE total ${grand:.2f}", flush=True)
