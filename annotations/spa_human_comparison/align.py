"""Align the human-annotated State Pension age sentences (comparisons.csv, documents/state_pension
segmentation) to the current segmentation (documents/state_pension_age/input.txt), by locating each
human sentence's normalised text inside the normalised new document."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
norm = lambda t: re.sub(r"[^a-z0-9]", "", re.sub(r"\*\*\(link: ?\d+\)|\(reference: ?[\d, ]+\)", "", str(t).lower()))


def new_sentences():
    rows = []
    for l in (ROOT / "documents/state_pension_age/input.txt").read_text().splitlines():
        if l.strip():
            sid, _, text = l.partition(" ")
            rows.append((sid, text))
    return rows


def fuzzy(t, new, threshold=0.85):
    """Best run of 1-3 consecutive new sentences matching t, if its similarity clears the threshold."""
    from difflib import SequenceMatcher
    normed = [(sid, norm(x)) for sid, x in new]
    best, score = None, threshold
    for k in (1, 2, 3):
        for j in range(len(normed) - k + 1):
            cand = "".join(x for _, x in normed[j:j + k])
            if abs(len(cand) - len(t)) > 0.3 * len(t):
                continue
            r = SequenceMatcher(None, t, cand, autojunk=False).ratio()
            if r > score:
                best, score = [sid for sid, _ in normed[j:j + k]], r
    return best


def alignment():
    """Return {human id: [new ids]} and the list of human ids that could not be located."""
    new = new_sentences()
    flat, owner = "", []
    for sid, text in new:
        t = norm(text); flat += t; owner += [sid] * len(t)
    human = pd.read_csv(ROOT / "annotations/comparisons.csv", encoding="utf-8-sig")
    human.columns = [c.strip() for c in human.columns]
    mapping, missing, pos = {}, [], 0
    for _, r in human.iterrows():
        t = norm(r["Sentence"])
        if not t:
            continue
        i = flat.find(t, pos)
        if i < 0:
            i = flat.find(t)                       # fall back to anywhere in the document
        if i < 0:
            best = fuzzy(t, new)                   # footnote markers, hyphenation or spacing differ
            if best:
                mapping[r["Sentence ID"]] = best
            else:
                missing.append(r["Sentence ID"])   # e.g. tables, which the current extraction drops
            continue
        mapping[r["Sentence ID"]] = list(dict.fromkeys(owner[i:i + len(t)]))
        pos = i + len(t)
    return mapping, missing


if __name__ == "__main__":
    m, miss = alignment()
    sizes = pd.Series({k: len(v) for k, v in m.items()})
    print(len(m), "located;", len(miss), "not located:", miss[:20])
    print("new sentences per human sentence:", sizes.value_counts().to_dict())
