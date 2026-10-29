"""State Pension age, diagnosis: the five current-pipeline runs (results/state_pension_age_rep{1..5},
Sonnet 5.5) against the human annotations in ../comparisons.csv (Annotator 1, Annotator 2, final consensus).

The human study used an older segmentation of the same PDF (documents/state_pension); align.py maps each
human sentence onto the current sentences. A human sentence counts as extracted by a run if the run put any
of its current sentences into diagnosis. Binary throughout: Yes against not-Yes (Maybe counts as not-Yes),
as in agreement_analysis.ipynb. Reads persisted artefacts only; no LLM calls.

    python compare.py        (writes summary_numbers.txt and figures/*.png next to this file)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from align import alignment  # noqa: E402

RESULTS = ROOT / "results"
RUNS = [RESULTS / f"state_pension_age_rep{i}" for i in range(1, 6)]
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
lines_out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines_out.append(s)


def norm_call(v):
    if pd.isna(v):
        return None
    v = str(v).strip().lower()
    if v in ("", "n/a", "na"):
        return None
    return "Maybe" if "maybe" in v else "Yes" if "yes" in v else "No" if "no" in v else None


def ids(path):
    return {l.split()[0] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()


def core_ids(d):
    w, c = d / "diagnosis" / "weights.json", d / "diagnosis" / "classification.json"
    names = {t["theme"] for t in json.loads(w.read_text())["themes"] if t["weight"] == "core"}
    return {i for t in json.loads(c.read_text())["themes"] if t["name"] in names for i in t["ids"]}


# ---- human annotations
h = pd.read_csv(ROOT / "annotations/comparisons.csv", encoding="utf-8-sig")
h = h.loc[:, ~h.columns.str.match(r"^Unnamed")]; h.columns = [c.strip() for c in h.columns]
h["annotator1"] = h["Extracted by Annotator 1"].map(norm_call)
h["annotator2"] = h["Extracted by Annotator 2"].map(norm_call)
cons = h["Consensus"].map(lambda v: "Disagree" if pd.notna(v) and "disagree" in str(v).lower() else norm_call(v))
h["disagreed"] = cons == "Disagree"
h["final"] = np.where(h["disagreed"], h["Final consensus"].map(norm_call), cons)

mapping, missing = alignment()
h = h[h["Sentence ID"].isin(mapping) & h["annotator1"].notna() & h["annotator2"].notna()].reset_index(drop=True)
say(f"Human sentences aligned to the current segmentation: {len(h)} (not located: {len(missing)}, all table rows)")

# ---- model calls on the human sentences
for i, d in enumerate(RUNS, 1):
    diag, core = ids(d / "diagnosis" / "lines.txt"), core_ids(d)
    h[f"run{i}"] = h["Sentence ID"].map(lambda s: any(x in diag for x in mapping[s]))
    h[f"core{i}"] = h["Sentence ID"].map(lambda s: any(x in core for x in mapping[s]))
run_cols = [f"run{i}" for i in range(1, 6)]
h["yes_count"] = h[run_cols].sum(axis=1)
h["majority"] = h["yes_count"] >= 3

B = pd.DataFrame({"Annotator 1": h["annotator1"] == "Yes", "Annotator 2": h["annotator2"] == "Yes"})
res = h["final"].notna()
FINAL = h.loc[res, "final"] == "Yes"


def k(a, b):
    return cohen_kappa_score(a, b), (np.asarray(a) == np.asarray(b)).mean()


say("\n== 1. Agreement with each human (binary Yes / not-Yes)")
kr, pr = k(B["Annotator 1"], B["Annotator 2"])
say(f"Annotator 1 vs Annotator 2 (human baseline): kappa {kr:.3f}, agreement {pr:.1%}")
rows = []
for who in ("Annotator 1", "Annotator 2"):
    per_run = [k(B[who], h[c]) for c in run_cols]
    km, pm = k(B[who], h["majority"])
    rows.append({"compared with": who, "kappa per run": "/".join(f"{x[0]:.2f}" for x in per_run),
                 "mean kappa": np.mean([x[0] for x in per_run]), "majority kappa": km, "majority agreement": pm})
    say(f"{who} vs model: per-run kappa {rows[-1]['kappa per run']} (mean {rows[-1]['mean kappa']:.3f}); "
        f"5-run majority kappa {km:.3f}, agreement {pm:.1%}")
per_run_final = [k(FINAL, h.loc[res, c]) for c in run_cols]
kf, pf = k(FINAL, h.loc[res, "majority"])
say(f"Final consensus vs model ({res.sum()} resolved sentences): per-run kappa "
    + "/".join(f"{x[0]:.2f}" for x in per_run_final) + f"; majority kappa {kf:.3f}, agreement {pf:.1%}")
for who in ("Annotator 1", "Annotator 2"):
    kk, pp = k(FINAL, B.loc[res, who])
    say(f"Final consensus vs {who}: kappa {kk:.3f}, agreement {pp:.1%}  (not independent: {who} helped set it)")

say("\n== 2. Run-to-run agreement on these sentences")
pair = [cohen_kappa_score(h[a], h[b]) for a, b in combinations(run_cols, 2)]
say(f"Mean pairwise kappa between runs {np.mean(pair):.3f} (range {min(pair):.2f}-{max(pair):.2f}); "
    f"same call in all five runs: {((h.yes_count == 0) | (h.yes_count == 5)).mean():.1%}")
say(f"Diagnosis sentences per run (on aligned human sentences): {h[run_cols].sum().tolist()}; "
    f"humans: Annotator 1 {int(B['Annotator 1'].sum())}, Annotator 2 {int(B['Annotator 2'].sum())}, final consensus {int(FINAL.sum())}")

say("\n== 3. The resolved human disagreements")
bin_dis = (B["Annotator 1"] != B["Annotator 2"]) & res          # disagreement on the binary call; Yes vs Maybe counts, Maybe vs No does not
dis = h[bin_dis]
fin_dis = dis["final"] == "Yes"
say(f"{len(dis)} resolved sentences where one annotator said Yes and the other did not.")
for who, col in (("Annotator 1", dis["annotator1"] == "Yes"), ("Annotator 2", dis["annotator2"] == "Yes"), ("model, 5-run majority", dis["majority"])):
    say(f"  final consensus sided with {who}: {(col == fin_dis).mean():.1%}")

say("\n== 4. Where the model departs from agreed human calls")
agree = B["Annotator 1"] == B["Annotator 2"]
added = h[agree & ~B["Annotator 1"] & h.majority]; dropped = h[agree & B["Annotator 1"] & ~h.majority]
say(f"Annotator 1 and Annotator 2 agree on {agree.sum()} sentences; the 5-run majority matches them on {(h.majority[agree] == B['Annotator 1'][agree]).mean():.1%}.")
say(f"  model added (both humans said not-Yes): {len(added)}  |  model dropped (both humans said Yes): {len(dropped)}")
say("  added: " + ", ".join(added["Sentence ID"])); say("  dropped: " + ", ".join(dropped["Sentence ID"]))

say("\n== 5. Core-theme sentences against the final consensus")
fy = FINAL.values
for i in range(1, 6):
    ext, core = h.loc[res, f"run{i}"].values, h.loc[res, f"core{i}"].values
    say(f"  run{i}: extracted {ext.sum()} (share the consensus calls Yes {fy[ext].mean():.0%}); "
        f"core {core.sum()} (share Yes {fy[core].mean():.0%}); consensus-Yes sentences recovered: extracted {ext[fy].mean():.0%}, core {core[fy].mean():.0%}")

say("\n== 6. Sanity checks (annotations/sanity_checks.txt)")
must, must_not, cur = [], [], None
for l in (ROOT / "annotations/sanity_checks.txt").read_text().splitlines():
    l = l.strip()
    if l.upper() == "MUST IDENTIFY": cur = must
    elif l.upper() == "MUST NOT IDENTIFY": cur = must_not
    elif l and cur is not None: cur.append(l)
hi = h.set_index("Sentence ID")
for c in run_cols:
    miss = [s for s in must if s in hi.index and not hi.loc[s, c]]
    wrong = [s for s in must_not if s in hi.index and hi.loc[s, c]]
    say(f"  {c}: {'PASS' if not miss and not wrong else 'FAIL'}" + (f"; missed {miss}" if miss else "") + (f"; wrongly flagged {wrong}" if wrong else ""))

(HERE / "summary_numbers.txt").write_text("\n".join(lines_out) + "\n")

# ---- figure: kappa against each reference, earlier extraction designs and the current pipeline
INK, INK_2, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIAG, DIAG_LIGHT = "#b5870e", "#eed593"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE, "text.color": INK,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
# Earlier designs, binary kappa, from agreement_analysis.ipynb sections 2 and 9 (original segmentation, 417 sentences)
earlier = {"Annotator 1": (0.578, 0.633), "Annotator 2": (0.401, 0.484)}
fig, ax = plt.subplots(figsize=(8.2, 3.9))
groups = ["Annotator 1", "Annotator 2"]
x = np.arange(len(groups)); w = 0.25
vals = [(earlier[g][0], earlier[g][1], [r for r in rows if r["compared with"] == g][0]["majority kappa"]) for g in groups]
labels = ("first prompt (single run)", "revised prompt (5-run majority)", "current LLM (5-run majority)")
cols = (BASELINE, DIAG_LIGHT, DIAG)
for j, (lab, col) in enumerate(zip(labels, cols)):
    v = [t[j] for t in vals]
    b = ax.bar(x + (j - 1) * w, v, w - 0.03, color=col, label=lab, zorder=2)
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.015, f"{vv:.2f}", ha="center", fontsize=8.5, color=INK_2)
ax.axhline(kr, color=INK, linewidth=1.2, linestyle="--", zorder=3)
ax.text(0.5, kr + 0.025, f"Annotator 1 vs Annotator 2: {kr:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"agreement with {g}" for g in groups])
ax.set_ylim(0, 1); ax.set_ylabel("Cohen's kappa (diagnosis, Yes vs not)")
ax.grid(axis="y", linewidth=0.6, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title("State Pension age, diagnosis: LLM against each human annotator", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "kappa_vs_humans.png", facecolor=SURFACE,
                                metadata={"Description": "Models: current pipeline Claude Sonnet 5.5; earlier bars from agreement_analysis.ipynb (earlier Sonnet)"})
print("-> figures/kappa_vs_humans.png")

# ---- report figure: humans with each other, the model with itself, the model with the consensus
fig, ax = plt.subplots(figsize=(7.8, 2.9))
pair_pct = np.mean([(h[a] == h[b]).mean() for a, b in combinations(run_cols, 2)])
bars = [("Researcher vs researcher\n(two annotators)", kr, pr, BASELINE),
        ("LLM vs itself\n(mean of the 10 run pairs)", np.mean(pair), pair_pct, DIAG_LIGHT),
        ("LLM vs human consensus\n(5-run majority)", kf, pf, DIAG)]
for i, (lab, v, pct, col) in enumerate(bars):
    ax.barh(i, v, 0.6, color=col, zorder=2)
    ax.text(v + 0.015, i, f"{v:.2f}   ({pct:.0%} of sentences agree)", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 1.32); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xlabel("Cohen's kappa (0 = chance, 1 = perfect agreement)")
ax.grid(axis="x", linewidth=0.6, color=GRID); ax.set_axisbelow(True)
ax.set_title("How much variation is there in assigning sentences to aspects?", loc="left", fontsize=11)
fig.tight_layout()
report_fig = ROOT / "annotations/spot_check_report/figures/fig8_human_agreement.png"
fig.savefig(report_fig, facecolor=SURFACE, metadata={"Description": "Models: Claude Sonnet 5.5 (current pipeline, 5 runs); human annotations from comparisons.csv"})
print("->", report_fig.relative_to(ROOT))
