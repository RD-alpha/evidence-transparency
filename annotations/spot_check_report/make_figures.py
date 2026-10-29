"""Figures for the spot-check report draft ("A new LLM powered tool for evaluating evidence transparency").

Reads persisted artefacts only (no LLM calls). Re-run after each scoring batch:

    cd annotations/spot_check_report && python make_figures.py

Writes PNGs to figures/ and prints the numbers quoted in the report.

Data sources
- Extraction and whole-pipeline reliability: letting agent fees (earlier models) and State Pension age (Sonnet/Haiku 5.5),
  five full runs each (results/<doc>_rep{1..5}), as in letting_agents_reproducibility.ipynb. Shown side by side, not pooled.
- Scoring reliability (sentence assignments held fixed, core themes + grouping + scoring repeated 3x): earlier models =
  results/<doc>/stability/find_core_run{1..3} for five original documents + the five letting-agent extractions;
  5.5 models = the five State Pension age extractions. Each run is measured against its own set's median.
- 2018 comparison: every document in ../scored_documents.csv against ../2018_report_scores_all.csv,
  using the score the pipeline stored (mean of core themes), as in 2018_report_kendall_analysis.ipynb.
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS, DOCUMENTS, OUT = ROOT / "results", ROOT / "documents", HERE / "figures"
OUT.mkdir(exist_ok=True)

ASPECTS = ["diagnosis", "proposal", "implementation", "testing"]
CORE_ASPECTS = ASPECTS[:3]
LABEL = {a: a.capitalize() for a in ASPECTS} | {"testing": "Testing/Evaluation", "total": "Total"}

# Aspect colours: the web app's hues (web_app/static/style.css, --cat-<aspect>), brightened for charts; light tints are --cat-<aspect>-bg-strong
ASPECT_COLOUR = {"diagnosis": "#b5870e", "proposal": "#3a7ab3", "implementation": "#4f914f", "testing": "#cc5638"}
ASPECT_LIGHT = {"diagnosis": "#eed593", "proposal": "#b7d3e8", "implementation": "#b8d6ac", "testing": "#e6b6a0"}
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ASPECT_COLOUR |= {"total": INK_2}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "grid.color": GRID, "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


EARLIER, CURRENT = "Claude Sonnet 5 (grouping: Haiku 4.5)", "Claude Sonnet 5.5 (grouping: Haiku 5.5)"


def save(fig, name, models):
    """Save a PNG; the models behind it go in the PNG metadata, not on the figure."""
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE,
                metadata={"Title": name, "Description": f"Models: {models}", "Software": "annotations/spot_check_report/make_figures.py"})
    plt.close(fig)
    print(f"  -> figures/{name}")


def fleiss_kappa(counts):
    counts = np.asarray(counts, float); n = counts.sum(1)[0]
    p = counts.sum(0) / counts.sum()
    P = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p ** 2).sum()
    return (P.mean() - Pe) / (1 - Pe) if Pe < 1 else np.nan


def score_of(d, a):
    p = d / a / "scoring.json"
    return json.loads(p.read_text())["score"] if p.exists() else None


# ---------------------------------------------------------------- 1. Extraction reliability (two documents)
# Each reproducibility document: five full runs in results/<doc>_rep{1..5}/, plus two downstream repeats on each run's
# extraction in results/<doc>_rep{n}/stability/find_core_run{1,2}/. The letting-agent runs used the earlier models
# (Sonnet 5 / Haiku 4.5); State Pension age used Sonnet 5.5 / Haiku 5.5, so the documents are shown side by side, not pooled.
REPRO = {"letting_agent_fees": "Letting agent fees", "state_pension_age": "State Pension age"}
MODELS = {"letting_agent_fees": EARLIER, "state_pension_age": CURRENT}
MIXED_REPRO = (f"letting agent fees and the five original 2018 documents: {EARLIER}; State Pension age: {CURRENT}")
MIXED_2018 = (f"original nine and added eleven documents (20): {EARLIER}; extension batches 1-9 (44): {CURRENT}")


def extracted(d, a):
    p = d / a / "lines.txt"
    return {l.split()[0] for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def core(d, a):
    w, c = d / a / "weights.json", d / a / "classification.json"
    if not (w.exists() and c.exists()):
        return set()
    names = {t["theme"] for t in json.loads(w.read_text())["themes"] if t["weight"] == "core"}
    return {i for t in json.loads(c.read_text())["themes"] if t["name"] in names for i in t["ids"]}


def complete(d, extra=()):
    return all((x / a / "scoring.json").exists() for x in (d, *extra) for a in ASPECTS)


available = {}
for doc in REPRO:
    runs = [RESULTS / f"{doc}_rep{i}" for i in range(1, 6)]
    if all(complete(d) for d in runs):
        available[doc] = runs
    else:
        print(f"  (skipping {doc}: five complete runs not found yet)")

ext_rows, n_sent, flows = [], {}, {}
for doc, runs in available.items():
    sent = [l.split()[0] for l in (DOCUMENTS / doc / "input.txt").read_text().splitlines() if l.strip()]
    n_sent[doc], nr = len(sent), len(runs)
    print(f"\n== Extraction reliability: {REPRO[doc]}, {len(sent)} sentences, {nr} full runs")
    for a in ASPECTS:
        for kind, fn in (("all extracted", extracted), ("core", core)):
            if kind == "core" and a == "testing":
                continue                                   # testing has no core themes
            yes = pd.DataFrame({i: pd.Series(sent).isin(fn(d, a)).values for i, d in enumerate(runs)}).sum(axis=1)
            ever = yes[yes > 0]
            ext_rows.append({"doc": doc, "aspect": a, "set": kind, "fleiss": fleiss_kappa(np.column_stack([nr - yes, yes])) if len(ever) else np.nan,
                             "ever": len(ever), "all_runs": int((ever == nr).sum()), "share_all": (ever == nr).mean() if len(ever) else np.nan,
                             "per_run": [len(fn(d, a)) for d in runs]})
    sets = {a: [extracted(d, a) for d in runs] for a in ASPECTS}
    def label(s, r):
        got = [a for a in ASPECTS if s in sets[a][r]]
        return got[0] if len(got) == 1 else ("none" if not got else "multi")
    flow = Counter()
    for s in sent:
        for p, q in combinations([label(s, r) for r in range(nr)], 2):
            if p != q:
                flow[tuple(sorted((p, q)))] += 1
    flows[doc] = flow
    tot_flow = sum(flow.values())
    print("  Run-pair label disagreements:")
    for (p, q), n in flow.most_common(6):
        print(f"    {p} vs {q}: {n} ({n / tot_flow:.0%})")
ext = pd.DataFrame(ext_rows)
print(ext.drop(columns="doc").assign(doc=ext.doc).round(3).to_string(index=False))

from matplotlib.patches import Patch


def extraction_figure(docs, title, name, models):
    fig, axes = plt.subplots(len(docs), 2, figsize=(11, 3.4 * len(docs)), squeeze=False)
    x = np.arange(4); w = 0.36
    for r_, doc in enumerate(docs):
        sub = ext[ext.doc == doc].set_index(["aspect", "set"])
        for c_, (col, panel_title, fmt) in enumerate((("fleiss", "Fleiss' kappa across five runs", "{:.2f}"),
                                                ("share_all", "Sentences picked in all five runs (share of those picked in any run)", "{:.0%}"))):
            ax = axes[r_, c_]
            for k, (kind, palette) in enumerate((("all extracted", ASPECT_LIGHT), ("core", ASPECT_COLOUR))):
                for j, a in enumerate(ASPECTS):
                    xx = j if a == "testing" else j + (k - 0.5) * w       # testing has one bar only
                    if (a, kind) not in sub.index:
                        continue
                    val = sub.loc[(a, kind), col]
                    if np.isnan(val):
                        if kind == "all extracted":
                            ax.text(j, 0.04, "no sentences\nin any run", ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)
                        continue
                    ax.bar(xx, val, w - 0.03, color=palette[a], zorder=2)
                    ax.text(xx, val + 0.015, fmt.format(val), ha="center", va="bottom", fontsize=8, color=INK_2)
            ax.set_xticks(x); ax.set_xticklabels([LABEL[a].replace("/", "/\n") for a in ASPECTS], fontsize=9); ax.set_ylim(0, 1.1)
            ax.grid(axis="y", linewidth=0.6); ax.set_axisbelow(True)
            if c_ == 1:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title((f"{REPRO[doc]} ({n_sent[doc]} sentences)\n" if c_ == 0 else "\n") + panel_title,
                         fontsize=9.5, color=INK, loc="left")
    axes[0, 0].legend(handles=[Patch(color=BASELINE, label="all sentences in the aspect (light)"),
                               Patch(color=INK_2, label="core-theme sentences only (dark)")], frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, name, models)


extraction_figure(list(available), "Reliability of assigning sentences to framework aspects",
                  "fig1_extraction_reliability.png", MIXED_REPRO)

# ---------------------------------------------------------------- 2/3. Scoring and whole-pipeline reliability
print("\n== Scoring reliability (sentence assignments fixed; core themes, grouping and scoring repeated)")


def score_matrix(dirs):          # runs x aspects; a missing aspect (nothing extracted) scores 0, as in the pipeline
    m = np.array([[score_of(d, a) or 0 for a in ASPECTS] for d in dirs], float)
    return np.column_stack([m, m.sum(1)])


def deviations(groups):
    dev = {k: [] for k in ASPECTS + ["total"]}
    for m in groups:
        med = np.median(m, axis=0)
        for j, k in enumerate(ASPECTS + ["total"]):
            dev[k] += list(m[:, j] - med[j])
    return dev


def nested(doc):                 # 5 extractions x 3 downstream repeats, or None if the repeats are not all there
    runs = available.get(doc)
    if not runs:
        return None
    groups = [[d, d / "stability" / "find_core_run1", d / "stability" / "find_core_run2"] for d in runs]
    return [score_matrix(g) for g in groups] if all(complete(*g[:1], g[1:]) for g in groups) else None


# Panel A (earlier models): the five original documents with three find_core runs, plus the five letting-agent extractions.
fixed_old = [score_matrix([RESULTS / doc / "stability" / f"find_core_run{r}" for r in (1, 2, 3)])
             for doc in ["civil_service_apprenticeship", "furniture_fire_safety", "letting_agent_fees", "lobster_ia", "modern_languages_a_level"]]
nest = {doc: nested(doc) for doc in REPRO}
fixed_panels = [("Five 2018 documents and letting agent fees", f"10 fixed sentence sets", fixed_old + (nest["letting_agent_fees"] or []))]
if nest["state_pension_age"]:
    fixed_panels.append(("State Pension age", "5 fixed sentence sets", nest["state_pension_age"]))
full_panels = [(REPRO[d], "5 extractions x 3 downstream repeats", [np.vstack(nest[d])]) for d in REPRO if nest[d]]
for _, sub, mats in fixed_panels:
    print(f"  {sub}: totals", [m[:, 4].astype(int).tolist() for m in mats])


def components(X):               # one-way random effects: between-extraction and within-extraction variance
    n, k = X.shape
    msb = k * X.mean(1).var(ddof=1); msw = X.var(axis=1, ddof=1).mean()
    sb = max((msb - msw) / k, 0.0)
    return sb, msw


print("\n== Whole pipeline: 5 extractions x 3 downstream repeats per document")
for d in REPRO:
    if not nest[d]:
        continue
    tot = np.array([m[:, 4] for m in nest[d]])
    sb, sw = components(tot)
    print(f"  {REPRO[d]}: totals by extraction {tot.astype(int).tolist()}; median {np.median(tot):.0f}; "
          f"SD between extractions {np.sqrt(sb):.2f}, within {np.sqrt(sw):.2f}; extraction share {sb / (sb + sw) if sb + sw else float('nan'):.2f}")


def draw_deviation(ax, head, sub, mats, subtitle=None):
    keys = ASPECTS + ["total"]
    levels = np.arange(-2, 3)
    dev = deviations(mats); n = len(dev["total"])
    for i, k in enumerate(keys):
        c = Counter(np.round(dev[k]).astype(int))
        for lv in levels:
            cnt = c.get(lv, 0)
            if not cnt:
                continue
            ax.scatter(i, lv, s=30 + 620 * cnt / n, color=ASPECT_COLOUR[k], alpha=0.9 if lv == 0 else 0.75,
                       edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.text(i + 0.3, lv, f"{cnt}", va="center", fontsize=8, color=INK_2)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.axvline(3.5, color=GRID, linewidth=1, linestyle=":")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([LABEL[k].replace("/", "/\n").replace("Implementation", "Implemen-\ntation") for k in keys], fontsize=8.5)
    ax.set_yticks(levels); ax.set_yticklabels([f"{v:+d}" if v else "median" for v in levels]); ax.set_ylim(-2.6, 2.6)
    ax.set_xlim(-0.5, len(keys) - 0.3)
    ax.grid(axis="y", linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title(f"{head}\n{subtitle}" if subtitle else f"{head}\n{sub}: {n} runs", loc="left", fontsize=9.5, color=INK)
    print(f"  [{head}]", "; ".join(f"{k} max |dev| {np.abs(np.array(dev[k])).max():.0f}, at median {np.mean(np.array(dev[k]) == 0):.0%}" for k in keys))


def deviation_plot(panels, title, name, models):
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.1), sharey=True, squeeze=False)
    for ax, (head, sub, mats) in zip(axes[0], panels):
        draw_deviation(ax, head, sub, mats)
    axes[0, 0].set_ylabel("points from that set's median")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.905, "Number beside each dot = runs at that distance from their own set's median.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, name, models)


deviation_plot(fixed_panels, "Scoring reliability: same sentences, core themes + grouping + scoring repeated 3 times", "fig2_scoring_reliability.png", MIXED_REPRO)
deviation_plot(full_panels, "Whole-LLM-process reliability: every step repeated", "fig3_pipeline_reliability.png", MIXED_REPRO)

# ---------------------------------------------------------------- State Pension age on its own (figures/state_pension_age/)
SPA = "state_pension_age"
if SPA in available and nest[SPA]:
    print("\n== State Pension age only")
    sp = f"{SPA}/"
    extraction_figure([SPA], "Reliability of assigning sentences to framework aspects: State Pension age", sp + "spa1_extraction_reliability.png", CURRENT)
    allruns = np.vstack(nest[SPA])
    KEYS = ASPECTS + ["total"]

    def two_panel(w=9.4):
        return plt.subplots(1, 2, figsize=(w, 4.4), gridspec_kw={"width_ratios": [4, 1.1]})

    def finish(fig, ax, axt, sub, title, name, ylevels=True):
        for ax_, top in ((ax, 3), (axt, 12)):
            ax_.set_ylim(-0.4 * top / 3, top * 1.13); ax_.set_yticks(range(4) if top == 3 else range(0, 13, 3))
            ax_.grid(axis="y", linewidth=0.6); ax_.set_axisbelow(True)
        ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[k] for k in ASPECTS], fontsize=9)
        axt.set_xticks([0]); axt.set_xticklabels(["Document total"], fontsize=9)
        ax.set_ylabel("Score (aspects 0-3; document total 0-12)")
        ax.set_title(f"State Pension age\n{sub}", loc="left", fontsize=9.5, color=INK)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, sp + name, CURRENT)

    # spa3: every score from the 15 runs, as in annotations/scoring_reproducibility.ipynb (dots jittered, bar = mean)
    fig, ax, axt = (lambda f, a: (f, a[0], a[1]))(*two_panel())
    jr = np.random.default_rng(0)
    for ax_, keys_ in ((ax, ASPECTS), (axt, ["total"])):
        for i, k in enumerate(keys_):
            v = allruns[:, KEYS.index(k)]
            ax_.scatter(i + jr.uniform(-0.13, 0.13, len(v)), v, s=60, color=ASPECT_COLOUR[k], alpha=0.8, edgecolors=SURFACE, linewidths=0.8, zorder=3)
            ax_.hlines(v.mean(), i - 0.27, i + 0.27, color=INK, linewidth=1.6, zorder=4)
        ax_.set_xlim(-0.55, len(keys_) - 0.45)
    fig.text(0.01, 0.905, "Each dot = one run; black bar = mean of the 15 runs.", fontsize=8.5, color=INK_2)
    finish(fig, ax, axt, "every step repeated: each of 5 sets of sentence assignments was scored 3 times: 15 runs total",
           "Reliability of the whole process", "spa3_pipeline_reliability.png")

    # spa2a: left = absolute score of each extraction's median (of its 3 scorings); right = every run's distance from that median
    fig = plt.figure(figsize=(14.2, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1.1, 5.6], wspace=0.28)
    axa, axt, axd = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    meds = np.array([np.median(m, axis=0) for m in nest[SPA]])           # 5 extractions x (4 aspects + total)
    for ax_, keys_, top in ((axa, ASPECTS, 3), (axt, ["total"], 12)):
        for i, k in enumerate(keys_):
            for lv, cnt in Counter(np.round(meds[:, KEYS.index(k)]).astype(int)).items():
                ax_.scatter(i, lv, s=30 + 620 * cnt / len(meds), color=ASPECT_COLOUR[k], alpha=0.85, edgecolors=SURFACE, linewidths=2, zorder=3)
                ax_.text(i + 0.3, lv, f"{cnt}", va="center", fontsize=8, color=INK_2)
        ax_.set_xlim(-0.5, len(keys_) - 0.3); ax_.set_ylim(-0.4 * top / 3, top * 1.13); ax_.set_yticks(range(4) if top == 3 else range(0, 13, 3))
        ax_.grid(axis="y", linewidth=0.6); ax_.set_axisbelow(True)
    axa.set_xticks(range(4)); axa.set_xticklabels([LABEL[k].replace("/", "/\n").replace("Implementation", "Implemen-\ntation") for k in ASPECTS], fontsize=8.5)
    axt.set_xticks([0]); axt.set_xticklabels(["Document\ntotal"], fontsize=8.5)
    axa.set_ylabel("Median score of an extraction's 3 scorings")
    axa.set_title("Absolute scale\nnumber beside each dot = extractions (of 5)\nwith that median score", loc="left", fontsize=9.5, color=INK)
    draw_deviation(axd, "Relative scale", "", nest[SPA], subtitle="each of 5 sets of sentence assignments was scored 3 times: 15 runs total")
    fig.suptitle("Reliability of the scoring prompt: State Pension age", x=0.01, y=0.99, ha="left", fontsize=11)
    fig.text(0.01, 0.93, "Right: number beside each dot = runs at that distance from their own set's median.", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.14); save(fig, sp + "spa2a_scoring_reliability_two_scale.png", CURRENT)

    # spa2: the plain version. One cluster of 3 dots per extraction (scored 3 times), 5 clusters per aspect, raw scores.
    # Document total (0-12) gets its own panel on its own scale.
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [4, 1.05], "wspace": 0.08})
    gap, sp_ = 9.0, 1.6                                  # between aspects; between clusters
    for ax_, keys_ in ((ax, ASPECTS), (axt, ["total"])):
        for i, k in enumerate(keys_):
            j = KEYS.index(k)
            for e, m in enumerate(nest[SPA]):
                ax_.scatter(i * gap + e * sp_ + np.array([-0.2, 0, 0.2]), m[:, j], s=75, color=ASPECT_COLOUR[k], edgecolors=SURFACE, linewidths=1.2, zorder=3)
            ax_.text(i * gap + 2 * sp_, -0.78 if ax_ is ax else -3.1, "Document total" if k == "total" else LABEL[k], ha="center", va="top", fontsize=10, color=INK)
            if i:
                ax_.axvline(i * gap - (gap - 4 * sp_) / 2, color=GRID, linewidth=1)
        ax_.set_xticks([i * gap + e * sp_ for i in range(len(keys_)) for e in range(5)]); ax_.set_xticklabels([str(e + 1) for _ in keys_ for e in range(5)], fontsize=8.5)
        ax_.set_xlim(-1.2, (len(keys_) - 1) * gap + 4 * sp_ + 1.2)
        ax_.grid(axis="y", linewidth=0.6); ax_.set_axisbelow(True)
    ax.set_ylim(-0.4, 3.4); ax.set_yticks(range(4)); ax.set_ylabel("Score (0-3)")
    axt.set_ylim(-1.6, 13.6); axt.set_yticks(range(0, 13, 3)); axt.set_ylabel("Document total (0-12)")
    axt.yaxis.set_label_position("right"); axt.yaxis.tick_right(); axt.spines["right"].set_visible(True); axt.spines["left"].set_visible(False)
    ax.set_xlabel("Set of sentence assignments (1-5), within each aspect", labelpad=26, fontsize=9)
    ax.set_title("State Pension age\neach of 5 sets of sentence assignments was scored 3 times: 15 runs total", loc="left", fontsize=9.5, color=INK)
    fig.suptitle("Reliability of the scoring prompt", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.905, "Each cluster of 3 dots = the same sentence assignments scored three times.", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.77, bottom=0.2); save(fig, sp + "spa2_scoring_reliability.png", CURRENT)

    # Totals by extraction: the live run and two downstream repeats on each extraction
    tot = np.array([m[:, 4] for m in nest[SPA]])
    live = score_matrix([RESULTS / SPA])[0]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for e in range(len(tot)):
        for r, (col, lab) in enumerate(((INK_2, "full run"), (INK_MUTED, "repeat 1"), (BASELINE, "repeat 2"))):
            ax.scatter(e + (r - 1) * 0.09, tot[e, r], s=70, color=col, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=lab if e == 0 else None)
    ax.axhline(np.median(tot), color=INK_2, linewidth=1, linestyle="--", zorder=0)
    ax.text(len(tot) - 0.6, np.median(tot) - 0.45, f"median of the 15 runs: {np.median(tot):.0f}", fontsize=8.5, color=INK_2, ha="right")
    ax.axhline(live[4], color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)
    ax.text(len(tot) - 0.6, live[4] + 0.5, f"separate run used in the 2018 comparison: {live[4]:.0f}", fontsize=8.5, color=INK_2, ha="right")
    ax.set_xticks(range(len(tot))); ax.set_xticklabels([f"extraction {i + 1}" for i in range(len(tot))]); ax.set_xlim(-0.5, len(tot) - 0.5)
    ax.set_ylim(0, 12); ax.set_yticks(range(0, 13, 2)); ax.set_ylabel("total score (out of 12)")
    ax.grid(axis="y", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")
    ax.set_title("State Pension age: total score from each extraction, scored three times", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, sp + "spa4_totals_by_extraction.png", CURRENT)

# ---------------------------------------------------------------- 4. Performance against 2018
print("\n== 2018 comparison")
CATS = ["Diagnosis", "Proposal", "Implementation", "Testing"]
reg = pd.read_csv(HERE.parent / "scored_documents.csv")
all18 = pd.read_csv(HERE.parent / "2018_report_scores_all.csv").drop_duplicates("Name")
reg = reg.merge(all18[["Name"] + CATS], left_on="name_2018", right_on="Name", how="left")
assert reg["Name"].notna().all()
old = reg[CATS].values.astype(float)
new = np.array([[score_of(RESULTS / p, a) or 0 for a in ASPECTS] for p in reg["results_path"]], float)
N = len(reg); ok = ~np.isnan(old).any(1)
print(f"  {N} documents ({int(ok.sum())} with all four 2018 scores)")

diffs = {a: (new[:, j] - old[:, j])[~np.isnan(old[:, j])] for j, a in enumerate(ASPECTS)}
diffs["total"] = new[ok].sum(1) - old[ok].sum(1)
for k, d in diffs.items():
    print(f"  {k}: mean shift {d.mean():+.2f}; lower/equal/higher {int((d < 0).sum())}/{int((d == 0).sum())}/{int((d > 0).sum())}")
rng = np.random.default_rng(0)
boot = [diffs["total"][rng.integers(0, len(diffs["total"]), len(diffs["total"]))].mean() for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  Total: mean 2018 {old[ok].sum(1).mean():.2f}, pipeline {new[ok].sum(1).mean():.2f}; shift {diffs['total'].mean():.2f} (95% CI {lo:.2f} to {hi:.2f})")

fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.25)
for j, a in enumerate(ASPECTS):
    ax = fig.add_subplot(gs[0, j]); d = diffs[a]; vals = np.arange(-3, 4)
    cnt = np.array([(d == v).sum() for v in vals])
    ax.bar(vals, cnt, 0.82, color=[ASPECT_COLOUR[a] if v else BASELINE for v in vals], zorder=2)
    for v, c in zip(vals, cnt):
        if c: ax.text(v, c + 0.6, str(c), ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(vals); ax.set_xticklabels([f"{v:+d}" if v else "0" for v in vals], fontsize=8)
    ax.set_title(f"{LABEL[a]}\nmean {d.mean():+.2f}", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, max(cnt) * 1.18); ax.grid(axis="y", linewidth=0.6); ax.set_axisbelow(True)
    if j == 0: ax.set_ylabel("documents")
ax = fig.add_subplot(gs[1, :]); d = diffs["total"]; vals = np.arange(int(d.min()) - 1, int(d.max()) + 2)
cnt = np.array([(d == v).sum() for v in vals])
ax.bar(vals, cnt, 0.82, color=[INK_2 if v else BASELINE for v in vals], zorder=2)
for v, c in zip(vals, cnt):
    if c: ax.text(v, c + 0.25, str(c), ha="center", fontsize=8, color=INK_2)
ax.axvline(d.mean(), color=INK, linewidth=1.5, linestyle="--", zorder=3)
ax.text(d.mean() + 0.12, max(cnt) * 1.32, f"mean {d.mean():+.1f} points", color=INK, fontsize=9, ha="left", va="top")
ax.set_xticks(vals); ax.set_xticklabels([f"{v:+d}" if v else "0" for v in vals])
ax.set_xlabel("LLM minus 2018 (document total, out of 12)"); ax.set_ylabel("documents")
ax.set_ylim(0, max(cnt) * 1.35); ax.grid(axis="y", linewidth=0.6); ax.set_axisbelow(True)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_title(f"Document total: {int((d < 0).sum())} lower, {int((d == 0).sum())} equal, {int((d > 0).sum())} higher",
             fontsize=9.5, color=INK, loc="left")
fig.suptitle("LLM score minus 2018 score", x=0.01, ha="left", fontsize=11, y=1.0)
fig.text(0.01, 0.955, "Negative = the LLM scored lower. Each aspect is out of 3; the total is out of 12.", fontsize=8.5, color=INK_2)
save(fig, "fig4_shift_from_2018.png", MIXED_2018)

TAU_LABEL = LABEL | {"total": "Document total"}

# Tau-b per aspect and total, with document-bootstrap 95% intervals
rows = []
keys = ASPECTS + ["total"]
for j, k in enumerate(keys):
    if k == "total":
        o, n = old[ok].sum(1), new[ok].sum(1)
    else:
        m = ~np.isnan(old[:, j]); o, n = old[m, j], new[m, j]
    r = kendalltau(o, n, variant="b")
    bs = []
    for _ in range(5000):
        ix = rng.integers(0, len(o), len(o))
        if o[ix].std() > 0 and n[ix].std() > 0:
            bs.append(kendalltau(o[ix], n[ix], variant="b").statistic)
    rows.append({"key": k, "n": len(o), "tau": r.statistic, "p": r.pvalue, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
taus = pd.DataFrame(rows).set_index("key")
print(taus.round(4).to_string())

fig, ax = plt.subplots(figsize=(8.4, 3.6))
for i, k in enumerate(keys):
    r = taus.loc[k]
    sig = r.p < 0.05                                   # not significant -> dotted interval
    ax.plot([r.lo, r.hi], [i, i], color=ASPECT_COLOUR[k], linewidth=2, linestyle="-" if sig else (0, (1, 2)),
            solid_capstyle="round", dash_capstyle="round", zorder=2)
    ax.scatter(r.tau, i, s=90, color=ASPECT_COLOUR[k], edgecolors=SURFACE, linewidths=2, zorder=3)
    ptxt = "p < 0.001" if r.p < 0.001 else f"p = {r.p:.3f}"
    ax.text(1.02, i, rf"$\tau_b$ = {r.tau:.2f}   {ptxt}", va="center", fontsize=8.5, color=INK_2, transform=ax.get_yaxis_transform())
ax.axvline(0, color=BASELINE, linewidth=1)
ax.set_yticks(range(len(keys))); ax.set_yticklabels([TAU_LABEL[k] for k in keys]); ax.invert_yaxis()
ax.set_xlim(-0.1, 1); ax.set_xlabel(r"Kendall's $\tau_b$ with the 2018 score (dot) and 95% interval over documents (line; dotted = p ≥ 0.05)")
ax.grid(axis="x", linewidth=0.6); ax.set_axisbelow(True)
ax.set_title("Do the LLM and the 2018 scorers rank documents the same way?", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "fig5_tau_b.png", MIXED_2018)

# Alternative without intervals: bars, solid where p < 0.05 and hatched where not
fig, ax = plt.subplots(figsize=(7.6, 3.4))
for i, k in enumerate(keys):
    r = taus.loc[k]; sig = r.p < 0.05
    ax.barh(i, r.tau, 0.62, color=ASPECT_COLOUR[k] if sig else SURFACE, edgecolor=ASPECT_COLOUR[k],
            linewidth=1.5, hatch=None if sig else "////", zorder=2)
    ptxt = "p < 0.001" if r.p < 0.001 else f"p = {r.p:.3f}"
    ax.text(r.tau + 0.015, i, f"{r.tau:.2f}   ({ptxt}" + (")" if sig else ", not statistically significant)"),
            va="center", fontsize=9, color=INK if sig else INK_2)
ax.set_yticks(range(len(keys))); ax.set_yticklabels([TAU_LABEL[k] for k in keys]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel(r"Kendall's $\tau_b$ with the 2018 score (0 = no shared ordering, 1 = identical ordering)")
ax.grid(axis="x", linewidth=0.6); ax.set_axisbelow(True)
ax.set_title("Do the LLM and the 2018 scorers rank documents the same way?", loc="left", fontsize=11, color=INK)
fig.tight_layout(); save(fig, "fig5b_tau_b_simple.png", MIXED_2018)

# ---------------------------------------------------------------- 5. 2018 score against pipeline score (scatter)
jit = lambda v, seed, a=0.12: v + np.random.default_rng(seed).uniform(-a, a, size=len(v))

ot, nt = old[ok].sum(1), new[ok].sum(1)
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.scatter(jit(ot, 1), jit(nt, 2), s=55, color=INK_2, edgecolors=SURFACE, linewidths=1, alpha=0.85, zorder=3)
ax.plot([-0.5, 12.5], [-0.5, 12.5], color=BASELINE, linewidth=1, linestyle="--", zorder=0)
ax.text(11.6, 12.1, "same score", fontsize=8, color=INK_MUTED, ha="right", rotation=45, rotation_mode="anchor")
ax.set_xlim(-0.5, 12.5); ax.set_ylim(-0.5, 12.5); ax.set_aspect("equal")
ax.set_xticks(range(0, 13, 2)); ax.set_yticks(range(0, 13, 2))
ax.set_xlabel("2018 document total"); ax.set_ylabel("LLM document total")
ax.grid(linewidth=0.6); ax.set_axisbelow(True)
ax.set_title(f"Document totals, 2018 against LLM\npoints jittered to separate ties",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); save(fig, "fig6_scatter_totals.png", MIXED_2018)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
for j, (ax, a) in enumerate(zip(axes, ASPECTS)):
    m = ~np.isnan(old[:, j]); o, n = old[m, j], new[m, j]
    ax.scatter(jit(o, 3 + j, 0.14), jit(n, 7 + j, 0.14), s=34, color=ASPECT_COLOUR[a], alpha=0.75, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.plot([-0.4, 3.4], [-0.4, 3.4], color=BASELINE, linewidth=1, linestyle="--", zorder=0)
    ax.set_xlim(-0.45, 3.45); ax.set_ylim(-0.45, 3.45); ax.set_aspect("equal")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.grid(linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("2018 score")
    ax.set_title(LABEL[a], loc="left", fontsize=10, color=INK)
axes[0].set_ylabel("LLM score")
fig.suptitle(f"Aspect scores, 2018 against LLM (points jittered to separate ties; dashed line = same score)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "fig7_scatter_aspects.png", MIXED_2018)

# ---------------------------------------------------------------- 6. Calibration of document totals to the 2018 scale
# Constant offset: 2018 total = pipeline total + c, c = the mean difference (documents with all four 2018 scores);
# then each policy's squared deviation from its calibrated score. A fitted line (2018 = a + b x pipeline) is printed
# for comparison only: its slope's interval includes 1 and it does no better out of sample. Leave-one-out MSE guards
# against judging a calibration on the documents it was fitted to. Writes calibration_by_policy.csv.
print("\n== Calibration of document totals")
y, x = old[ok].sum(1), new[ok].sum(1); names_ok = reg["name_2018"].values[ok]; n_ok = len(y)
b_cal, a_cal = np.polyfit(x, y, 1)
c_off = (y - x).mean()
resid = y - (x + c_off)


def loo(pred):
    return np.array([y[i] - pred(x[np.arange(n_ok) != i], y[np.arange(n_ok) != i], x[i]) for i in range(n_ok)])


cv = {"constant offset": loo(lambda xt, yt, xi: xi + (yt - xt).mean()),
      "linear": loo(lambda xt, yt, xi: np.polyval(np.polyfit(xt, yt, 1), xi)),
      "2018 average only": loo(lambda xt, yt, xi: yt.mean())}
rng_c = np.random.default_rng(0)
slopes = [np.polyfit(x[i], y[i], 1)[0] for i in (rng_c.integers(0, n_ok, n_ok) for _ in range(4000))]
print(f"  2018 = {a_cal:.2f} + {b_cal:.2f} x pipeline (slope 95% CI {np.percentile(slopes, 2.5):.2f} to {np.percentile(slopes, 97.5):.2f})")
print(f"  MSE raw {np.mean((y - x) ** 2):.2f}; constant offset {np.var(y - x):.2f} (LOO {np.mean(cv['constant offset'] ** 2):.2f}); "
      f"linear {np.mean((y - a_cal - b_cal * x) ** 2):.2f} (LOO {np.mean(cv['linear'] ** 2):.2f}); 2018 average only {np.var(y):.2f} (LOO {np.mean(cv['2018 average only'] ** 2):.2f})")
print(f"  constant offset c = {c_off:+.2f}")
print(f"  after offset calibration: RMSE {np.sqrt(np.mean(resid ** 2)):.2f}; median squared deviation {np.median(resid ** 2):.2f}; "
      f"within 2 points {np.mean(abs(resid) <= 2):.0%}; variance explained (LOO) {1 - np.mean(cv['constant offset'] ** 2) / np.mean(cv['2018 average only'] ** 2):.0%}")
cal = pd.DataFrame({"policy": names_ok, "2018 total": y.astype(int), "pipeline total": x.astype(int),
                    "calibrated (pipeline + offset)": np.round(x + c_off, 2), "deviation (2018 - calibrated)": np.round(resid, 2),
                    "squared deviation": np.round(resid ** 2, 2)}).sort_values("squared deviation", ascending=False)
cal.to_csv(HERE / "calibration_by_policy.csv", index=False)
print("  -> calibration_by_policy.csv")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
ax.scatter(jit(x, 11), jit(y, 12), s=50, color=INK_2, edgecolors=SURFACE, linewidths=1, alpha=0.85, zorder=3)
xs = np.array([-0.5, 12.5])
ax.plot(xs, xs, color=BASELINE, linewidth=1, linestyle="--", zorder=1, label="same score")
ax.plot(xs, xs + c_off, color=INK, linewidth=1.8, zorder=2, label=f"calibration: 2018 = LLM + {c_off:.1f}")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_xlim(-0.5, 12.5); ax.set_ylim(-0.5, 12.5); ax.set_aspect("equal")
ax.set_xticks(range(0, 13, 2)); ax.set_yticks(range(0, 13, 2))
ax.set_xlabel("LLM document total"); ax.set_ylabel("2018 document total"); ax.grid(linewidth=0.6); ax.set_axisbelow(True)
ax.set_title("Calibrating LLM totals to the 2018 scale\n", loc="left", fontsize=10.5, color=INK)

vals_r, counts_r = np.unique(np.round(resid, 2), return_counts=True)
ax2.axvspan(-2, 2, color=GRID, alpha=0.5, zorder=0)
ax2.axvline(0, color=BASELINE, linewidth=1)
ax2.bar(vals_r, counts_r, 0.8, color=INK_2, zorder=2)
for v_, c_ in zip(vals_r, counts_r):
    ax2.text(v_, c_ + 0.3, str(c_), ha="center", fontsize=8.5, color=INK_2)
ax2.set_xlim(-6.5, 6.5); ax2.set_ylim(0, counts_r.max() * 1.3)
ax2.set_xlabel("2018 total minus calibrated LLM total (points)"); ax2.set_ylabel("policies")
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax2.grid(axis="y", linewidth=0.6); ax2.set_axisbelow(True)
ax2.text(0, counts_r.max() * 1.2, f"{np.mean(abs(resid) <= 2):.0%} of policies within 2 points (shaded)", ha="center", fontsize=8.5, color=INK_2)
ax2.set_title(f"Deviation per policy after calibration\nmean squared deviation {np.mean(resid ** 2):.1f}; typical deviation (RMSE) {np.sqrt(np.mean(resid ** 2)):.1f} points",
              loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); save(fig, "fig9_calibration.png", MIXED_2018)
