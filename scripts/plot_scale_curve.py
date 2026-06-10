"""Accuracy vs KB scale on WDC LSPM eval (5,170-cell subset).

For each KB size in {100k, 500k, 2M, 8M} we restrict the candidate pool to
that many KB entities (all eval-gold IDs included, plus uniformly sampled
distractors) and report retrieval Accuracy on the same 5,170 labelled
queries.  Both methods use the minimal serialisation protocol (raw cell
text, no header prepend); absolute numbers therefore differ from the
header-aware headline numbers in Table I.  The point of this figure is
the shape of the curve: how Accuracy degrades as the distractor pool
grows from 10^5 to 10^7.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- font/style standard, matching the other paper figures ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'axes.titlesize': 17,
    'axes.titleweight': 'bold',
    'figure.dpi': 220,
    'savefig.bbox': 'tight',
})

# data from SCALE_RESULT lines in ./results/log_scale_{hypertab,rsupcon}_v2.txt
DATA = {
    "HyperTabAlign-direct": dict(
        scales=[100_000, 500_000, 2_000_000, 8_043_290],
        acc   =[0.9184,    0.8890,    0.8584,     0.8213],
        hit5  =[0.9545,    0.9356,    0.9093,     0.8793],
        hit10 =[0.9613,    0.9470,    0.9248,     0.8996],
        mrr10 =[0.9344,    0.9091,    0.8813,     0.8477],
    ),
    "R-SupCon": dict(
        scales=[100_000, 500_000, 2_000_000, 8_043_290],
        acc   =[0.9246,    0.8959,    0.8625,     0.8232],
        hit5  =[0.9547,    0.9364,    0.9128,     0.8822],
        hit10 =[0.9598,    0.9462,    0.9255,     0.9010],
        mrr10 =[0.9380,    0.9135,    0.8840,     0.8500],
    ),
}

OUT = "C:/Users/Administrator/_scale_curve.png"

colors = {
    "HyperTabAlign-direct": "#2E7D32",  # green
    "R-SupCon":             "#1976D2",  # blue
}
markers = {
    "HyperTabAlign-direct": "o",
    "R-SupCon":             "s",
}

fig, ax = plt.subplots(figsize=(8.6, 5.2))

for name, d in DATA.items():
    ax.plot(d["scales"], d["acc"], marker=markers[name], markersize=12,
            color=colors[name], linewidth=2.8, label=name, zorder=4)
    # annotate each point with the Acc value
    for x, y in zip(d["scales"], d["acc"]):
        dy = 0.012 if name == "HyperTabAlign-direct" else -0.018
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 14 if dy > 0 else -14),
                    textcoords="offset points", fontsize=11,
                    color=colors[name], ha="center",
                    va="bottom" if dy > 0 else "top", zorder=5)

ax.set_xscale("log")
ax.set_xlabel(r"KB size  (number of candidate entities, log scale)")
ax.set_ylabel("Top-1 Accuracy")
ax.set_title("Retrieval accuracy degrades as candidate pool scales to $\\mathbf{8 \\times 10^6}$")

# x-tick formatting in clean human form
ticks = [100_000, 500_000, 2_000_000, 8_043_290]
labels = ["100 k", "500 k", "2 M", "8 M"]
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

ax.set_ylim(0.78, 0.96)
ax.grid(True, which="major", alpha=0.30, zorder=0)
ax.grid(True, which="minor", alpha=0.10, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotate the headline drop on each curve
for name, d in DATA.items():
    drop = (d["acc"][0] - d["acc"][-1]) * 100
    ax.annotate(
        f"$\\Delta$ = {drop:.1f} pts\n(from 100 k to 8 M)",
        xy=(d["scales"][-1], d["acc"][-1]),
        xytext=(0.55, 0.06 if name == "HyperTabAlign-direct" else 0.16),
        textcoords="axes fraction",
        fontsize=12, color=colors[name], ha="left",
        bbox=dict(facecolor="white", edgecolor=colors[name], lw=1.0, alpha=0.95, pad=3),
    )

ax.legend(loc="upper right", frameon=True, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print(f"wrote {OUT}")
