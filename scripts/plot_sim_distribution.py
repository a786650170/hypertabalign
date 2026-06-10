"""Render KDE figure of similarity distributions.

Three curves on the same cosine-similarity axis:
  - positives  (q vs gold)
  - hard-mined (top-48 nearest from live KB index, gold excluded)
  - random    (uniform KB samples, gold excluded)
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---- font/style standard (paper-ready) ----
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

JSONL = "C:/Users/Administrator/_sim_distribution_v3.json"
OUT = "C:/Users/Administrator/_sim_distribution.png"

d = json.load(open(JSONL, "r", encoding="utf-8"))
pos = np.asarray(d["sims_positive"])
hard = np.asarray(d["sims_hard"])
rand = np.asarray(d["sims_random"])
print(f"loaded: pos={len(pos)}  hard={len(hard)}  rand={len(rand)}")

# common x axis
xs = np.linspace(-0.8, 1.02, 1024)

fig, ax = plt.subplots(figsize=(8.6, 5.0))


def shade(arr, color, label, lw=2.4, alpha_fill=0.18, bw=0.06):
    kde = gaussian_kde(arr, bw_method=bw)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, lw=lw, label=label, zorder=4)
    ax.fill_between(xs, 0, ys, color=color, alpha=alpha_fill, zorder=2)
    return ys


ys_rand = shade(rand, color="#1976D2", label=f"random KB negatives  (n={len(rand)})")
ys_hard = shade(hard, color="#EF6C00", label=f"live-mined hard negatives (top-48,  n={len(hard)})")
ys_pos  = shade(pos,  color="#2E7D32", label=f"positives  (n={len(pos)})", lw=2.8, alpha_fill=0.24, bw=0.08)

# annotate the means
means = {
    "random": (float(rand.mean()), "#1976D2"),
    "hard":   (float(hard.mean()), "#EF6C00"),
    "positive": (float(pos.mean()), "#2E7D32"),
}
ymax = max(ys_rand.max(), ys_hard.max(), ys_pos.max())
for lab, (m, c) in means.items():
    ax.axvline(m, color=c, ls="--", lw=1.2, alpha=0.65, zorder=3)
    ax.text(m, ymax * 0.97, f"mean={m:+.3f}", color=c, fontsize=11,
            ha="center", rotation=90, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
            zorder=5)

# shaded "useful gradient" band: where positive density is non-trivial
ax.axvspan(0.78, 1.02, color="green", alpha=0.06, zorder=1)
ax.text(0.90, ymax * 0.10, "useful contrastive\ngradient lives here",
        fontsize=12, color="#2E7D32", ha="center", va="center", style="italic",
        bbox=dict(facecolor="white", edgecolor="#2E7D32", alpha=0.95, lw=0.8, pad=3.5),
        zorder=5)

ax.set_xlabel("Cosine similarity to query")
ax.set_ylabel("Density")
ax.set_title("Similarity distributions under the trained retrieval head")
ax.set_xlim(-0.8, 1.02)
ax.set_ylim(0, ymax * 1.10)
ax.grid(True, axis="y", alpha=0.30, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", frameon=True, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print(f"wrote {OUT}")
