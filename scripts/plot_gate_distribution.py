"""Render the gate-weight (alpha_v) distribution figure.

Per-cell, per-layer alpha = softmax([logit_row, logit_col]) so
alpha_row + alpha_col = 1. The trained model's alpha_row is heavily
right-skewed (mean ~0.98), so a flat violin/box on [0,1] is unreadable.
We use a stacked histogram of alpha_col on a zoomed x-axis to make the
spread visible, plus a per-layer mean annotation. The point is to show
the per-cell gate is actively learned (not stuck at uniform), not to
sell a particular row-vs-col bias.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

JSONL = "C:/Users/Administrator/gate_alpha_dump.jsonl"
OUT = "C:/Users/Administrator/_gate_distribution.png"


def load_alpha(path, layer_keys=("0", "1")):
    by_layer = {int(k): [] for k in layer_keys}
    gamma = None
    n_used = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_header"):
                continue
            if gamma is None and r.get("gamma") is not None:
                gamma = float(r["gamma"])
            ap = r.get("alpha_per_layer") or {}
            tmp = {}
            ok = True
            for k in layer_keys:
                v = ap.get(k)
                if v is None or len(v) != 2:
                    ok = False; break
                tmp[int(k)] = (float(v[0]), float(v[1]))  # (alpha_row, alpha_col)
            if not ok:
                continue
            for L, ac in tmp.items():
                by_layer[L].append(ac)
            n_used += 1
    print(f"used n_used={n_used}")
    out = {}
    for L, lst in by_layer.items():
        arr = np.asarray(lst)  # [N, 2]
        out[L] = dict(alpha_row=arr[:, 0], alpha_col=arr[:, 1])
    return out, gamma


by_layer, gamma = load_alpha(JSONL)
layers = sorted(by_layer.keys())
for L in layers:
    a = by_layer[L]['alpha_row']
    print(f"layer {L}:  n={len(a)}  mean(alpha_row)={a.mean():.3f}  "
          f"median={np.median(a):.3f}  std={a.std():.3f}  "
          f"p1={np.percentile(a,1):.3f}  p5={np.percentile(a,5):.3f}")

# ---- two-panel figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.28))
colors = ["#1976D2", "#EF6C00"]  # layer 1 blue, layer 2 orange

# ---- Panel A: histogram of alpha_row, zoomed to [0.5, 1.02] where the data lives ----
# (almost no cells have alpha_row < 0.5; full [0,1] range wastes 70% of the panel)
bins = np.linspace(0.50, 1.00, 51)
for i, L in enumerate(layers):
    arr = by_layer[L]['alpha_row']
    arr_in_range = arr[arr >= 0.50]
    ax1.hist(arr_in_range, bins=bins, color=colors[i], alpha=0.55, edgecolor="black",
             linewidth=0.5, label=f"Layer {L+1} (mean = {arr.mean():.3f})",
             zorder=3)

ax1.set_yscale("log")
ax1.set_xlim(0.50, 1.015)
ax1.set_xlabel(r"$\alpha_{\mathrm{row}}$  (higher = more row-dominant; $0.5$ = uniform)")
ax1.set_ylabel("Count of cells  (log scale)")
ax1.set_title(r"(a) Per-cell $\alpha_{\mathrm{row}}$ distribution  (zoomed to $[0.5,\,1.0]$)")
ax1.grid(True, axis="y", alpha=0.30, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper left", frameon=True, framealpha=0.95)

# ---- Panel B: ECDF, makes the spread (and how many cells deviate from row-dominant) visible ----
for i, L in enumerate(layers):
    arr = np.sort(by_layer[L]['alpha_row'])
    n = len(arr)
    y = np.arange(1, n + 1) / n
    ax2.plot(arr, y, color=colors[i], lw=2.6, label=f"Layer {L+1}", zorder=3)

ax2.axvline(0.5, color="gray", lw=1.3, ls="--", alpha=0.75, zorder=2)
ax2.set_xlim(0.0, 1.02)
ax2.set_ylim(0.0, 1.02)
ax2.set_xlabel(r"$\alpha_{\mathrm{row}}$  (per-cell row-vs-col gate weight)")
ax2.set_ylabel(r"ECDF  (fraction of cells with $\alpha_{\mathrm{row}} \leq x$)")
ax2.set_title(r"(b) ECDF of $\alpha_{\mathrm{row}}$")
ax2.grid(True, alpha=0.30, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper left", frameon=True, framealpha=0.95)

# annotate the p5 / p50 / p95 of layer 2 (final block)
arr2 = by_layer[1]['alpha_row']
for q, lab in [(0.05, "5%"), (0.50, "50%"), (0.95, "95%")]:
    v = float(np.quantile(arr2, q))
    ax2.scatter([v], [q], s=70, color="#C62828", edgecolor="white", linewidth=1.3, zorder=5)
    ax2.text(v, q + 0.04, f"{lab}: $\\alpha_{{\\mathrm{{row}}}}={v:.3f}$", fontsize=11,
             color="#C62828", ha="center", va="bottom",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

# overall title (one line)
title = f"Per-cell gate weights are actively learned, not stuck at uniform initialisation"
if gamma is not None:
    title += f"  ($\\gamma = {gamma:.3f}$)"
fig.suptitle(title, fontsize=17, fontweight="bold", y=1.00)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print(f"wrote {OUT}")
