"""Plot the per-epoch training loss for RoCEL Attempt 1, RoCEL Attempt 2
(patched with collapse guard), and a HyperTabAlign reference run.

Output: _loss_curves.png  -- a publication-ready figure for paper Section V-F.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT = "C:/Users/Administrator/_loss_curves.png"

# ---- data (parsed from training logs) ----
rocel_a1_x = [1, 2, 3, 4]
rocel_a1_y = [4.2467, 0.0000, 0.0000, 0.0000]

rocel_a2_x = [1, 2]
rocel_a2_y = [4.2599, 0.0000]

hyper_x = [1, 2, 3, 4, 5]
hyper_y = [4.0200, 1.0233, 0.7076, 0.6120, 0.5793]

# ---- plot ----
fig, ax = plt.subplots(figsize=(8.4, 5.0))

ax.set_xlim(0.65, 5.7)
ax.set_ylim(-0.6, 5.4)

# shaded collapse zone (drawn FIRST so it's behind lines)
ax.axhspan(-0.20, 0.12, color="gray", alpha=0.18, zorder=1)
ax.text(5.65, -0.04, "rank-1 collapse zone",
        fontsize=12, color="#333", ha="right", va="center", style="italic", zorder=2)

# HyperTabAlign clean curve
ax.plot(hyper_x, hyper_y, "o-", color="#2E7D32", linewidth=2.6, markersize=10,
        label="HyperTabAlign (clean convergence)", zorder=4)
for x, y in zip(hyper_x, hyper_y):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 14), textcoords="offset points",
                fontsize=12, color="#2E7D32", ha="center", zorder=5)

# RoCEL Attempt 1
ax.plot(rocel_a1_x, rocel_a1_y, "s-", color="#C62828", linewidth=2.4, markersize=10,
        label="RoCEL Attempt 1 (no collapse guard)", zorder=4)
ax.annotate(f"{rocel_a1_y[0]:.2f}", (rocel_a1_x[0], rocel_a1_y[0]),
            xytext=(15, 8), textcoords="offset points",
            fontsize=12, color="#C62828", ha="left", zorder=5)
ax.annotate("collapse at epoch 2,\nsilently saved as best",
            xy=(rocel_a1_x[1], rocel_a1_y[1]),
            xytext=(2.45, 1.65),
            fontsize=12, color="#C62828", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.4),
            zorder=5)

# RoCEL Attempt 2 (patched)
rocel_a2_x_plot = [x + 0.06 for x in rocel_a2_x]
ax.plot(rocel_a2_x_plot, rocel_a2_y, "^-", color="#EF6C00", linewidth=2.4, markersize=11,
        label="RoCEL Attempt 2 (with collapse guard)", zorder=4)
ax.annotate(f"{rocel_a2_y[0]:.2f}", (rocel_a2_x_plot[0], rocel_a2_y[0]),
            xytext=(10, -16), textcoords="offset points",
            fontsize=12, color="#EF6C00", ha="left", zorder=5)
ax.annotate("epoch 1: 99.97% steps NaN, FLAGGED FAILED\nepoch 2: 100% NaN, FLAGGED FAILED",
            xy=(rocel_a2_x_plot[1], rocel_a2_y[1]),
            xytext=(2.85, 3.05),
            fontsize=12, color="#EF6C00", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#EF6C00", lw=1.3),
            zorder=5)

# axes
ax.set_xlabel("Training epoch")
ax.set_ylabel("End-of-epoch training loss")
ax.set_title("RoCEL training collapses on WDC LSPM 8M; HyperTabAlign converges cleanly")
ax.set_xticks([1, 2, 3, 4, 5])
ax.grid(True, axis="y", alpha=0.30, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print(f"wrote {OUT}")
