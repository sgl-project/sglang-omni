#!/usr/bin/env python3
"""Generate benchmark plots for S2-Pro blog post."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

RESULTS_DIR = "results/s2pro_vendored_full"
OUT_DIR = "results/s2pro_vendored_full/plots"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(f"{RESULTS_DIR}/results.csv")
wer = json.load(open(f"{RESULTS_DIR}/wer.json"))
wer_values = [s["wer"] for s in wer["per_sample"]]

# ── Figure 1: Key metrics summary (horizontal bar) ──
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

metrics = [
    ("Throughput\n(tok/s)", df["tok_per_s"].mean(), df["tok_per_s"].median(), "tok/s", "#4C72B0"),
    ("RTF", df["rtf"].mean(), df["rtf"].median(), "", "#DD8452"),
    ("TTFB\n(ms)", df["ttfb_ms"].mean(), df["ttfb_ms"].median(), "ms", "#55A868"),
    ("WER\n(%)", np.mean(wer_values) * 100, np.median(wer_values) * 100, "%", "#C44E52"),
]

for ax, (label, mean, median, unit, color) in zip(axes, metrics):
    bars = ax.bar(["Mean", "Median"], [mean, median], color=color, width=0.5, alpha=0.85)
    for bar, val in zip(bars, [mean, median]):
        fmt = f"{val:.1f}" if val >= 1 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                fmt, ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel(unit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("S2 on SGLang — seed-tts-eval EN (1088 samples, 1×H200)", fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/summary_metrics.png")
plt.savefig(f"{OUT_DIR}/summary_metrics.svg")
plt.close()

# ── Figure 2: RTF distribution histogram ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["rtf"], bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Real-time (RTF=1.0)")
ax.axvline(x=df["rtf"].mean(), color="#DD8452", linestyle="-", linewidth=2, label=f'Mean RTF={df["rtf"].mean():.3f}')
ax.set_xlabel("Real-Time Factor (RTF)")
ax.set_ylabel("Count")
ax.set_title("RTF Distribution — S2 on SGLang (1×H200, BS=1)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rtf_distribution.png")
plt.savefig(f"{OUT_DIR}/rtf_distribution.svg")
plt.close()

# ── Figure 3: Throughput distribution ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["tok_per_s"], bins=50, color="#55A868", alpha=0.8, edgecolor="white")
ax.axvline(x=df["tok_per_s"].mean(), color="#DD8452", linestyle="-", linewidth=2,
           label=f'Mean={df["tok_per_s"].mean():.1f} tok/s')
ax.set_xlabel("Throughput (tok/s)")
ax.set_ylabel("Count")
ax.set_title("Per-Request Throughput — S2 on SGLang (1×H200, BS=1)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/throughput_distribution.png")
plt.savefig(f"{OUT_DIR}/throughput_distribution.svg")
plt.close()

# ── Figure 4: WER distribution ──
fig, ax = plt.subplots(figsize=(8, 4))
wer_pct = [w * 100 for w in wer_values]
ax.hist(wer_pct, bins=50, color="#C44E52", alpha=0.8, edgecolor="white")
ax.axvline(x=np.mean(wer_pct), color="#DD8452", linestyle="-", linewidth=2,
           label=f'Mean WER={np.mean(wer_pct):.2f}%')
ax.set_xlabel("Word Error Rate (%)")
ax.set_ylabel("Count")
ax.set_title("WER Distribution — S2 on SGLang (1×H200, seed-tts-eval EN)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/wer_distribution.png")
plt.savefig(f"{OUT_DIR}/wer_distribution.svg")
plt.close()

# ── Figure 5: Latency vs generated tokens (scatter) ──
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df["gen_tokens"], df["latency_s"], c=df["tok_per_s"],
                cmap="viridis", alpha=0.6, s=15, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax, label="tok/s")
ax.set_xlabel("Generated Tokens")
ax.set_ylabel("Latency (s)")
ax.set_title("Latency vs Output Length — S2 on SGLang (1×H200)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/latency_vs_tokens.png")
plt.savefig(f"{OUT_DIR}/latency_vs_tokens.svg")
plt.close()

print(f"Plots saved to {OUT_DIR}/")
print("Files:", os.listdir(OUT_DIR))
