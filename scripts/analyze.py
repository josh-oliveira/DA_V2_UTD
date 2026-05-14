"""
analyze_metadata.py
-------------------
Step 3 of the Depth-Anything-V2 fine-tuning pipeline.

Loads the metadata.json produced by resize_images.py and (optionally) the
audit_report.json produced by audit_images.py to run a full statistical
analysis of the dataset. When the audit report is provided, per-sample
mean brightness and Laplacian variance are merged in and included across
all sections of the report.

Produces a printed report and a multi-panel visualization saved as
dataset_analysis.png.

Usage:
    # Without audit data:
    python analyze_metadata.py --metadata /path/to/processed/metadata.json

    # With audit data (recommended):
    python analyze_metadata.py \\
        --metadata     /path/to/processed/metadata.json \\
        --audit_report /path/to/raw/audit_report.json

Requirements:
    pip install numpy matplotlib scipy
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats as scipy_stats


# ──────────────────────────────────────────────
# LOAD & PARSE
# ──────────────────────────────────────────────

def load_audit_data(audit_path: Path) -> dict:
    """
    Read audit_report.json and return a dict keyed by original filename,
    containing mean_brightness and laplacian_variance per image.
    """
    with open(audit_path) as f:
        report = json.load(f)

    audit_lookup = {}
    for filename, entry in report["images"].items():
        checks     = entry.get("checks", {})
        brightness = checks.get("exposure", {}).get("mean_brightness", None)
        laplacian  = checks.get("blur",     {}).get("laplacian_variance", None)
        audit_lookup[filename] = {
            "mean_brightness":    brightness,
            "laplacian_variance": laplacian,
        }

    return audit_lookup


def load_metadata(meta_path: Path, audit_path) -> list:
    """
    Load metadata.json and optionally merge per-sample audit values.
    Images whose audit data is missing get None for brightness/laplacian.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    audit_lookup = load_audit_data(audit_path) if audit_path else {}

    records = []
    for name, entry in meta["images"].items():
        if entry["status"] != "ok":
            continue
        c    = entry["crop_info"]
        orig = entry["original_filename"]

        audit_vals = audit_lookup.get(orig, {})

        records.append({
            "name":               entry["sequential_name"],
            "original_name":      orig,
            "orig_w":             c["original_width"],
            "orig_h":             c["original_height"],
            "resized_w":          c["resized_width"],
            "resized_h":          c["resized_height"],
            "crop_x":             c["crop_x"],
            "crop_y":             c["crop_y"],
            "final_size":         c["final_width"],
            "output_kb":          entry["output_size_kb"],
            "quality":            entry["save_quality"],
            # audit-sourced — None if audit report not provided
            "mean_brightness":    audit_vals.get("mean_brightness"),
            "laplacian_variance": audit_vals.get("laplacian_variance"),
        })

    return records


def compute_features(records: list) -> list:
    target = records[0]["final_size"]

    for r in records:
        r["aspect_ratio"]     = r["orig_w"] / r["orig_h"]
        r["orientation"]      = (
            "landscape" if r["aspect_ratio"] > 1.05
            else "portrait" if r["aspect_ratio"] < 0.95
            else "square"
        )
        r["orig_megapixels"]  = (r["orig_w"] * r["orig_h"]) / 1_000_000
        r["scale_factor"]     = target / min(r["orig_w"], r["orig_h"])
        r["was_upscaled"]     = r["scale_factor"] > 1.0
        r["crop_retention"]   = (target * target) / (r["resized_w"] * r["resized_h"]) * 100
        r["content_lost_pct"] = 100 - r["crop_retention"]

        ideal_crop_x       = (r["resized_w"] - target) / 2
        ideal_crop_y       = (r["resized_h"] - target) / 2
        r["crop_offset_x"] = abs(r["crop_x"] - ideal_crop_x)
        r["crop_offset_y"] = abs(r["crop_y"] - ideal_crop_y)

        short_edge = min(r["orig_w"], r["orig_h"])
        if short_edge >= 3000:
            r["res_tier"] = "4K+"
        elif short_edge >= 1440:
            r["res_tier"] = "2K"
        elif short_edge >= 700:
            r["res_tier"] = "HD"
        else:
            r["res_tier"] = "Sub-HD"

    return records


def has_audit_data(records: list) -> bool:
    return any(r["mean_brightness"] is not None for r in records)


# ──────────────────────────────────────────────
# STATS HELPERS
# ──────────────────────────────────────────────

def summary_stats(values, label: str) -> dict:
    arr = np.array(values)
    return {
        "label":  label,
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


def print_stats(s: dict):
    print(f"  Mean:   {s['mean']:.3f}    Std: {s['std']:.3f}")
    print(f"  Median: {s['median']:.3f}  IQR: {s['p25']:.3f}–{s['p75']:.3f}")
    print(f"  Range:  {s['min']:.3f} – {s['max']:.3f}")


# ──────────────────────────────────────────────
# PRINTED REPORT
# ──────────────────────────────────────────────

def print_report(records: list):
    n      = len(records)
    target = records[0]["final_size"]
    audit  = has_audit_data(records)

    def get(key):
        return [r[key] for r in records]

    def get_valid(key):
        vals = [(r[key], r) for r in records if r[key] is not None]
        return [v for v, _ in vals], [r for _, r in vals]

    print("\n" + "=" * 60)
    print(f"DATASET STATISTICAL ANALYSIS  ({n} images, {target}x{target})")
    if audit:
        print("  Brightness + sharpness data: included (from audit report)")
    else:
        print("  Brightness + sharpness data: not available (no --audit_report)")
    print("=" * 60)

    # ── Orientation ──
    print("\n-- ORIENTATION ------------------------------------------")
    for orientation in ["landscape", "portrait", "square"]:
        count = sum(1 for r in records if r["orientation"] == orientation)
        bar   = "#" * count
        print(f"  {orientation:10s}  {count:3d}  {bar}  ({count/n*100:.1f}%)")

    # ── Aspect ratio ──
    print("\n-- ASPECT RATIO (w/h) -----------------------------------")
    print_stats(summary_stats(get("aspect_ratio"), "aspect_ratio"))
    extreme = [r for r in records if r["aspect_ratio"] > 2.0 or r["aspect_ratio"] < 0.5]
    if extreme:
        print(f"  WARNING: {len(extreme)} extreme aspect ratio image(s) (>2:1 or <1:2):")
        for r in extreme:
            print(f"     {r['original_name']}  AR={r['aspect_ratio']:.2f}")

    # ── Resolution tiers ──
    print("\n-- RESOLUTION TIERS -------------------------------------")
    for tier in ["4K+", "2K", "HD", "Sub-HD"]:
        count = sum(1 for r in records if r["res_tier"] == tier)
        bar   = "#" * count
        print(f"  {tier:8s}  {count:3d}  {bar}  ({count/n*100:.1f}%)")
    print(f"\n  Megapixels (original):")
    print_stats(summary_stats(get("orig_megapixels"), "megapixels"))

    # ── Scale factor ──
    print("\n-- SCALE FACTOR (resize step) ---------------------------")
    print("  < 1.0 = downscaled (good)   > 1.0 = upscaled (bad)")
    print_stats(summary_stats(get("scale_factor"), "scale_factor"))
    upscaled = [r for r in records if r["was_upscaled"]]
    if upscaled:
        print(f"\n  WARNING: {len(upscaled)} image(s) were UPSCALED (original < {target}px short edge):")
        for r in upscaled:
            short = min(r["orig_w"], r["orig_h"])
            print(f"     {r['original_name']}  short_edge={short}px  scale={r['scale_factor']:.2f}x")
    else:
        print(f"  OK: All images were downscaled")

    # ── Crop retention ──
    print("\n-- CROP RETENTION ---------------------------------------")
    print("  % of resized image that survived the center crop")
    print_stats(summary_stats(get("crop_retention"), "retention_%"))
    low_retention = [r for r in records if r["crop_retention"] < 60]
    if low_retention:
        print(f"\n  WARNING: {len(low_retention)} image(s) with <60% content retention:")
        for r in low_retention:
            print(f"     {r['original_name']}  AR={r['aspect_ratio']:.2f}  kept={r['crop_retention']:.1f}%")
    else:
        print(f"  OK: All images retain >=60% of content")

    # ── File size ──
    print("\n-- OUTPUT FILE SIZE (KB) --------------------------------")
    print("  Proxy for scene complexity and texture density")
    print_stats(summary_stats(get("output_kb"), "output_kb"))
    low_kb  = [r for r in records if r["output_kb"] < 20]
    high_kb = [r for r in records if r["output_kb"] > 300]
    if low_kb:
        print(f"\n  NOTE: {len(low_kb)} very small file(s) (<20KB) — likely flat/low-texture scenes:")
        for r in low_kb:
            print(f"     {r['original_name']}  {r['output_kb']}KB")
    if high_kb:
        print(f"\n  NOTE: {len(high_kb)} large file(s) (>300KB) — high-texture/complex scenes:")
        for r in high_kb:
            print(f"     {r['original_name']}  {r['output_kb']}KB")

    # ── Mean brightness ──
    if audit:
        print("\n-- MEAN BRIGHTNESS (per sample) -------------------------")
        print("  0-255 scale.  Healthy range: 20-235")
        print("  Low  (<20)  -> underexposed, weak encoder features")
        print("  High (>235) -> overexposed, clipped highlights")
        bvals, brecords = get_valid("mean_brightness")
        if bvals:
            print_stats(summary_stats(bvals, "mean_brightness"))
            dark   = [r for r in brecords if r["mean_brightness"] < 20]
            bright = [r for r in brecords if r["mean_brightness"] > 235]
            if dark:
                print(f"\n  WARNING: {len(dark)} underexposed image(s) (mean < 20):")
                for r in dark:
                    print(f"     {r['original_name']}  brightness={r['mean_brightness']:.1f}")
            if bright:
                print(f"\n  WARNING: {len(bright)} overexposed image(s) (mean > 235):")
                for r in bright:
                    print(f"     {r['original_name']}  brightness={r['mean_brightness']:.1f}")
            if not dark and not bright:
                print(f"  OK: All images within healthy brightness range")
            print(f"\n  Per-sample brightness (sorted low to high):")
            for r in sorted(brecords, key=lambda x: x["mean_brightness"]):
                flag = " WARNING" if r["mean_brightness"] < 20 or r["mean_brightness"] > 235 else ""
                print(f"     {r['name']}  {r['original_name']:35s}  {r['mean_brightness']:6.1f}{flag}")

    # ── Laplacian variance ──
    if audit:
        print("\n-- LAPLACIAN VARIANCE / SHARPNESS (per sample) ----------")
        print("  Higher = sharper edges = better depth label boundaries")
        print("  Blur threshold used during audit: 100.0")
        lvals, lrecords = get_valid("laplacian_variance")
        if lvals:
            print_stats(summary_stats(lvals, "laplacian_variance"))
            soft   = [r for r in lrecords if r["laplacian_variance"] < 100]
            medium = [r for r in lrecords if 100 <= r["laplacian_variance"] < 300]
            sharp  = [r for r in lrecords if r["laplacian_variance"] >= 300]
            print(f"\n  Distribution:")
            print(f"    Sharp   (>=300):   {len(sharp):3d}  ({len(sharp)/len(lrecords)*100:.1f}%)")
            print(f"    Medium (100-299):  {len(medium):3d}  ({len(medium)/len(lrecords)*100:.1f}%)")
            print(f"    Soft    (<100):    {len(soft):3d}  ({len(soft)/len(lrecords)*100:.1f}%)")
            if soft:
                print(f"\n  WARNING: {len(soft)} soft image(s) below blur threshold:")
                for r in soft:
                    print(f"     {r['original_name']}  variance={r['laplacian_variance']:.1f}")
            else:
                print(f"  OK: All images above blur threshold")
            print(f"\n  Per-sample sharpness (sorted low to high):")
            for r in sorted(lrecords, key=lambda x: x["laplacian_variance"]):
                flag = " WARNING" if r["laplacian_variance"] < 100 else ""
                print(f"     {r['name']}  {r['original_name']:35s}  {r['laplacian_variance']:8.1f}{flag}")

    # ── Correlations ──
    print("\n-- CORRELATION ANALYSIS ---------------------------------")
    kb_arr  = np.array(get("output_kb"))
    ret_arr = np.array(get("crop_retention"))
    ar_arr  = np.array(get("aspect_ratio"))
    mp_arr  = np.array(get("orig_megapixels"))
    sf_arr  = np.array(get("scale_factor"))

    pairs = [
        ("File size (KB)",    kb_arr,  "Crop retention (%)", ret_arr),
        ("Aspect ratio",      ar_arr,  "Crop retention (%)", ret_arr),
        ("Megapixels",        mp_arr,  "Scale factor",       sf_arr),
    ]

    if audit:
        bvals, _ = get_valid("mean_brightness")
        lvals, _ = get_valid("laplacian_variance")
        if len(bvals) == n and len(lvals) == n:
            b_arr = np.array(bvals)
            l_arr = np.array(lvals)
            pairs += [
                ("Brightness",         b_arr, "File size (KB)",      kb_arr),
                ("Brightness",         b_arr, "Laplacian variance",   l_arr),
                ("Laplacian variance",  l_arr, "File size (KB)",      kb_arr),
                ("Brightness",         b_arr, "Crop retention (%)",  ret_arr),
            ]

    for x_label, x, y_label, y in pairs:
        if len(x) >= 5:
            r_val, p = scipy_stats.pearsonr(x, y)
            sig = "  * significant" if p < 0.05 else ""
            print(f"  {x_label:25s} <-> {y_label:22s}  r={r_val:+.3f}  p={p:.3f}{sig}")

    # ── Health summary ──
    print("\n-- DATASET HEALTH SUMMARY -------------------------------")
    issues = 0
    if upscaled:
        print(f"  WARNING: {len(upscaled)} upscaled image(s) — consider removing")
        issues += len(upscaled)
    if low_retention:
        print(f"  WARNING: {len(low_retention)} low-retention image(s) — verify crop visually")
        issues += len(low_retention)
    if extreme:
        print(f"  WARNING: {len(extreme)} extreme aspect ratio image(s)")
        issues += len(extreme)
    if audit:
        bvals, brecords = get_valid("mean_brightness")
        lvals, lrecords = get_valid("laplacian_variance")
        bad_exposure = [r for r in brecords if r["mean_brightness"] < 20 or r["mean_brightness"] > 235]
        bad_blur     = [r for r in lrecords if r["laplacian_variance"] < 100]
        if bad_exposure:
            print(f"  WARNING: {len(bad_exposure)} image(s) outside healthy brightness range")
            issues += len(bad_exposure)
        if bad_blur:
            print(f"  WARNING: {len(bad_blur)} image(s) below blur threshold — soft depth labels")
            issues += len(bad_blur)
    if issues == 0:
        print(f"  OK: No major issues detected — dataset looks clean")
    print(f"\n  Total usable images: {n}")
    print()


# ──────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────

COLORS = {
    "primary":   "#4C8BF5",
    "secondary": "#F5A623",
    "accent":    "#2ECC71",
    "warning":   "#E74C3C",
    "neutral":   "#95A5A6",
    "landscape": "#4C8BF5",
    "portrait":  "#F5A623",
    "square":    "#2ECC71",
}


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color="#2a2a5a", linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a5a")


def plot_analysis(records: list, output_path: Path):
    audit  = has_audit_data(records)
    n_rows = 4 if audit else 3
    n      = len(records)

    def get(key):
        return np.array([r[key] for r in records])

    def get_valid(key):
        vals = [r[key] for r in records if r[key] is not None]
        return np.array(vals) if vals else np.array([])

    fig = plt.figure(figsize=(18, 6 * n_rows), facecolor="#1a1a2e")
    fig.suptitle(
        f"Dataset Analysis  -  {n} images  -  "
        f"{records[0]['final_size']}x{records[0]['final_size']} processed"
        + ("  -  brightness & sharpness included" if audit else ""),
        fontsize=15, color="white", fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── Row 0 ──

    # Orientation pie
    ax1 = fig.add_subplot(gs[0, 0])
    ori_counts = {o: sum(1 for r in records if r["orientation"] == o)
                  for o in ["landscape", "portrait", "square"]}
    labels = [k for k, v in ori_counts.items() if v > 0]
    sizes  = [v for k, v in ori_counts.items() if v > 0]
    colors = [COLORS[l] for l in labels]
    _, _, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
    ax1.set_title("Orientation Split", fontsize=10, fontweight="bold", color="white")

    # Aspect ratio histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ar = get("aspect_ratio")
    ax2.hist(ar, bins=max(8, n // 3), color=COLORS["primary"], edgecolor="#1a1a2e", alpha=0.9)
    ax2.axvline(1.0, color=COLORS["warning"], linestyle="--", linewidth=1.5, label="Square (1:1)")
    ax2.axvline(float(np.mean(ar)), color=COLORS["accent"], linestyle="-", linewidth=1.5,
                label=f"Mean {np.mean(ar):.2f}")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax2, "Aspect Ratio Distribution", "Width / Height", "Count")

    # Resolution tiers
    ax3 = fig.add_subplot(gs[0, 2])
    tiers       = ["Sub-HD", "HD", "2K", "4K+"]
    tier_counts = [sum(1 for r in records if r["res_tier"] == t) for t in tiers]
    tier_colors = [COLORS["warning"], COLORS["secondary"], COLORS["primary"], COLORS["accent"]]
    bars = ax3.bar(tiers, tier_counts, color=tier_colors, edgecolor="#1a1a2e", alpha=0.9)
    for bar, cnt in zip(bars, tier_counts):
        if cnt > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(cnt), ha="center", color="white", fontsize=10, fontweight="bold")
    style_ax(ax3, "Resolution Tiers", "Tier", "Count")

    # ── Row 1 ──

    # Scale factor per image
    ax4 = fig.add_subplot(gs[1, 0])
    sf = get("scale_factor")
    sf_colors = [COLORS["warning"] if v > 1 else COLORS["accent"] for v in sf]
    ax4.bar(range(len(sf)), sf, color=sf_colors, edgecolor="#1a1a2e", alpha=0.9, width=0.8)
    ax4.axhline(1.0, color="white", linestyle="--", linewidth=1.5)
    legend_elements = [
        Patch(facecolor=COLORS["accent"],  label="Downscaled"),
        Patch(facecolor=COLORS["warning"], label="Upscaled"),
    ]
    ax4.legend(handles=legend_elements, fontsize=8, labelcolor="white",
               facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax4, "Scale Factor per Image", "Image Index", "Scale Factor")

    # Crop retention per image
    ax5 = fig.add_subplot(gs[1, 1])
    ret = get("crop_retention")
    ret_colors = [COLORS["warning"] if v < 60 else COLORS["primary"] for v in ret]
    ax5.bar(range(len(ret)), ret, color=ret_colors, edgecolor="#1a1a2e", alpha=0.9, width=0.8)
    ax5.axhline(60, color=COLORS["warning"], linestyle="--", linewidth=1.5, label="60% threshold")
    ax5.axhline(float(np.mean(ret)), color=COLORS["accent"], linestyle="-", linewidth=1.5,
                label=f"Mean {np.mean(ret):.1f}%")
    ax5.set_ylim(0, 105)
    ax5.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax5, "Crop Content Retention (%)", "Image Index", "% Retained")

    # File size distribution
    ax6 = fig.add_subplot(gs[1, 2])
    kb = get("output_kb")
    ax6.hist(kb, bins=max(8, n // 3), color=COLORS["secondary"], edgecolor="#1a1a2e", alpha=0.9)
    ax6.axvline(float(np.median(kb)), color=COLORS["accent"], linestyle="-", linewidth=1.5,
                label=f"Median {np.median(kb):.0f}KB")
    ax6.axvline(float(np.mean(kb)), color=COLORS["primary"], linestyle="--", linewidth=1.5,
                label=f"Mean {np.mean(kb):.0f}KB")
    ax6.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax6, "Output File Size (KB)\n(proxy for scene complexity)", "KB", "Count")

    # ── Row 2 ──

    # Aspect ratio vs crop retention scatter
    ax7 = fig.add_subplot(gs[2, 0])
    ar_vals  = get("aspect_ratio")
    ret_vals = get("crop_retention")
    sc_colors = [COLORS["landscape"] if r["orientation"] == "landscape"
                 else COLORS["portrait"] if r["orientation"] == "portrait"
                 else COLORS["square"] for r in records]
    ax7.scatter(ar_vals, ret_vals, c=sc_colors, alpha=0.85, s=60,
                edgecolors="#1a1a2e", linewidth=0.5)
    ax7.axhline(60, color=COLORS["warning"], linestyle="--", linewidth=1.2, alpha=0.7)
    if len(ar_vals) >= 5:
        m, b_c = np.polyfit(ar_vals, ret_vals, 1)
        x_line = np.linspace(ar_vals.min(), ar_vals.max(), 100)
        ax7.plot(x_line, m * x_line + b_c, color="white", linewidth=1.2, alpha=0.5, linestyle=":")
    legend_elements = [
        Patch(facecolor=COLORS["landscape"], label="Landscape"),
        Patch(facecolor=COLORS["portrait"],  label="Portrait"),
        Patch(facecolor=COLORS["square"],    label="Square"),
    ]
    ax7.legend(handles=legend_elements, fontsize=8, labelcolor="white",
               facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax7, "Aspect Ratio vs Crop Retention", "Aspect Ratio (w/h)", "Content Retained (%)")

    # Megapixels vs file size scatter
    ax8 = fig.add_subplot(gs[2, 1])
    mp_vals = get("orig_megapixels")
    kb_vals = get("output_kb")
    ax8.scatter(mp_vals, kb_vals, c=COLORS["primary"], alpha=0.85, s=60,
                edgecolors="#1a1a2e", linewidth=0.5)
    if len(mp_vals) >= 5:
        m, b_c = np.polyfit(mp_vals, kb_vals, 1)
        x_line = np.linspace(mp_vals.min(), mp_vals.max(), 100)
        ax8.plot(x_line, m * x_line + b_c, color=COLORS["accent"], linewidth=1.5,
                 linestyle="--", alpha=0.8, label="Trend")
        ax8.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#2a2a5a")
    style_ax(ax8, "Original Megapixels vs Output File Size\n(complexity by resolution)",
             "Megapixels", "Output KB")

    # Per-image heatmap — 4 rows base, 6 rows with audit data
    ax9 = fig.add_subplot(gs[2, 2])
    base_metrics = {
        "Scale factor":   get("scale_factor"),
        "Crop retention": get("crop_retention") / 100,
        "File size norm": get("output_kb") / (get("output_kb").max() + 1e-6),
        "Aspect ratio":   get("aspect_ratio") / (get("aspect_ratio").max() + 1e-6),
    }
    if audit:
        bv = get_valid("mean_brightness")
        lv = get_valid("laplacian_variance")
        if len(bv) == n and len(lv) == n:
            base_metrics["Brightness norm"] = bv / 255.0
            base_metrics["Sharpness norm"]  = np.clip(lv / lv.max(), 0, 1)

    matrix = np.array(list(base_metrics.values()))
    im = ax9.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                    interpolation="nearest")
    ax9.set_yticks(range(len(base_metrics)))
    ax9.set_yticklabels(list(base_metrics.keys()), fontsize=8)
    ax9.set_xlabel("Image Index", fontsize=9)
    ax9.tick_params(colors="white", labelsize=8)
    ax9.xaxis.label.set_color("white")
    row_label = f"{len(base_metrics)}-row" if audit else "4-row"
    ax9.set_title(f"Per-Image Metrics Heatmap\n({row_label}, normalised, green=good)",
                  fontsize=10, fontweight="bold", color="white")
    for spine in ax9.spines.values():
        spine.set_edgecolor("#2a2a5a")
    cbar = fig.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white", labelsize=8)

    # ── Row 3 — only rendered when audit data is present ──

    if audit:
        bv = get_valid("mean_brightness")
        lv = get_valid("laplacian_variance")

        # Mean brightness per image
        ax10 = fig.add_subplot(gs[3, 0])
        if len(bv) == n:
            b_colors = []
            for v in bv:
                if v < 20 or v > 235:
                    b_colors.append(COLORS["warning"])
                elif v < 60 or v > 200:
                    b_colors.append(COLORS["secondary"])
                else:
                    b_colors.append(COLORS["accent"])
            ax10.bar(range(n), bv, color=b_colors, edgecolor="#1a1a2e", alpha=0.9, width=0.8)
            ax10.axhline(20,  color=COLORS["warning"], linestyle="--", linewidth=1.2)
            ax10.axhline(235, color=COLORS["warning"], linestyle="--", linewidth=1.2)
            ax10.axhline(float(np.mean(bv)), color=COLORS["accent"], linestyle="-",
                         linewidth=1.5, label=f"Mean {np.mean(bv):.1f}")
            ax10.set_ylim(0, 270)
            legend_elements = [
                Patch(facecolor=COLORS["accent"],    label="Healthy (60-200)"),
                Patch(facecolor=COLORS["secondary"], label="Marginal (20-60 or 200-235)"),
                Patch(facecolor=COLORS["warning"],   label="Out of range"),
            ]
            ax10.legend(handles=legend_elements, fontsize=7, labelcolor="white",
                        facecolor="#1a1a2e", edgecolor="#2a2a5a")
        style_ax(ax10, "Mean Brightness per Sample\n(0-255, healthy: 20-235)",
                 "Image Index", "Mean Brightness")

        # Laplacian variance per image
        ax11 = fig.add_subplot(gs[3, 1])
        if len(lv) == n:
            l_colors = []
            for v in lv:
                if v < 100:
                    l_colors.append(COLORS["warning"])
                elif v < 300:
                    l_colors.append(COLORS["secondary"])
                else:
                    l_colors.append(COLORS["accent"])
            ax11.bar(range(n), lv, color=l_colors, edgecolor="#1a1a2e", alpha=0.9, width=0.8)
            ax11.axhline(100, color=COLORS["warning"], linestyle="--", linewidth=1.5,
                         label="Blur threshold (100)")
            ax11.axhline(float(np.mean(lv)), color=COLORS["accent"], linestyle="-",
                         linewidth=1.5, label=f"Mean {np.mean(lv):.1f}")
            ax11.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#2a2a5a")
            legend_elements = [
                Patch(facecolor=COLORS["accent"],    label="Sharp (>=300)"),
                Patch(facecolor=COLORS["secondary"], label="Medium (100-299)"),
                Patch(facecolor=COLORS["warning"],   label="Soft (<100)"),
            ]
            ax11.legend(handles=legend_elements, fontsize=7, labelcolor="white",
                        facecolor="#1a1a2e", edgecolor="#2a2a5a")
        style_ax(ax11, "Laplacian Variance per Sample\n(sharpness - higher is better)",
                 "Image Index", "Laplacian Variance")

        # Brightness vs Laplacian scatter
        ax12 = fig.add_subplot(gs[3, 2])
        if len(bv) == n and len(lv) == n:
            point_colors = []
            for b_val, l_val in zip(bv, lv):
                if (b_val < 20 or b_val > 235) and l_val < 100:
                    point_colors.append(COLORS["warning"])
                elif b_val < 20 or b_val > 235 or l_val < 100:
                    point_colors.append(COLORS["secondary"])
                else:
                    point_colors.append(COLORS["accent"])
            ax12.scatter(bv, lv, c=point_colors, alpha=0.85, s=70,
                         edgecolors="#1a1a2e", linewidth=0.5)
            ax12.axvline(20,  color=COLORS["warning"], linestyle="--", linewidth=1.0, alpha=0.6)
            ax12.axvline(235, color=COLORS["warning"], linestyle="--", linewidth=1.0, alpha=0.6)
            ax12.axhline(100, color=COLORS["warning"], linestyle="--", linewidth=1.0, alpha=0.6)
            if len(bv) >= 5:
                r_val, p_val = scipy_stats.pearsonr(bv, lv)
                ax12.text(0.05, 0.92, f"r = {r_val:+.3f}  p = {p_val:.3f}",
                          transform=ax12.transAxes, color="white", fontsize=8,
                          bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.7))
            legend_elements = [
                Patch(facecolor=COLORS["accent"],    label="Both healthy"),
                Patch(facecolor=COLORS["secondary"], label="One issue"),
                Patch(facecolor=COLORS["warning"],   label="Both issues"),
            ]
            ax12.legend(handles=legend_elements, fontsize=7, labelcolor="white",
                        facecolor="#1a1a2e", edgecolor="#2a2a5a")
        style_ax(ax12, "Brightness vs Sharpness\n(healthy images cluster top-centre)",
                 "Mean Brightness (0-255)", "Laplacian Variance")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Visualization saved -> {output_path}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of dataset metadata.")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.json produced by resize_images.py",
    )
    parser.add_argument(
        "--audit_report",
        type=str,
        default=None,
        help="Path to audit_report.json produced by audit_images.py. "
             "When provided, per-sample brightness and Laplacian variance are merged in.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the plot PNG. Default: same folder as metadata.json",
    )
    args = parser.parse_args()

    meta_path  = Path(args.metadata).resolve()
    audit_path = Path(args.audit_report).resolve() if args.audit_report else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else meta_path.parent

    if not meta_path.exists():
        print(f"Error: metadata.json not found at {meta_path}")
        sys.exit(1)
    if audit_path and not audit_path.exists():
        print(f"Error: audit_report.json not found at {audit_path}")
        sys.exit(1)

    records = load_metadata(meta_path, audit_path)
    if not records:
        print("No successfully processed images found in metadata.")
        sys.exit(1)

    records = compute_features(records)
    print_report(records)

    plot_path = output_dir / "dataset_analysis.png"
    print("Generating visualization...")
    plot_analysis(records, plot_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()