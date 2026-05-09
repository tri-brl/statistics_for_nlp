"""
Stage 6d — Section 8: Confidence Intervals and ROPE
=====================================================
8.1 CIs on acoustic contrasts (forest plot)
8.2 CIs on neural contrasts (forest plot)
8.3 ROPE definition and classification
8.4 Summary table

Inputs:
  C:\\Users\\aviba\\data\\features_acoustic_norm.csv
  C:\\Users\\aviba\\data\\features_*_pca.npz

Outputs:
  results/6d_acoustic_CIs.csv
  results/6d_neural_CIs.csv
  results/6d_ROPE_classification.csv
  results/6d_forest_acoustic.png
  results/6d_forest_neural.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\aviba\data")
RESULTS_DIR = Path(r"C:\Users\aviba\results")
RESULTS_DIR.mkdir(exist_ok=True)

ACOUSTIC_NORM = DATA_DIR / "features_acoustic_norm.csv"
NEURAL_FILES  = {
    "whisper_L2":  DATA_DIR / "features_whisper_L2.npz",
    "whisper_L5":  DATA_DIR / "features_whisper_L5.npz",
    "xlsr_L3":     DATA_DIR / "features_xlsr_L3.npz",
    "xlsr_L12":    DATA_DIR / "features_xlsr_L12.npz",
    "xlsr_L21":    DATA_DIR / "features_xlsr_L21.npz",
}

VOWELS     = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ", "ə"}
B          = 2000   # bootstrap iterations
ALPHA      = 0.05
RANDOM_STATE = 42
rng        = np.random.default_rng(RANDOM_STATE)

# ── ROPE definitions (as per assignment) ──────────────────────────────────────
# Acoustic: ±20 Hz on raw F1/F2 (JND ~3-5% of formant value)
# But we work in Lobanov units so convert:
# typical F1 ~500Hz, std ~100Hz -> 20Hz ~ 0.2 Lobanov units
ROPE_ACOUSTIC = (-0.2, 0.2)

# Neural: [0, delta0] where delta0 = mean intra-speaker cosine distance
# We'll compute this from the data and set it per model
# As a starting point use 0.05 (will be refined per model)
ROPE_NEURAL_DEFAULT = (0.0, 0.05)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df       = pd.read_csv(ACOUSTIC_NORM)
vowel_df = df[df["phoneme"].isin(VOWELS)].copy()

key_meta = {}
for _, row in df.iterrows():
    key = (f"{row['speaker']}__{row['phoneme']}__{row['sentence_id']}"
           f"__{row['onset']:.4f}")
    key_meta[key] = {
        "phoneme": row["phoneme"],
        "L1":      row["L1"],
        "gender":  row["gender"],
        "speaker": row["speaker"],
    }

speakers = sorted(df["speaker"].unique())
print(f"  Tokens: {len(df)}, Vowels: {len(vowel_df)}, Speakers: {len(speakers)}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.1 BOOTSTRAP CIs ON ACOUSTIC CONTRASTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8.1 Bootstrap CIs — acoustic (speaker-level resampling)")
print("="*60)

ac_ci_rows = []

for phoneme in sorted(vowel_df["phoneme"].unique()):
    for formant in ["F1_lob", "F2_lob"]:
        # per-speaker means
        spk_means = (vowel_df[vowel_df["phoneme"] == phoneme]
                     .groupby(["speaker", "L1"])[formant]
                     .mean().reset_index())

        l1_spks = spk_means[spk_means["L1"] == "fr"]["speaker"].tolist()
        l2_spks = spk_means[spk_means["L1"] == "ru"]["speaker"].tolist()

        if len(l1_spks) < 2 or len(l2_spks) < 2:
            continue

        l1_means = spk_means[spk_means["L1"] == "fr"].set_index("speaker")[formant]
        l2_means = spk_means[spk_means["L1"] == "ru"].set_index("speaker")[formant]

        obs_diff = l1_means.mean() - l2_means.mean()

        # bootstrap at speaker level
        boot_diffs = []
        for _ in range(B):
            s_l1 = rng.choice(l1_spks, size=len(l1_spks), replace=True)
            s_l2 = rng.choice(l2_spks, size=len(l2_spks), replace=True)
            d    = l1_means[s_l1].mean() - l2_means[s_l2].mean()
            boot_diffs.append(d)

        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

        ac_ci_rows.append({
            "phoneme":  phoneme,
            "formant":  formant,
            "obs_diff": round(obs_diff, 4),
            "ci_lo":    round(ci_lo, 4),
            "ci_hi":    round(ci_hi, 4),
            "sig":      not (ci_lo <= 0 <= ci_hi),
        })
        print(f"  /{phoneme}/ {formant}: {obs_diff:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

ac_ci_df = pd.DataFrame(ac_ci_rows)
ac_ci_df.to_csv(RESULTS_DIR / "6d_acoustic_CIs.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 8.2 BOOTSTRAP CIs ON NEURAL CONTRASTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8.2 Bootstrap CIs — neural (speaker-level resampling)")
print("="*60)

neu_ci_rows  = []
intra_spk_distances = {}  # for ROPE computation

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    print(f"\n  {name} ...")
    data = np.load(npz_path)
    keys = list(data.files)

    # precompute per-speaker per-phoneme mean vectors
    spk_ph_vecs = {}
    for k in keys:
        m = key_meta.get(k)
        if m is None or m["phoneme"] not in VOWELS:
            continue
        key2 = (m["speaker"], m["phoneme"])
        if key2 not in spk_ph_vecs:
            spk_ph_vecs[key2] = []
        spk_ph_vecs[key2].append(data[k].astype(np.float64))
    for key2 in spk_ph_vecs:
        spk_ph_vecs[key2] = np.mean(spk_ph_vecs[key2], axis=0)

    # compute intra-speaker distance for ROPE
    # = mean cosine distance between same speaker different phoneme pairs
    intra_dists = []
    for spk in speakers:
        spk_phs = [p for (s, p) in spk_ph_vecs if s == spk]
        for i in range(len(spk_phs)):
            for j in range(i + 1, len(spk_phs)):
                v1 = spk_ph_vecs[(spk, spk_phs[i])]
                v2 = spk_ph_vecs[(spk, spk_phs[j])]
                d  = 1 - np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                intra_dists.append(d)

    rope_delta = np.mean(intra_dists) if intra_dists else 0.05
    intra_spk_distances[name] = round(rope_delta, 4)
    print(f"    ROPE delta (mean intra-speaker dist): {rope_delta:.4f}")

    # bootstrap CIs per phoneme
    l1_speakers = sorted({key_meta[k]["speaker"] for k in keys
                          if key_meta.get(k, {}).get("L1") == "fr"})
    l2_speakers = sorted({key_meta[k]["speaker"] for k in keys
                          if key_meta.get(k, {}).get("L1") == "ru"})

    for phoneme in sorted(VOWELS):
        valid_l1 = [s for s in l1_speakers if (s, phoneme) in spk_ph_vecs]
        valid_l2 = [s for s in l2_speakers if (s, phoneme) in spk_ph_vecs]

        if len(valid_l1) < 2 or len(valid_l2) < 2:
            continue

        # observed centroid distance
        c1_obs = np.mean([spk_ph_vecs[(s, phoneme)] for s in valid_l1], axis=0)
        c2_obs = np.mean([spk_ph_vecs[(s, phoneme)] for s in valid_l2], axis=0)
        obs_d  = 1 - np.dot(c1_obs, c2_obs) / (
            np.linalg.norm(c1_obs) * np.linalg.norm(c2_obs) + 1e-8)

        # bootstrap
        boot_dists = []
        for _ in range(B):
            s1 = rng.choice(valid_l1, size=len(valid_l1), replace=True)
            s2 = rng.choice(valid_l2, size=len(valid_l2), replace=True)
            c1 = np.mean([spk_ph_vecs[(s, phoneme)] for s in s1], axis=0)
            c2 = np.mean([spk_ph_vecs[(s, phoneme)] for s in s2], axis=0)
            d  = 1 - np.dot(c1, c2) / (
                np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
            boot_dists.append(d)

        ci_lo, ci_hi = np.percentile(boot_dists, [2.5, 97.5])

        neu_ci_rows.append({
            "model":      name,
            "phoneme":    phoneme,
            "obs_dist":   round(obs_d, 4),
            "ci_lo":      round(ci_lo, 4),
            "ci_hi":      round(ci_hi, 4),
            "rope_delta": round(rope_delta, 4),
        })

    print(f"    Done {len([r for r in neu_ci_rows if r['model'] == name])} phonemes")

neu_ci_df = pd.DataFrame(neu_ci_rows)
neu_ci_df.to_csv(RESULTS_DIR / "6d_neural_CIs.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 8.3 + 8.4 ROPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8.3/8.4 ROPE classification")
print("="*60)

rope_rows = []

# acoustic
for _, row in ac_ci_df.iterrows():
    lo, hi     = ROPE_ACOUSTIC
    ci_lo, ci_hi = row["ci_lo"], row["ci_hi"]

    if ci_hi < lo or ci_lo > hi:
        classification = "non-equivalent"
    elif ci_lo >= lo and ci_hi <= hi:
        classification = "equivalent"
    else:
        classification = "indeterminate"

    rope_rows.append({
        "representation": f"acoustic_{row['formant']}",
        "phoneme":        row["phoneme"],
        "estimate":       row["obs_diff"],
        "ci_lo":          ci_lo,
        "ci_hi":          ci_hi,
        "rope_lo":        lo,
        "rope_hi":        hi,
        "classification": classification,
    })

# neural
for _, row in neu_ci_df.iterrows():
    rope_lo = 0.0
    rope_hi = intra_spk_distances.get(row["model"], 0.05)
    ci_lo, ci_hi = row["ci_lo"], row["ci_hi"]

    if ci_lo > rope_hi:
        classification = "non-equivalent"
    elif ci_hi <= rope_hi:
        classification = "equivalent"
    else:
        classification = "indeterminate"

    rope_rows.append({
        "representation": row["model"],
        "phoneme":        row["phoneme"],
        "estimate":       row["obs_dist"],
        "ci_lo":          ci_lo,
        "ci_hi":          ci_hi,
        "rope_lo":        rope_lo,
        "rope_hi":        rope_hi,
        "classification": classification,
    })

rope_df = pd.DataFrame(rope_rows)
rope_df.to_csv(RESULTS_DIR / "6d_ROPE_classification.csv", index=False)

print("\nROPE classification summary:")
print(rope_df.groupby(["representation", "classification"])
      .size().unstack(fill_value=0).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# FOREST PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Generating forest plots ...")

# ── acoustic forest plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

for ax, formant in zip(axes, ["F1_lob", "F2_lob"]):
    fdf = ac_ci_df[ac_ci_df["formant"] == formant].copy()
    fdf = fdf.sort_values("obs_diff")
    y   = range(len(fdf))

    colors = ["#e74c3c" if r["sig"] else "#95a5a6"
              for _, r in fdf.iterrows()]

    ax.barh(list(y), fdf["ci_hi"] - fdf["ci_lo"],
            left=fdf["ci_lo"], height=0.5,
            color=colors, alpha=0.5)
    ax.scatter(fdf["obs_diff"], list(y), color=colors, zorder=5, s=60)

    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvspan(*ROPE_ACOUSTIC, alpha=0.08, color="gray", label="ROPE")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"/{p}/" for p in fdf["phoneme"]], fontsize=11)
    ax.set_xlabel("L1 - L2 difference (Lobanov units)")
    ax.set_title(f"{formant}: L1 vs L2 contrast\nwith 95% bootstrap CI")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Forest plot — Acoustic L1/L2 contrasts (F1 and F2)", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "6d_forest_acoustic.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> results/6d_forest_acoustic.png")


# ── neural forest plot ────────────────────────────────────────────────────────
model_names = list(NEURAL_FILES.keys())
fig, axes   = plt.subplots(1, len(model_names), figsize=(20, 8), sharey=False)

for ax, name in zip(axes, model_names):
    ndf = neu_ci_df[neu_ci_df["model"] == name].copy()
    if len(ndf) == 0:
        ax.set_visible(False)
        continue

    ndf   = ndf.sort_values("obs_dist")
    y     = range(len(ndf))
    rope_hi = intra_spk_distances.get(name, 0.05)

    rope_rows_model = rope_df[rope_df["representation"] == name]
    color_map = {"non-equivalent": "#e74c3c",
                 "equivalent":     "#2ecc71",
                 "indeterminate":  "#f39c12"}

    for i, (_, row) in enumerate(ndf.iterrows()):
        cl = rope_df[(rope_df["representation"] == name) &
                     (rope_df["phoneme"] == row["phoneme"])]["classification"]
        c  = color_map.get(cl.values[0] if len(cl) > 0 else "indeterminate",
                           "#95a5a6")
        ax.barh(i, row["ci_hi"] - row["ci_lo"],
                left=row["ci_lo"], height=0.5, color=c, alpha=0.5)
        ax.scatter(row["obs_dist"], i, color=c, zorder=5, s=60)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvspan(0, rope_hi, alpha=0.08, color="gray", label=f"ROPE [0,{rope_hi:.3f}]")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"/{p}/" for p in ndf["phoneme"]], fontsize=9)
    ax.set_xlabel("cosine distance")
    ax.set_title(f"{name}\nROPE δ={rope_hi:.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)

# legend
patches = [mpatches.Patch(color=c, label=l)
           for l, c in [("non-equivalent", "#e74c3c"),
                        ("equivalent",     "#2ecc71"),
                        ("indeterminate",  "#f39c12")]]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Forest plot — Neural L1/L2 contrasts by model", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "6d_forest_neural.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> results/6d_forest_neural.png")


# ── done ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Section 6d complete. Files saved:")
print("="*60)
for p in sorted(RESULTS_DIR.glob("6d_*")):
    print(f"  {p.name}")