import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\aviba\data")
RESULTS_DIR = Path(r"C:\Users\aviba\results")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "pca_plots").mkdir(exist_ok=True)
(RESULTS_DIR / "umap_plots").mkdir(exist_ok=True)

ACOUSTIC_NORM = DATA_DIR / "features_acoustic_norm.csv"

NEURAL_FILES = {
    "whisper_L2":  DATA_DIR / "features_whisper_L2.npz",
    "whisper_L5":  DATA_DIR / "features_whisper_L5.npz",
    "xlsr_L3":     DATA_DIR / "features_xlsr_L3.npz",
    "xlsr_L12":    DATA_DIR / "features_xlsr_L12.npz",
    "xlsr_L21":    DATA_DIR / "features_xlsr_L21.npz",
}

PCA_FILES = {k: DATA_DIR / f"features_{k}_pca.npz" for k in NEURAL_FILES}
UMAP_FILES = {k: DATA_DIR / f"features_{k}_umap.npz" for k in NEURAL_FILES}

# French oral vowels only for vowel chart
VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ", "ə"}

# colour palette for phonemes
PHONEME_COLORS = {
    "a": "#e6194b", "ɑ": "#f58231", "e": "#ffe119", "ɛ": "#bfef45",
    "i": "#3cb44b", "o": "#42d4f4", "ɔ": "#4363d8", "u": "#911eb4",
    "y": "#f032e6", "ø": "#a9a9a9", "œ": "#9a6324", "ə": "#469990",
}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(ACOUSTIC_NORM)
vowel_df = df[df["phoneme"].isin(VOWELS)].copy()
print(f"  Total tokens : {len(df)}")
print(f"  Vowel tokens : {len(vowel_df)}")
print(f"  Speakers     : {df['speaker'].nunique()}")


# ── helper: confidence ellipse ────────────────────────────────────────────────
def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


# ── 5.1 vowel chart ───────────────────────────────────────────────────────────
print("\n[5.1] Vowel chart ...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
groups = {"L1 (French)": "fr", "L2 (Russian)": "ru"}

for ax, (group_label, l1_val) in zip(axes, groups.items()):
    gdf = vowel_df[vowel_df["L1"] == l1_val]

    for phoneme in sorted(gdf["phoneme"].unique()):
        pdf = gdf[gdf["phoneme"] == phoneme]
        if len(pdf) < 3:
            continue
        color = PHONEME_COLORS.get(phoneme, "#333333")

        # centroid
        cx = pdf["F2_lob"].mean()
        cy = pdf["F1_lob"].mean()
        ax.scatter(cx, cy, color=color, s=120, zorder=5)
        ax.annotate(f"/{phoneme}/", (cx, cy),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=11, color=color, fontweight="bold")

        # 95% confidence ellipse
        confidence_ellipse(
            pdf["F2_lob"].values, pdf["F1_lob"].values,
            ax, n_std=2.0,
            facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5
        )

    ax.set_xlabel("F2 (Lobanov)", fontsize=12)
    ax.set_ylabel("F1 (Lobanov)", fontsize=12)
    ax.set_title(f"Vowel space — {group_label}", fontsize=13)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

plt.suptitle("Vowel chart: F1 vs F2 (Lobanov normalised) with 95% confidence ellipses",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vowel_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> results/vowel_chart.png")


# ── 5.1 boxplots F1/F2 per phoneme by L1 and gender ──────────────────────────
print("\n[5.1] Box plots ...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
formants = ["F1_lob", "F2_lob"]
factors  = [("L1", "L1 status"), ("gender", "Gender")]

for col, (factor, factor_label) in enumerate(factors):
    for row, formant in enumerate(formants):
        ax = axes[row, col]
        plot_df = vowel_df[["phoneme", formant, factor]].dropna()
        sns.boxplot(
            data=plot_df,
            x="phoneme", y=formant, hue=factor,
            ax=ax, palette="Set2",
            order=sorted(plot_df["phoneme"].unique()),
        )
        ax.set_title(f"{formant} by phoneme — stratified by {factor_label}")
        ax.set_xlabel("phoneme")
        ax.set_ylabel(formant)
        ax.tick_params(axis="x", labelsize=9)

plt.suptitle("F1/F2 (Lobanov) per phoneme by L1 status and gender", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "boxplots_F1_F2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> results/boxplots_F1_F2.png")


# ── 5.1 variance decomposition ────────────────────────────────────────────────
print("\n[5.1] Variance decomposition ...")

var_rows = []
for phoneme in sorted(vowel_df["phoneme"].unique()):
    pdf = vowel_df[vowel_df["phoneme"] == phoneme]["F1_lob"].dropna()
    if len(pdf) < 5:
        continue
    total_var = pdf.var()

    # inter-speaker: variance of per-speaker means
    spk_means = vowel_df[vowel_df["phoneme"] == phoneme].groupby("speaker")["F1_lob"].mean()
    inter_var = spk_means.var() if len(spk_means) > 1 else 0

    # intra-speaker: mean of per-speaker variances
    spk_vars  = vowel_df[vowel_df["phoneme"] == phoneme].groupby("speaker")["F1_lob"].var()
    intra_var = spk_vars.mean()

    residual  = max(0, total_var - inter_var - intra_var)

    var_rows.append({
        "phoneme":    phoneme,
        "total_var":  round(total_var, 4),
        "inter_spk":  round(inter_var, 4),
        "intra_spk":  round(intra_var, 4),
        "residual":   round(residual, 4),
        "inter_pct":  round(inter_var / total_var * 100, 1) if total_var > 0 else 0,
        "intra_pct":  round(intra_var / total_var * 100, 1) if total_var > 0 else 0,
    })

var_df = pd.DataFrame(var_rows)
var_df.to_csv(RESULTS_DIR / "variance_decomposition.csv", index=False)
print(var_df[["phoneme", "inter_pct", "intra_pct"]].to_string(index=False))


# ── 5.2 PCA plots ─────────────────────────────────────────────────────────────
print("\n[5.2] PCA 2D plots ...")

# build key -> metadata lookup
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

def plot_2d(coords, labels, title, out_path, palette=None, hue_order=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1], "label": labels
    })
    sns.scatterplot(
        data=plot_df, x="x", y="y", hue="label",
        palette=palette, hue_order=hue_order,
        alpha=0.4, s=15, ax=ax, linewidth=0,
    )
    ax.set_title(title, fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, markerscale=2)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

for name, pca_path in PCA_FILES.items():
    if not pca_path.exists():
        continue
    pca_data = np.load(pca_path, allow_pickle=True)
    keys     = list(pca_data["keys"])
    coords   = pca_data["coords_2d"]

    phonemes = [key_meta.get(k, {}).get("phoneme", "?") for k in keys]
    l1s      = [key_meta.get(k, {}).get("L1", "?")      for k in keys]
    genders  = [key_meta.get(k, {}).get("gender", "?")  for k in keys]

    # filter to vowels only for phoneme plot
    vowel_mask = np.array([p in VOWELS for p in phonemes])

    plot_2d(coords[vowel_mask],
            [phonemes[i] for i in range(len(phonemes)) if vowel_mask[i]],
            f"PCA 2D — {name} — coloured by phoneme (vowels)",
            RESULTS_DIR / "pca_plots" / f"{name}_phoneme.png")

    plot_2d(coords, l1s,
            f"PCA 2D — {name} — coloured by L1",
            RESULTS_DIR / "pca_plots" / f"{name}_L1.png",
            palette={"fr": "#4C72B0", "ru": "#DD8452"})

    plot_2d(coords, genders,
            f"PCA 2D — {name} — coloured by gender",
            RESULTS_DIR / "pca_plots" / f"{name}_gender.png",
            palette={"f": "#e377c2", "m": "#1f77b4"})

    print(f"  Saved PCA plots for {name}")


# ── 5.2 UMAP plots ────────────────────────────────────────────────────────────
print("\n[5.2] UMAP 2D plots ...")

for name, umap_path in UMAP_FILES.items():
    if not umap_path.exists():
        continue
    umap_data = np.load(umap_path, allow_pickle=True)
    keys      = list(umap_data["keys"])
    coords    = umap_data["coords_umap"]

    phonemes = [key_meta.get(k, {}).get("phoneme", "?") for k in keys]
    l1s      = [key_meta.get(k, {}).get("L1", "?")      for k in keys]
    genders  = [key_meta.get(k, {}).get("gender", "?")  for k in keys]

    vowel_mask = np.array([p in VOWELS for p in phonemes])

    plot_2d(coords[vowel_mask],
            [phonemes[i] for i in range(len(phonemes)) if vowel_mask[i]],
            f"UMAP 2D — {name} — coloured by phoneme (vowels)",
            RESULTS_DIR / "umap_plots" / f"{name}_phoneme.png")

    plot_2d(coords, l1s,
            f"UMAP 2D — {name} — coloured by L1",
            RESULTS_DIR / "umap_plots" / f"{name}_L1.png",
            palette={"fr": "#4C72B0", "ru": "#DD8452"})

    plot_2d(coords, genders,
            f"UMAP 2D — {name} — coloured by gender",
            RESULTS_DIR / "umap_plots" / f"{name}_gender.png",
            palette={"f": "#e377c2", "m": "#1f77b4"})

    print(f"  Saved UMAP plots for {name}")


# ── 5.2 between-class variance ratio ─────────────────────────────────────────
print("\n[5.2] Between-class variance ratio ...")

var_ratio_rows = []

for name, pca_path in PCA_FILES.items():
    if not pca_path.exists():
        continue
    pca_data  = np.load(pca_path, allow_pickle=True)
    keys      = list(pca_data["keys"])
    coords    = pca_data["coords_2d"]
    phonemes  = np.array([key_meta.get(k, {}).get("phoneme", "?") for k in keys])

    # filter vowels only
    mask     = np.array([p in VOWELS for p in phonemes])
    coords_v = coords[mask]
    phones_v = phonemes[mask]

    if len(coords_v) < 10:
        continue

    total_var   = coords_v.var(axis=0).sum()
    grand_mean  = coords_v.mean(axis=0)
    between_var = 0
    for ph in np.unique(phones_v):
        ph_coords = coords_v[phones_v == ph]
        between_var += len(ph_coords) * ((ph_coords.mean(axis=0) - grand_mean) ** 2).sum()
    between_var /= len(coords_v)

    ratio = between_var / total_var if total_var > 0 else 0
    var_ratio_rows.append({"model": name, "between_class_variance_ratio": round(ratio, 4)})
    print(f"  {name:15s} between-class variance ratio: {ratio:.4f}")

pd.DataFrame(var_ratio_rows).to_csv(RESULTS_DIR / "variance_ratios.csv", index=False)


# ── 5.2 cosine similarity within vs between phonemes ─────────────────────────
print("\n[5.2] Cosine similarity within/between phonemes ...")

cos_rows = []

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data     = np.load(npz_path)
    keys     = list(data.files)
    phonemes = np.array([key_meta.get(k, {}).get("phoneme", "?") for k in keys])

    # vowels only, subsample for speed
    mask = np.array([p in VOWELS for p in phonemes])
    keys_v   = [k for k, m in zip(keys, mask) if m]
    phones_v = phonemes[mask]

    # subsample max 2000 vowel tokens
    if len(keys_v) > 2000:
        idx      = np.random.choice(len(keys_v), 2000, replace=False)
        keys_v   = [keys_v[i] for i in idx]
        phones_v = phones_v[idx]

    matrix = np.stack([data[k] for k in keys_v]).astype(np.float32)

    # cosine similarity matrix
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / (norms + 1e-8)
    sim    = normed @ normed.T  # (N, N)

    within_sims  = []
    between_sims = []

    for i in range(len(keys_v)):
        for j in range(i + 1, len(keys_v)):
            s = sim[i, j]
            if phones_v[i] == phones_v[j]:
                within_sims.append(s)
            else:
                between_sims.append(s)

    mean_within  = np.mean(within_sims)
    mean_between = np.mean(between_sims)
    ratio        = mean_within / mean_between if mean_between > 0 else np.nan

    cos_rows.append({
        "model":         name,
        "within_phoneme_sim":  round(float(mean_within), 4),
        "between_phoneme_sim": round(float(mean_between), 4),
        "ratio":               round(float(ratio), 4),
    })
    print(f"  {name:15s} within={mean_within:.4f}  between={mean_between:.4f}  "
          f"ratio={ratio:.4f}")

pd.DataFrame(cos_rows).to_csv(RESULTS_DIR / "cosine_similarity.csv", index=False)


# ── 5.3 Mantel test ───────────────────────────────────────────────────────────
print("\n[5.3] Mantel test (acoustic RSM vs neural RSMs) ...")

def mantel_test(rsm1, rsm2, n_permutations=999, random_state=42):
    """
    Mantel test: rank correlation between upper triangles of two RSMs.
    Returns r, p_value.
    """
    rng  = np.random.default_rng(random_state)
    n    = rsm1.shape[0]
    idx  = np.triu_indices(n, k=1)
    v1   = rsm1[idx]
    v2   = rsm2[idx]
    obs_r, _ = spearmanr(v1, v2)

    count = 0
    for _ in range(n_permutations):
        perm   = rng.permutation(n)
        rsm1_p = rsm1[np.ix_(perm, perm)]
        r_p, _ = spearmanr(rsm1_p[idx], v2)
        if r_p >= obs_r:
            count += 1

    p_val = (count + 1) / (n_permutations + 1)
    return obs_r, p_val

# build acoustic RSM from per-phoneme centroids (Lobanov F1/F2)
vowel_centroids_ac = (vowel_df.groupby("phoneme")[["F1_lob", "F2_lob"]]
                      .mean().dropna())
common_phonemes    = sorted(vowel_centroids_ac.index)
ac_matrix          = vowel_centroids_ac.loc[common_phonemes].values

# negative Euclidean distance as similarity (as per assignment)
ac_rsm = -cdist(ac_matrix, ac_matrix, metric="euclidean")

mantel_rows = []

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data = np.load(npz_path)
    keys = list(data.files)

    # per-phoneme mean embeddings for common vowels
    phoneme_vecs = {}
    for ph in common_phonemes:
        ph_keys = [k for k in keys
                   if key_meta.get(k, {}).get("phoneme") == ph]
        if not ph_keys:
            continue
        vecs = np.stack([data[k] for k in ph_keys]).astype(np.float64)
        phoneme_vecs[ph] = vecs.mean(axis=0)

    if len(phoneme_vecs) < 3:
        continue

    ph_list   = sorted(phoneme_vecs.keys())
    neu_matrix = np.stack([phoneme_vecs[p] for p in ph_list])
    neu_rsm    = -cdist(neu_matrix, neu_matrix, metric="cosine")

    # align RSMs to same phoneme set
    common = sorted(set(ph_list) & set(common_phonemes))
    ac_idx  = [common_phonemes.index(p) for p in common]
    neu_idx = [ph_list.index(p)         for p in common]

    ac_rsm_sub  = ac_rsm[np.ix_(ac_idx, ac_idx)]
    neu_rsm_sub = neu_rsm[np.ix_(neu_idx, neu_idx)]

    r, p = mantel_test(ac_rsm_sub, neu_rsm_sub, n_permutations=999)
    mantel_rows.append({
        "model":   name,
        "mantel_r": round(r, 4),
        "p_value":  round(p, 4),
        "n_phonemes": len(common),
    })
    print(f"  {name:15s} Mantel r={r:.4f}  p={p:.4f}  "
          f"(n={len(common)} phonemes)")

pd.DataFrame(mantel_rows).to_csv(RESULTS_DIR / "mantel_results.csv", index=False)


# ── done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Section 5 complete. Files saved to results/:")
print("=" * 55)
for p in sorted(RESULTS_DIR.rglob("*.png")):
    print(f"  {p.relative_to(RESULTS_DIR)}")
for p in sorted(RESULTS_DIR.glob("*.csv")):
    print(f"  {p.name}")