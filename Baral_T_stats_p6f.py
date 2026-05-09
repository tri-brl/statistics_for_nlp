"""
Stage 6e — Section 9: Hierarchical Clustering
===============================================
9.1 Clustering of French oral vowels
9.2 Consonants vs vowels
9.3 Clustering of speakers
9.4 Number of clusters (silhouette + dendrogram)

Inputs:
  C:\\Users\\aviba\\data\\features_acoustic_norm.csv
  C:\\Users\\aviba\\data\\features_*_pca.npz (50D)
  Neural raw npz files for cosine distance

Outputs:
  results/6e_ARI_vowels.csv
  results/6e_ARI_speakers.csv
  results/6e_silhouette.csv
  results/6e_dendrograms/     — one per representation
  results/6e_heatmaps/        — distance matrix heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\aviba\data")
RESULTS_DIR = Path(r"C:\Users\aviba\results")
(RESULTS_DIR / "6e_dendrograms").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "6e_heatmaps").mkdir(parents=True, exist_ok=True)

ACOUSTIC_NORM = DATA_DIR / "features_acoustic_norm.csv"
NEURAL_FILES  = {
    "whisper_L2":  DATA_DIR / "features_whisper_L2.npz",
    "whisper_L5":  DATA_DIR / "features_whisper_L5.npz",
    "xlsr_L3":     DATA_DIR / "features_xlsr_L3.npz",
    "xlsr_L12":    DATA_DIR / "features_xlsr_L12.npz",
    "xlsr_L21":    DATA_DIR / "features_xlsr_L21.npz",
}
PCA_FILES = {k: DATA_DIR / f"features_{k}_pca.npz" for k in NEURAL_FILES}

VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ", "ə"}

# consonants: covering different manner and place classes
# plosives: p, t, k, d
# fricatives: f, s, ʃ, ʒ
# sonorants: l, ʁ
CONSONANTS = {"p", "t", "k", "d", "f", "s", "ʃ", "ʒ", "l", "ʁ"}

# ground truth phonological categories for ARI evaluation
# front/back distinction
FRONT_BACK = {
    "i": "front", "e": "front", "ɛ": "front", "y": "front",
    "ø": "front", "œ": "front",
    "a": "back",  "ɑ": "back",  "u": "back",  "o": "back",
    "ɔ": "back",  "ə": "back",
}
# high/mid/low distinction
HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "œ": "mid",  "ɔ": "mid",
    "a": "low",  "ɑ": "low",  "ə": "mid",
}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df       = pd.read_csv(ACOUSTIC_NORM)
vowel_df = df[df["phoneme"].isin(VOWELS)].copy()
cons_df  = df[df["phoneme"].isin(CONSONANTS)].copy()

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

print(f"  Vowel tokens: {len(vowel_df)}, Consonant tokens: {len(cons_df)}")


# ── helpers ───────────────────────────────────────────────────────────────────
def ward_cluster(matrix, labels, metric="euclidean"):
    """
    Hierarchical clustering with Ward linkage.
    matrix: (n_items, n_features) or (n_items, n_items) distance matrix
    Returns linkage matrix Z and labels.
    """
    if metric == "precomputed":
        condensed = squareform(matrix)
        Z = linkage(condensed, method="ward")
    else:
        Z = linkage(matrix, method="ward", metric=metric)
    return Z


def compute_ari(Z, labels, n_clusters, true_labels):
    """Cut dendrogram at n_clusters and compute ARI against true labels."""
    pred = fcluster(Z, n_clusters, criterion="maxclust")
    le   = LabelEncoder()
    true_enc = le.fit_transform(true_labels)
    return adjusted_rand_score(true_enc, pred)


def plot_dendrogram(Z, labels, title, out_path, color_threshold=None):
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 6))
    dendrogram(
        Z, labels=labels, ax=ax,
        color_threshold=color_threshold,
        leaf_rotation=45, leaf_font_size=11,
    )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_heatmap(dist_matrix, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        dist_matrix, xticklabels=labels, yticklabels=labels,
        cmap="viridis_r", ax=ax, annot=True, fmt=".2f",
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def silhouette_sweep(matrix, labels, metric="precomputed"):
    """Compute silhouette for k=2..n-1 clusters."""
    results = []
    Z = ward_cluster(matrix, labels, metric="precomputed")
    for k in range(2, len(labels)):
        pred = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(pred)) < 2:
            continue
        try:
            s = silhouette_score(matrix, pred, metric="precomputed")
            results.append({"k": k, "silhouette": round(s, 4)})
        except Exception:
            pass
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 9.1 CLUSTERING OF FRENCH ORAL VOWELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9.1 Clustering of French oral vowels")
print("="*60)

ari_rows     = []
sil_rows     = []

# ── acoustic vowel clustering ─────────────────────────────────────────────────
print("\n  Acoustic ...")
ac_centroids = (vowel_df.groupby("phoneme")[["F1_lob", "F2_lob"]]
                .mean().dropna())
ac_phonemes  = sorted(ac_centroids.index)
ac_matrix    = ac_centroids.loc[ac_phonemes].values

D_ac = squareform(pdist(ac_matrix, metric="euclidean"))
Z_ac = ward_cluster(D_ac, ac_phonemes, metric="precomputed")

plot_dendrogram(Z_ac, [f"/{p}/" for p in ac_phonemes],
                "Acoustic vowel clustering (Ward, Euclidean)",
                RESULTS_DIR / "6e_dendrograms" / "acoustic_vowels.png")

plot_heatmap(D_ac, [f"/{p}/" for p in ac_phonemes],
             "Acoustic vowel distance matrix",
             RESULTS_DIR / "6e_heatmaps" / "acoustic_vowels.png")

# ARI for front/back and high/mid/low
for gt_name, gt_dict in [("front_back", FRONT_BACK), ("height", HEIGHT)]:
    gt_labels = [gt_dict.get(p, "other") for p in ac_phonemes]
    n_true    = len(set(gt_labels))
    ari       = compute_ari(Z_ac, ac_phonemes, n_true, gt_labels)
    ari_rows.append({
        "representation": "acoustic",
        "phoneme_set":    "vowels",
        "ground_truth":   gt_name,
        "n_clusters":     n_true,
        "ARI":            round(ari, 4),
    })
    print(f"    ARI ({gt_name}): {ari:.4f}")

# silhouette
sil_ac = silhouette_sweep(D_ac, ac_phonemes)
sil_ac["representation"] = "acoustic"
sil_rows.append(sil_ac)
best_k_ac = sil_ac.loc[sil_ac["silhouette"].idxmax(), "k"] if len(sil_ac) > 0 else 2
print(f"    Best k (silhouette): {best_k_ac}")


# ── neural vowel clustering ───────────────────────────────────────────────────
for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    print(f"\n  {name} ...")
    data = np.load(npz_path)
    keys = list(data.files)

    # per-phoneme mean embeddings
    ph_vecs = {}
    for ph in VOWELS:
        ph_keys = [k for k in keys
                   if key_meta.get(k, {}).get("phoneme") == ph]
        if ph_keys:
            ph_vecs[ph] = np.stack([data[k] for k in ph_keys]).mean(axis=0)

    phonemes = sorted(ph_vecs.keys())
    if len(phonemes) < 3:
        continue

    neu_matrix = np.stack([ph_vecs[p] for p in phonemes])
    D_neu      = squareform(pdist(neu_matrix, metric="cosine"))

    Z_neu = ward_cluster(D_neu, phonemes, metric="precomputed")

    plot_dendrogram(Z_neu, [f"/{p}/" for p in phonemes],
                    f"{name} vowel clustering (Ward, cosine)",
                    RESULTS_DIR / "6e_dendrograms" / f"{name}_vowels.png")

    plot_heatmap(D_neu, [f"/{p}/" for p in phonemes],
                 f"{name} vowel distance matrix",
                 RESULTS_DIR / "6e_heatmaps" / f"{name}_vowels.png")

    for gt_name, gt_dict in [("front_back", FRONT_BACK), ("height", HEIGHT)]:
        gt_labels = [gt_dict.get(p, "other") for p in phonemes]
        n_true    = len(set(gt_labels))
        ari       = compute_ari(Z_neu, phonemes, n_true, gt_labels)
        ari_rows.append({
            "representation": name,
            "phoneme_set":    "vowels",
            "ground_truth":   gt_name,
            "n_clusters":     n_true,
            "ARI":            round(ari, 4),
        })
        print(f"    ARI ({gt_name}): {ari:.4f}")

    sil_neu = silhouette_sweep(D_neu, phonemes)
    sil_neu["representation"] = name
    sil_rows.append(sil_neu)
    if len(sil_neu) > 0:
        best_k = sil_neu.loc[sil_neu["silhouette"].idxmax(), "k"]
        print(f"    Best k (silhouette): {best_k}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.2 CONSONANTS VS VOWELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9.2 Consonants vs vowels")
print("="*60)

# manner class ground truth for ARI
MANNER = {
    "p": "plosive",  "t": "plosive",  "k": "plosive", "d": "plosive",
    "f": "fricative","s": "fricative","ʃ": "fricative","ʒ": "fricative",
    "l": "sonorant", "ʁ": "sonorant",
    "a": "vowel", "e": "vowel", "i": "vowel", "o": "vowel",
    "u": "vowel", "y": "vowel", "ø": "vowel", "œ": "vowel",
    "ɛ": "vowel", "ɑ": "vowel", "ɔ": "vowel", "ə": "vowel",
}
CV_LABEL = {p: "vowel" if p in VOWELS else "consonant"
            for p in list(VOWELS) + list(CONSONANTS)}

ALL_PHONES = VOWELS | CONSONANTS

# acoustic: F1/F2 + duration for consonants
print("\n  Acoustic (vowels + consonants) ...")
all_df = df[df["phoneme"].isin(ALL_PHONES)].copy()

# use F1, F2, duration_ms
ac_all_centroids = (all_df.groupby("phoneme")[["F1_lob", "F2_lob", "duration_ms"]]
                    .mean().dropna())
# standardise duration
ac_all_centroids["duration_ms"] = (
    (ac_all_centroids["duration_ms"] - ac_all_centroids["duration_ms"].mean())
    / ac_all_centroids["duration_ms"].std()
)

all_phonemes = sorted(ac_all_centroids.index)
ac_all_mat   = ac_all_centroids.loc[all_phonemes].values
D_ac_all     = squareform(pdist(ac_all_mat, metric="euclidean"))
Z_ac_all     = ward_cluster(D_ac_all, all_phonemes, metric="precomputed")

plot_dendrogram(Z_ac_all, [f"/{p}/" for p in all_phonemes],
                "Acoustic: vowels + consonants (Ward, Euclidean)",
                RESULTS_DIR / "6e_dendrograms" / "acoustic_all.png",
                color_threshold=0.7 * max(Z_ac_all[:, 2]))

# ARI: CV boundary
cv_labels = [CV_LABEL.get(p, "other") for p in all_phonemes]
ari_cv_ac = compute_ari(Z_ac_all, all_phonemes, 2, cv_labels)
ari_rows.append({
    "representation": "acoustic",
    "phoneme_set":    "vowels+consonants",
    "ground_truth":   "CV_boundary",
    "n_clusters":     2,
    "ARI":            round(ari_cv_ac, 4),
})
print(f"    ARI (CV boundary): {ari_cv_ac:.4f}")

# neural: vowels + consonants
for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    print(f"\n  {name} (vowels + consonants) ...")
    data = np.load(npz_path)
    keys = list(data.files)

    ph_vecs = {}
    for ph in ALL_PHONES:
        ph_keys = [k for k in keys
                   if key_meta.get(k, {}).get("phoneme") == ph]
        if ph_keys:
            ph_vecs[ph] = np.stack([data[k] for k in ph_keys]).mean(axis=0)

    phonemes = sorted(ph_vecs.keys())
    if len(phonemes) < 4:
        continue

    neu_mat = np.stack([ph_vecs[p] for p in phonemes])
    D_neu   = squareform(pdist(neu_mat, metric="cosine"))
    Z_neu   = ward_cluster(D_neu, phonemes, metric="precomputed")

    plot_dendrogram(Z_neu, [f"/{p}/" for p in phonemes],
                    f"{name}: vowels + consonants",
                    RESULTS_DIR / "6e_dendrograms" / f"{name}_all.png",
                    color_threshold=0.7 * max(Z_neu[:, 2]))

    cv_labels = [CV_LABEL.get(p, "other") for p in phonemes]
    ari_cv    = compute_ari(Z_neu, phonemes, 2, cv_labels)
    ari_rows.append({
        "representation": name,
        "phoneme_set":    "vowels+consonants",
        "ground_truth":   "CV_boundary",
        "n_clusters":     2,
        "ARI":            round(ari_cv, 4),
    })
    print(f"    ARI (CV boundary): {ari_cv:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.3 SPEAKER CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9.3 Speaker clustering")
print("="*60)

speakers     = sorted(df["speaker"].unique())
spk_L1       = df.groupby("speaker")["L1"].first()
spk_gender   = df.groupby("speaker")["gender"].first()

spk_ari_rows = []

# acoustic: per speaker, concatenate per-vowel mean F1/F2
# acoustic: find common vowels first
common_vowels_ac = sorted([
    ph for ph in VOWELS
    if vowel_df[(vowel_df["phoneme"] == ph)]["speaker"].nunique() == len(speakers)
])
print(f"  Common vowels (acoustic): {common_vowels_ac}")

spk_ac_vecs = {}
for spk in speakers:
    row_parts = []
    for ph in common_vowels_ac:
        vals = vowel_df[(vowel_df["speaker"] == spk) &
                        (vowel_df["phoneme"] == ph)][["F1_lob", "F2_lob"]].mean()
        row_parts.extend(vals.tolist())
    vec = np.array(row_parts)
    if not np.isnan(vec).all():
        spk_ac_vecs[spk] = np.nan_to_num(vec)

spk_list   = sorted(spk_ac_vecs.keys())
spk_ac_mat = np.stack([spk_ac_vecs[s] for s in spk_list])
D_spk_ac   = squareform(pdist(spk_ac_mat, metric="euclidean"))
Z_spk_ac   = ward_cluster(D_spk_ac, spk_list, metric="precomputed")

plot_dendrogram(Z_spk_ac,
                [f"{s}\n({spk_L1[s]},{spk_gender[s]})" for s in spk_list],
                "Acoustic speaker clustering",
                RESULTS_DIR / "6e_dendrograms" / "acoustic_speakers.png")

for gt_name, gt_series in [("L1", spk_L1), ("gender", spk_gender)]:
    gt_labels = [gt_series[s] for s in spk_list]
    n_true    = len(set(gt_labels))
    ari       = compute_ari(Z_spk_ac, spk_list, n_true, gt_labels)
    spk_ari_rows.append({
        "representation": "acoustic",
        "ground_truth":   gt_name,
        "ARI":            round(ari, 4),
    })
    print(f"    ARI ({gt_name}): {ari:.4f}")

# neural speaker clustering
for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    print(f"\n  {name} speaker clustering ...")
    data = np.load(npz_path)
    keys = list(data.files)

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

    # concatenate per-vowel means per speaker
    # find the vowels that ALL speakers have
    common_vowels = sorted([
        ph for ph in VOWELS
        if all((spk, ph) in spk_ph_vecs for spk in speakers)
    ])
    print(f"    Common vowels across all speakers: {common_vowels}")

    # concatenate only common vowels
    spk_neu_vecs = {}
    for spk in speakers:
        parts = [spk_ph_vecs[(spk, ph)] for ph in common_vowels]
        if parts:
            spk_neu_vecs[spk] = np.concatenate(parts)

    spk_list_n  = sorted(spk_neu_vecs.keys())
    if len(spk_list_n) < 3:
        continue

    spk_neu_mat = np.stack([spk_neu_vecs[s] for s in spk_list_n])
    D_spk_neu   = squareform(pdist(spk_neu_mat, metric="cosine"))
    Z_spk_neu   = ward_cluster(D_spk_neu, spk_list_n, metric="precomputed")

    plot_dendrogram(Z_spk_neu,
                    [f"{s}\n({spk_L1[s]},{spk_gender[s]})" for s in spk_list_n],
                    f"{name} speaker clustering",
                    RESULTS_DIR / "6e_dendrograms" / f"{name}_speakers.png")

    for gt_name, gt_series in [("L1", spk_L1), ("gender", spk_gender)]:
        gt_labels = [gt_series[s] for s in spk_list_n]
        n_true    = len(set(gt_labels))
        ari       = compute_ari(Z_spk_neu, spk_list_n, n_true, gt_labels)
        spk_ari_rows.append({
            "representation": name,
            "ground_truth":   gt_name,
            "ARI":            round(ari, 4),
        })
        print(f"    ARI ({gt_name}): {ari:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Saving ...")
print("="*60)

ari_df = pd.DataFrame(ari_rows)
ari_df.to_csv(RESULTS_DIR / "6e_ARI_vowels.csv", index=False)

spk_ari_df = pd.DataFrame(spk_ari_rows)
spk_ari_df.to_csv(RESULTS_DIR / "6e_ARI_speakers.csv", index=False)

sil_df = pd.concat(sil_rows, ignore_index=True) if sil_rows else pd.DataFrame()
sil_df.to_csv(RESULTS_DIR / "6e_silhouette.csv", index=False)

print("\n=== ARI — vowel clustering ===")
print(ari_df.pivot_table(
    index="representation", columns="ground_truth", values="ARI"
).to_string())

print("\n=== ARI — speaker clustering ===")
print(spk_ari_df.pivot_table(
    index="representation", columns="ground_truth", values="ARI"
).to_string())

print("\nDone. Files saved:")
for p in sorted(RESULTS_DIR.glob("6e_*")):
    print(f"  {p.name}")
for p in sorted((RESULTS_DIR / "6e_dendrograms").glob("*.png")):
    print(f"  6e_dendrograms/{p.name}")