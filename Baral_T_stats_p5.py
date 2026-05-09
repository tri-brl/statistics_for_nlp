import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\aviba\data")
ACOUSTIC_CSV = DATA_DIR / "features_acoustic.csv"

NEURAL_FILES = {
    "whisper_L2":  DATA_DIR / "features_whisper_L2.npz",
    "whisper_L5":  DATA_DIR / "features_whisper_L5.npz",
    "xlsr_L3":     DATA_DIR / "features_xlsr_L3.npz",
    "xlsr_L12":    DATA_DIR / "features_xlsr_L12.npz",
    "xlsr_L21":    DATA_DIR / "features_xlsr_L21.npz",
}

PCA_DIMS_VIZ      = 2
PCA_DIMS_ANALYSIS = 50
UMAP_DIMS         = 2
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.1
RANDOM_STATE      = 42

# ── 1. Lobanov normalisation ──────────────────────────────────────────────────
print("=" * 55)
print("Step 1 — Lobanov normalisation")
print("=" * 55)

df = pd.read_csv(ACOUSTIC_CSV)
print(f"Loaded {len(df)} tokens, {df['speaker'].nunique()} speakers")

# Lobanov: per speaker, per formant — normalise using vowel tokens only
# vowel set: IPA symbols that are vowels
VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ",
          "ə", "ɑ̃", "ɛ̃", "œ̃", "ɔ̃", "ɪ", "ʉ"}

vowel_mask = df["phoneme"].isin(VOWELS)
print(f"  Vowel tokens for normalisation: {vowel_mask.sum()} / {len(df)}")

df["F1_lob"] = np.nan
df["F2_lob"] = np.nan

for spk, spk_df in df.groupby("speaker"):
    # compute mean and std from vowel tokens only for this speaker
    vowel_idx = spk_df.index[spk_df["phoneme"].isin(VOWELS)]

    for formant in ["F1", "F2"]:
        vowel_vals = df.loc[vowel_idx, formant].dropna()
        if len(vowel_vals) < 2:
            print(f"  [WARN] {spk} has too few vowel tokens for Lobanov")
            continue
        mu  = vowel_vals.mean()
        std = vowel_vals.std()
        if std == 0:
            continue
        # apply to ALL tokens for this speaker (not just vowels)
        df.loc[spk_df.index, f"{formant}_lob"] = (
            df.loc[spk_df.index, formant] - mu
        ) / std

# quick sanity check
print(f"\n  Lobanov F1 stats (vowels only):")
vowel_df = df[vowel_mask]
print(f"    mean : {vowel_df['F1_lob'].mean():.4f}  (should be ~0)")
print(f"    std  : {vowel_df['F1_lob'].std():.4f}")
print(f"    range: {vowel_df['F1_lob'].min():.2f} to {vowel_df['F1_lob'].max():.2f}")

out_path = DATA_DIR / "features_acoustic_norm.csv"
df.to_csv(out_path, index=False)
print(f"\n  Saved -> {out_path}")


# ── 2. PCA on neural representations ─────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 2 — PCA on neural representations")
print("=" * 55)

pca_models = {}   # save fitted PCA objects for reuse in analysis

for name, npz_path in NEURAL_FILES.items():
    print(f"\n  {name} ...")

    if not npz_path.exists():
        print(f"    [SKIP] file not found: {npz_path}")
        continue

    data   = np.load(npz_path)
    keys   = list(data.files)
    matrix = np.stack([data[k] for k in keys]).astype(np.float64)

    # drop any rows with nan
    nan_mask = np.isnan(matrix).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"    [WARN] dropping {nan_mask.sum()} NaN vectors")
        matrix = matrix[~nan_mask]
        keys   = [k for k, m in zip(keys, nan_mask) if not m]

    print(f"    Matrix shape : {matrix.shape}")

    # standardise before PCA
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # 2D PCA for visualisation
    pca_2d = PCA(n_components=PCA_DIMS_VIZ, random_state=RANDOM_STATE)
    coords_2d = pca_2d.fit_transform(matrix_scaled)
    var_2d = pca_2d.explained_variance_ratio_.sum()
    print(f"    PCA 2D variance explained : {var_2d*100:.1f}%")

    # 50D PCA for analysis
    n_components_50 = min(PCA_DIMS_ANALYSIS, matrix.shape[1], matrix.shape[0])
    pca_50 = PCA(n_components=n_components_50, random_state=RANDOM_STATE)
    coords_50 = pca_50.fit_transform(matrix_scaled)
    var_50 = pca_50.explained_variance_ratio_.sum()
    print(f"    PCA 50D variance explained: {var_50*100:.1f}%")

    # save
    out_path = DATA_DIR / f"features_{name}_pca.npz"
    np.savez_compressed(
        out_path,
        keys      = np.array(keys),
        coords_2d = coords_2d.astype(np.float32),
        coords_50 = coords_50.astype(np.float32),
        var_2d    = np.array([var_2d]),
        var_50    = np.array([var_50]),
    )
    print(f"    Saved -> {out_path}")

    pca_models[name] = {
        "scaler": scaler,
        "pca_2d": pca_2d,
        "pca_50": pca_50,
        "keys":   keys,
    }

# save PCA models for reuse
with open(DATA_DIR / "pca_models.pkl", "wb") as f:
    pickle.dump(pca_models, f)
print(f"\n  Saved PCA models -> {DATA_DIR / 'pca_models.pkl'}")


# ── 3. UMAP on neural representations ────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 3 — UMAP on neural representations")
print("=" * 55)
print("  (this takes a few minutes per model)")

for name, npz_path in NEURAL_FILES.items():
    print(f"\n  {name} ...")

    if name not in pca_models:
        print(f"    [SKIP] no PCA model found")
        continue

    # use the 50D PCA coords as input to UMAP (faster + denoised)
    pca_npz   = np.load(DATA_DIR / f"features_{name}_pca.npz")
    coords_50 = pca_npz["coords_50"].astype(np.float64)

    reducer = umap.UMAP(
        n_components  = UMAP_DIMS,
        n_neighbors   = UMAP_N_NEIGHBORS,
        min_dist      = UMAP_MIN_DIST,
        random_state  = RANDOM_STATE,
        verbose       = False,
    )
    coords_umap = reducer.fit_transform(coords_50)
    print(f"    UMAP output shape: {coords_umap.shape}")

    out_path = DATA_DIR / f"features_{name}_umap.npz"
    np.savez_compressed(
        out_path,
        keys        = pca_npz["keys"],
        coords_umap = coords_umap.astype(np.float32),
    )
    print(f"    Saved -> {out_path}")


# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Stage 5 complete. Files saved:")
print("=" * 55)
for p in sorted(DATA_DIR.glob("features_*norm*")) :
    print(f"  {p.name}")
for p in sorted(DATA_DIR.glob("features_*pca*")):
    print(f"  {p.name}")
for p in sorted(DATA_DIR.glob("features_*umap*")):
    print(f"  {p.name}")
print(f"  pca_models.pkl")