import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix, f1_score
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

VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ", "ə"}
N_PERMUTATIONS = 5000
RANDOM_STATE   = 42
rng = np.random.default_rng(RANDOM_STATE)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
df       = pd.read_csv(ACOUSTIC_NORM)
vowel_df = df[df["phoneme"].isin(VOWELS)].copy()

# key -> metadata lookup for neural files
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

print(f"  Tokens: {len(df)}, Vowels: {len(vowel_df)}, "
      f"Speakers: {df['speaker'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.1 GROUP COMPARISONS — ACOUSTIC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6.1 Group comparisons — acoustic")
print("="*60)

# ── normality + homogeneity ───────────────────────────────────────────────────
print("\n  Shapiro-Wilk + Levene per vowel ...")
norm_rows = []

for phoneme in sorted(vowel_df["phoneme"].unique()):
    for formant in ["F1_lob", "F2_lob"]:
        l1_vals = vowel_df[(vowel_df["phoneme"] == phoneme) &
                           (vowel_df["L1"] == "fr")][formant].dropna()
        l2_vals = vowel_df[(vowel_df["phoneme"] == phoneme) &
                           (vowel_df["L1"] == "ru")][formant].dropna()

        if len(l1_vals) < 3 or len(l2_vals) < 3:
            continue

        sw_l1 = shapiro(l1_vals) if len(l1_vals) <= 5000 else (np.nan, np.nan)
        sw_l2 = shapiro(l2_vals) if len(l2_vals) <= 5000 else (np.nan, np.nan)
        lev   = levene(l1_vals, l2_vals)

        norm_rows.append({
            "phoneme":     phoneme,
            "formant":     formant,
            "n_L1":        len(l1_vals),
            "n_L2":        len(l2_vals),
            "sw_p_L1":     round(sw_l1[1], 4),
            "sw_p_L2":     round(sw_l2[1], 4),
            "normal_L1":   sw_l1[1] > 0.05,
            "normal_L2":   sw_l2[1] > 0.05,
            "levene_p":    round(lev.pvalue, 4),
            "equal_var":   lev.pvalue > 0.05,
        })

norm_df = pd.DataFrame(norm_rows)
norm_df.to_csv(RESULTS_DIR / "6b_normality_levene.csv", index=False)
print(norm_df[["phoneme", "formant", "normal_L1",
               "normal_L2", "equal_var"]].to_string(index=False))


# ── t-test or Mann-Whitney + BH correction ────────────────────────────────────
print("\n  L1 vs L2 tests + BH correction ...")
test_rows = []

for phoneme in sorted(vowel_df["phoneme"].unique()):
    for formant in ["F1_lob", "F2_lob"]:
        l1_vals = vowel_df[(vowel_df["phoneme"] == phoneme) &
                           (vowel_df["L1"] == "fr")][formant].dropna()
        l2_vals = vowel_df[(vowel_df["phoneme"] == phoneme) &
                           (vowel_df["L1"] == "ru")][formant].dropna()

        if len(l1_vals) < 3 or len(l2_vals) < 3:
            continue

        # decide test based on normality
        norm_row = norm_df[(norm_df["phoneme"] == phoneme) &
                           (norm_df["formant"] == formant)]
        both_normal = (norm_row["normal_L1"].values[0] and
                       norm_row["normal_L2"].values[0]) if len(norm_row) > 0 else False

        if both_normal:
            stat, p = ttest_ind(l1_vals, l2_vals, equal_var=False)
            test_used = "welch_t"
        else:
            stat, p = mannwhitneyu(l1_vals, l2_vals, alternative="two-sided")
            test_used = "mann_whitney"

        effect_size = (l1_vals.mean() - l2_vals.mean()) / np.sqrt(
            (l1_vals.std()**2 + l2_vals.std()**2) / 2
        )

        test_rows.append({
            "phoneme":     phoneme,
            "formant":     formant,
            "mean_L1":     round(l1_vals.mean(), 4),
            "mean_L2":     round(l2_vals.mean(), 4),
            "diff":        round(l1_vals.mean() - l2_vals.mean(), 4),
            "test":        test_used,
            "stat":        round(stat, 4),
            "p_raw":       round(p, 4),
            "cohens_d":    round(effect_size, 4),
        })

test_df = pd.DataFrame(test_rows)

# BH correction across all tests
_, p_adj, _, _ = multipletests(test_df["p_raw"], method="fdr_bh")
test_df["p_adj_BH"]    = p_adj.round(4)
test_df["sig_raw"]     = test_df["p_raw"] < 0.05
test_df["sig_BH"]      = test_df["p_adj_BH"] < 0.05

test_df.to_csv(RESULTS_DIR / "6b_L1L2_acoustic_tests.csv", index=False)

print("\n  Significant after BH correction:")
sig = test_df[test_df["sig_BH"]]
print(sig[["phoneme", "formant", "diff", "test",
           "p_raw", "p_adj_BH"]].to_string(index=False))


# ── residual gender effect after Lobanov ──────────────────────────────────────
print("\n  Residual gender effect (paired test at speaker level) ...")
gender_rows = []

for phoneme in sorted(vowel_df["phoneme"].unique()):
    for formant in ["F1_lob", "F2_lob"]:
        spk_means = (vowel_df[vowel_df["phoneme"] == phoneme]
                     .groupby(["speaker", "gender"])[formant]
                     .mean().reset_index())
        male   = spk_means[spk_means["gender"] == "m"][formant].dropna()
        female = spk_means[spk_means["gender"] == "f"][formant].dropna()

        if len(male) < 2 or len(female) < 2:
            continue

        stat, p = mannwhitneyu(male, female, alternative="two-sided")
        gender_rows.append({
            "phoneme":   phoneme,
            "formant":   formant,
            "mean_m":    round(male.mean(), 4),
            "mean_f":    round(female.mean(), 4),
            "p_value":   round(p, 4),
            "sig":       p < 0.05,
        })

gender_df = pd.DataFrame(gender_rows)
gender_df.to_csv(RESULTS_DIR / "6b_gender_test.csv", index=False)
print(gender_df[gender_df["sig"]][["phoneme", "formant",
                                    "mean_m", "mean_f", "p_value"]].to_string(index=False))


# ── permutation test on neural representations ────────────────────────────────
print("\n  Permutation test: L1 vs L2 on neural representations ...")
perm_rows = []

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data = np.load(npz_path)
    keys = list(data.files)

    for phoneme in sorted(VOWELS):
        ph_keys = [k for k in keys
                   if key_meta.get(k, {}).get("phoneme") == phoneme]
        if len(ph_keys) < 4:
            continue

        l1_keys = [k for k in ph_keys if key_meta[k]["L1"] == "fr"]
        l2_keys = [k for k in ph_keys if key_meta[k]["L1"] == "ru"]

        if len(l1_keys) < 2 or len(l2_keys) < 2:
            continue

        l1_vecs = np.stack([data[k] for k in l1_keys]).astype(np.float64)
        l2_vecs = np.stack([data[k] for k in l2_keys]).astype(np.float64)

        # observed centroid distance
        l1_centroid = l1_vecs.mean(axis=0)
        l2_centroid = l2_vecs.mean(axis=0)

        def cosine_dist(a, b):
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        obs_dist = cosine_dist(l1_centroid, l2_centroid)

        # permutation null distribution
        all_vecs = np.vstack([l1_vecs, l2_vecs])
        n_l1     = len(l1_vecs)
        null_dists = []

        for _ in range(N_PERMUTATIONS):
            perm     = rng.permutation(len(all_vecs))
            c1       = all_vecs[perm[:n_l1]].mean(axis=0)
            c2       = all_vecs[perm[n_l1:]].mean(axis=0)
            null_dists.append(cosine_dist(c1, c2))

        p_val = (np.sum(np.array(null_dists) >= obs_dist) + 1) / (N_PERMUTATIONS + 1)

        perm_rows.append({
            "model":    name,
            "phoneme":  phoneme,
            "obs_dist": round(obs_dist, 4),
            "p_raw":    round(p_val, 4),
        })

perm_df = pd.DataFrame(perm_rows)

# BH correction
_, p_adj, _, _ = multipletests(perm_df["p_raw"], method="fdr_bh")
perm_df["p_adj_BH"] = p_adj.round(4)
perm_df["sig_BH"]   = perm_df["p_adj_BH"] < 0.05
perm_df.to_csv(RESULTS_DIR / "6b_permutation_neural.csv", index=False)

print("\n  Significant neural L1/L2 differences after BH correction:")
sig_perm = perm_df[perm_df["sig_BH"]]
if len(sig_perm) > 0:
    print(sig_perm[["model", "phoneme", "obs_dist",
                     "p_raw", "p_adj_BH"]].to_string(index=False))
else:
    print("  None significant after BH correction")


# ══════════════════════════════════════════════════════════════════════════════
# 6.2 INTER-PHONEME DISTANCES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6.2 Inter-phoneme distances")
print("="*60)

# ── acoustic distance matrices ────────────────────────────────────────────────
print("\n  Acoustic distance matrices ...")

vowel_centroids = (vowel_df.groupby("phoneme")[["F1_lob", "F2_lob"]]
                   .mean().dropna())
common_phonemes = sorted(vowel_centroids.index)
ac_matrix       = vowel_centroids.loc[common_phonemes].values

# pooled within-phoneme covariance for Mahalanobis
cov_list = []
for ph in common_phonemes:
    ph_data = vowel_df[vowel_df["phoneme"] == ph][["F1_lob", "F2_lob"]].dropna()
    if len(ph_data) >= 3:
        cov_list.append(np.cov(ph_data.values.T))
pooled_cov = np.mean(cov_list, axis=0)

try:
    cov_inv    = np.linalg.inv(pooled_cov)
    D_maha     = cdist(ac_matrix, ac_matrix, metric="mahalanobis", VI=cov_inv)
except np.linalg.LinAlgError:
    print("  [WARN] Mahalanobis failed, using identity")
    D_maha = cdist(ac_matrix, ac_matrix, metric="euclidean")

D_eucl = cdist(ac_matrix, ac_matrix, metric="euclidean")

print(f"  Acoustic distance matrix: {len(common_phonemes)} phonemes")
print(f"  Phonemes: {common_phonemes}")


# ── neural distance matrices ──────────────────────────────────────────────────
print("\n  Neural distance matrices ...")
neural_dist_matrices = {}

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data = np.load(npz_path)
    keys = list(data.files)

    phoneme_vecs = {}
    for ph in common_phonemes:
        ph_keys = [k for k in keys
                   if key_meta.get(k, {}).get("phoneme") == ph]
        if ph_keys:
            vecs = np.stack([data[k] for k in ph_keys]).astype(np.float64)
            phoneme_vecs[ph] = vecs.mean(axis=0)

    if len(phoneme_vecs) < 3:
        continue

    ph_list    = sorted(phoneme_vecs.keys())
    neu_matrix = np.stack([phoneme_vecs[p] for p in ph_list])
    D_neu      = cdist(neu_matrix, neu_matrix, metric="cosine")
    neural_dist_matrices[name] = (ph_list, D_neu)
    print(f"  {name}: {len(ph_list)} phonemes")


# ── Mantel test between all distance matrix pairs ─────────────────────────────
print("\n  Mantel tests between distance matrices ...")

def mantel_test(d1, d2, n_perm=999):
    rng_m = np.random.default_rng(42)
    n     = d1.shape[0]
    idx   = np.triu_indices(n, k=1)
    v1, v2 = d1[idx], d2[idx]
    obs_r, _ = spearmanr(v1, v2)
    count = sum(
        spearmanr(d1[np.ix_(p := rng_m.permutation(n), p)][idx], v2)[0] >= obs_r
        for _ in range(n_perm)
    )
    return obs_r, (count + 1) / (n_perm + 1)

mantel_rows = []

# acoustic Euclidean vs each neural
for name, (ph_list, D_neu) in neural_dist_matrices.items():
    common = sorted(set(ph_list) & set(common_phonemes))
    ac_idx  = [common_phonemes.index(p) for p in common]
    neu_idx = [ph_list.index(p)         for p in common]

    r_e, p_e = mantel_test(D_eucl[np.ix_(ac_idx, ac_idx)],
                            D_neu[np.ix_(neu_idx, neu_idx)])
    r_m, p_m = mantel_test(D_maha[np.ix_(ac_idx, ac_idx)],
                            D_neu[np.ix_(neu_idx, neu_idx)])

    mantel_rows.append({"pair": f"euclidean vs {name}",
                         "mantel_r": round(r_e, 4), "p_value": round(p_e, 4)})
    mantel_rows.append({"pair": f"mahalanobis vs {name}",
                         "mantel_r": round(r_m, 4), "p_value": round(p_m, 4)})
    print(f"  Euclidean vs {name}: r={r_e:.4f} p={p_e:.4f}")
    print(f"  Mahalanobis vs {name}: r={r_m:.4f} p={p_m:.4f}")

# neural vs neural pairs
neural_names = list(neural_dist_matrices.keys())
for i in range(len(neural_names)):
    for j in range(i + 1, len(neural_names)):
        n1, (ph1, D1) = neural_names[i], neural_dist_matrices[neural_names[i]]
        n2, (ph2, D2) = neural_names[j], neural_dist_matrices[neural_names[j]]
        common = sorted(set(ph1) & set(ph2))
        i1 = [ph1.index(p) for p in common]
        i2 = [ph2.index(p) for p in common]
        r, p = mantel_test(D1[np.ix_(i1, i1)], D2[np.ix_(i2, i2)])
        mantel_rows.append({"pair": f"{n1} vs {n2}",
                             "mantel_r": round(r, 4), "p_value": round(p, 4)})
        print(f"  {n1} vs {n2}: r={r:.4f} p={p:.4f}")

pd.DataFrame(mantel_rows).to_csv(RESULTS_DIR / "6b_mantel_distances.csv", index=False)


# ── bootstrap CIs on selected phoneme pairs (fast version) ───────────────────
print("\n  Bootstrap CIs on phoneme pair distances ...")

TARGET_PAIRS = [("e", "ɛ"), ("i", "y"), ("a", "ɑ")]
B            = 2000
boot_rows    = []

for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data = np.load(npz_path)
    keys = list(data.files)

    # precompute per-speaker per-phoneme mean vectors once
    spk_ph_vecs = {}  # (speaker, phoneme) -> mean vector
    for k in keys:
        m = key_meta.get(k)
        if m is None:
            continue
        key2 = (m["speaker"], m["phoneme"])
        if key2 not in spk_ph_vecs:
            spk_ph_vecs[key2] = []
        spk_ph_vecs[key2].append(data[k].astype(np.float64))

    for key2 in spk_ph_vecs:
        spk_ph_vecs[key2] = np.mean(spk_ph_vecs[key2], axis=0)

    speakers = sorted({k[0] for k in spk_ph_vecs.keys()})

    for ph1, ph2 in TARGET_PAIRS:
        # only keep speakers that have both phonemes
        valid_spks = [s for s in speakers
                      if (s, ph1) in spk_ph_vecs
                      and (s, ph2) in spk_ph_vecs]
        if len(valid_spks) < 3:
            continue

        # observed distance
        c1_obs = np.mean([spk_ph_vecs[(s, ph1)] for s in valid_spks], axis=0)
        c2_obs = np.mean([spk_ph_vecs[(s, ph2)] for s in valid_spks], axis=0)
        obs_d  = 1 - np.dot(c1_obs, c2_obs) / (
            np.linalg.norm(c1_obs) * np.linalg.norm(c2_obs) + 1e-8)

        # bootstrap — resample speakers
        boot_dists = []
        for _ in range(B):
            sampled = rng.choice(valid_spks, size=len(valid_spks), replace=True)
            c1 = np.mean([spk_ph_vecs[(s, ph1)] for s in sampled], axis=0)
            c2 = np.mean([spk_ph_vecs[(s, ph2)] for s in sampled], axis=0)
            d  = 1 - np.dot(c1, c2) / (
                np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
            boot_dists.append(d)

        ci_lo, ci_hi = np.percentile(boot_dists, [2.5, 97.5])
        boot_rows.append({
            "model": name, "pair": f"{ph1}-{ph2}",
            "obs":   round(obs_d, 4),
            "mean":  round(np.mean(boot_dists), 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        })
        print(f"  {name} /{ph1}/-/{ph2}/: "
              f"{np.mean(boot_dists):.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / "6b_bootstrap_CIs.csv", index=False)


# ── nearest-centroid classifier (leave-one-speaker-out) ───────────────────────
print("\n  Nearest-centroid classifier (LOSO) ...")

def loso_classifier(keys, key_meta, get_vec, phonemes_set):
    speakers  = sorted({key_meta[k]["speaker"]
                        for k in keys if k in key_meta
                        and key_meta[k]["phoneme"] in phonemes_set})
    y_true, y_pred = [], []

    for test_spk in speakers:
        train_keys = [k for k in keys
                      if key_meta.get(k, {}).get("speaker") != test_spk
                      and key_meta.get(k, {}).get("phoneme") in phonemes_set]
        test_keys  = [k for k in keys
                      if key_meta.get(k, {}).get("speaker") == test_spk
                      and key_meta.get(k, {}).get("phoneme") in phonemes_set]

        if not train_keys or not test_keys:
            continue

        # compute centroids per phoneme from training data
        train_phones = sorted({key_meta[k]["phoneme"] for k in train_keys})
        centroids    = {}
        for ph in train_phones:
            ph_vecs = [get_vec(k) for k in train_keys
                       if key_meta[k]["phoneme"] == ph]
            centroids[ph] = np.mean(ph_vecs, axis=0)

        centroid_matrix = np.stack(list(centroids.values()))
        centroid_labels = list(centroids.keys())

        for k in test_keys:
            vec  = get_vec(k).reshape(1, -1)
            dist = cdist(vec, centroid_matrix, metric="cosine")[0]
            pred = centroid_labels[np.argmin(dist)]
            y_true.append(key_meta[k]["phoneme"])
            y_pred.append(pred)

    return y_true, y_pred

clf_rows = []
all_preds = {}  # store for McNemar

# acoustic classifier
print("  Acoustic ...")
ac_keys = [f"{row['speaker']}__{row['phoneme']}__{row['sentence_id']}__{row['onset']:.4f}"
           for _, row in vowel_df.iterrows()]

ac_vec_map = {}
for _, row in vowel_df.iterrows():
    k = f"{row['speaker']}__{row['phoneme']}__{row['sentence_id']}__{row['onset']:.4f}"
    ac_vec_map[k] = np.array([row["F1_lob"], row["F2_lob"]])

valid_ac_keys = [k for k in ac_keys if k in ac_vec_map
                 and not np.isnan(ac_vec_map[k]).any()]

y_true_ac, y_pred_ac = loso_classifier(
    valid_ac_keys, key_meta,
    lambda k: ac_vec_map[k], VOWELS
)
acc_ac = np.mean(np.array(y_true_ac) == np.array(y_pred_ac))
f1_ac  = f1_score(y_true_ac, y_pred_ac, average="macro", zero_division=0)
clf_rows.append({"model": "acoustic", "accuracy": round(acc_ac, 4),
                 "macro_f1": round(f1_ac, 4)})
all_preds["acoustic"] = (y_true_ac, y_pred_ac)
print(f"    accuracy={acc_ac:.4f}  macro_f1={f1_ac:.4f}")

# neural classifiers
for name, npz_path in NEURAL_FILES.items():
    if not npz_path.exists():
        continue
    data = np.load(npz_path)
    keys = [k for k in data.files if k in key_meta
            and key_meta[k]["phoneme"] in VOWELS]

    y_true_n, y_pred_n = loso_classifier(
        keys, key_meta,
        lambda k, d=data: d[k].astype(np.float64), VOWELS
    )
    acc  = np.mean(np.array(y_true_n) == np.array(y_pred_n))
    f1   = f1_score(y_true_n, y_pred_n, average="macro", zero_division=0)
    clf_rows.append({"model": name, "accuracy": round(acc, 4),
                     "macro_f1": round(f1, 4)})
    all_preds[name] = (y_true_n, y_pred_n)
    print(f"    {name}: accuracy={acc:.4f}  macro_f1={f1:.4f}")

clf_df = pd.DataFrame(clf_rows)
clf_df.to_csv(RESULTS_DIR / "6b_classifier_results.csv", index=False)


# ── confusion matrices ────────────────────────────────────────────────────────
print("\n  Confusion matrices ...")
n_models = len(all_preds)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for ax, (name, (yt, yp)) in zip(axes, all_preds.items()):
    labels = sorted(set(yt) | set(yp))
    cm     = confusion_matrix(yt, yp, labels=labels, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, ax=ax, cmap="Blues",
                annot_kws={"size": 7})
    ax.set_title(f"{name}\nacc={np.mean(np.array(yt)==np.array(yp)):.3f}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.tick_params(labelsize=8)

for ax in axes[n_models:]:
    ax.set_visible(False)

plt.suptitle("Confusion matrices — nearest-centroid LOSO classifier", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "6b_confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved -> results/6b_confusion_matrices.png")


# ── McNemar test ──────────────────────────────────────────────────────────────
print("\n  McNemar test across representation types ...")
from statsmodels.stats.contingency_tables import mcnemar

mcnemar_rows = []
model_names  = list(all_preds.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        n1, n2   = model_names[i], model_names[j]
        yt1, yp1 = all_preds[n1]
        yt2, yp2 = all_preds[n2]

        # align on common true labels
        min_len  = min(len(yt1), len(yt2))
        correct1 = np.array(yt1[:min_len]) == np.array(yp1[:min_len])
        correct2 = np.array(yt2[:min_len]) == np.array(yp2[:min_len])

        # contingency table
        b = np.sum(correct1 & ~correct2)   # n1 right, n2 wrong
        c = np.sum(~correct1 & correct2)   # n1 wrong, n2 right

        if b + c == 0:
            continue

        table    = np.array([[np.sum(correct1 & correct2), b],
                              [c, np.sum(~correct1 & ~correct2)]])
        result   = mcnemar(table, exact=False, correction=True)
        mcnemar_rows.append({
            "model1":  n1,
            "model2":  n2,
            "b":       int(b),
            "c":       int(c),
            "chi2":    round(result.statistic, 4),
            "p_value": round(result.pvalue, 4),
            "sig":     result.pvalue < 0.05,
        })
        print(f"  {n1} vs {n2}: chi2={result.statistic:.3f} "
              f"p={result.pvalue:.4f} {'*' if result.pvalue < 0.05 else ''}")

pd.DataFrame(mcnemar_rows).to_csv(RESULTS_DIR / "6b_mcnemar.csv", index=False)


# ── done ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Section 6b complete. Files saved to results/:")
print("="*60)
for p in sorted(RESULTS_DIR.glob("6b_*")):
    print(f"  {p.name}")