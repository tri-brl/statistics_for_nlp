"""
Stage 6c — Section 7: Linear Mixed-Effects Models
===================================================
Uses statsmodels MixedLM (no R required).

For each representation type:
  1. Null model (random intercept only) -> ICC
  2. Main effects model (L1 + gender)
  3. Full model (L1 * gender interaction)
  4. Extended model (+ vowel height)
  5. Random slope model (L1 | speaker)

Inputs:
  C:\\Users\\aviba\\data\\features_acoustic_norm.csv
  C:\\Users\\aviba\\data\\features_*_pca.npz

Outputs:
  results/6c_ICC.csv
  results/6c_model_comparison.csv
  results/6c_fixed_effects.csv
  results/6c_R2.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(r"C:\Users\aviba\data")
RESULTS_DIR = Path(r"C:\Users\aviba\results")
RESULTS_DIR.mkdir(exist_ok=True)

ACOUSTIC_NORM = DATA_DIR / "features_acoustic_norm.csv"
PCA_FILES = {
    "whisper_L2":  DATA_DIR / "features_whisper_L2_pca.npz",
    "whisper_L5":  DATA_DIR / "features_whisper_L5_pca.npz",
    "xlsr_L3":     DATA_DIR / "features_xlsr_L3_pca.npz",
    "xlsr_L12":    DATA_DIR / "features_xlsr_L12_pca.npz",
    "xlsr_L21":    DATA_DIR / "features_xlsr_L21_pca.npz",
}

VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɑ", "ɔ", "ə"}
VOWEL_HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "œ": "mid",  "ɔ": "mid",
    "a": "low",  "ɑ": "low",  "ə": "mid",
}

# ── load acoustic data ────────────────────────────────────────────────────────
print("Loading data ...")
df       = pd.read_csv(ACOUSTIC_NORM)
vowel_df = df[df["phoneme"].isin(VOWELS)].copy()
vowel_df["height"]  = vowel_df["phoneme"].map(VOWEL_HEIGHT)
vowel_df["is_L2"]   = (vowel_df["L1"] == "ru").astype(int)
vowel_df["is_male"] = (vowel_df["gender"] == "m").astype(int)
vowel_df = vowel_df.dropna(subset=["F1_lob", "F2_lob", "is_L2", "is_male"])
print(f"  Vowel tokens: {len(vowel_df)}, Speakers: {vowel_df['speaker'].nunique()}")


# ── helpers ───────────────────────────────────────────────────────────────────
def get_var_random(result):
    """Robustly extract random intercept variance from statsmodels result."""
    try:
        cov = result.cov_re
        if hasattr(cov, "iloc"):
            return float(cov.iloc[0, 0])
        return float(cov[0, 0])
    except Exception:
        return 0.0


def fit_lme(formula, data, groups, reml=False):
    try:
        model  = smf.mixedlm(formula, data, groups=data[groups])
        result = model.fit(reml=reml, method="powell")
        return result
    except Exception as e:
        print(f"    [WARN] model failed: {e}")
        return None


def compute_icc(null_result):
    try:
        var_u = get_var_random(null_result)
        var_e = null_result.scale
        total = var_u + var_e
        if total == 0:
            return 0.0, round(var_u, 6), round(var_e, 6)
        icc = var_u / total
        return round(icc, 4), round(var_u, 6), round(var_e, 6)
    except Exception as e:
        print(f"    [ICC error] {e}")
        return 0.0, 0.0, 0.0


def lrt(m0, m1):
    try:
        lr_stat = 2 * (m1.llf - m0.llf)
        df_diff = max(1, int(m1.df_modelwc - m0.df_modelwc))
        p_val   = stats.chi2.sf(lr_stat, df_diff)
        return round(lr_stat, 4), df_diff, round(p_val, 4)
    except Exception:
        return np.nan, np.nan, np.nan


def marginal_r2(result):
    try:
        var_fixed  = np.var(result.fittedvalues)
        var_random = get_var_random(result)
        var_resid  = result.scale
        total      = var_fixed + var_random + var_resid
        if total == 0:
            return 0.0, 0.0
        r2_m = var_fixed / total
        r2_c = (var_fixed + var_random) / total
        return round(r2_m, 4), round(r2_c, 4)
    except Exception as e:
        print(f"    [R2 error] {e}")
        return 0.0, 0.0


def run_model_sequence(data, response, groups, rep_name):
    print(f"\n  [{rep_name}] response={response}")
    icc_rows, comp_rows, fe_rows, r2_rows = [], [], [], []

    # 1. null model
    m0 = fit_lme(f"{response} ~ 1", data, groups)
    if m0 is None:
        return icc_rows, comp_rows, fe_rows, r2_rows

    icc, var_u, var_e = compute_icc(m0)
    icc_rows.append({
        "representation": rep_name,
        "response":       response,
        "ICC":            icc,
        "var_random":     var_u,
        "var_residual":   var_e,
        "AIC_null":       round(m0.aic, 2),
    })
    print(f"    ICC={icc:.4f}  var_u={var_u:.6f}  var_e={var_e:.4f}")

    # 2. main effects
    m1 = fit_lme(f"{response} ~ is_L2 + is_male", data, groups)

    # 3. interaction
    m2 = fit_lme(f"{response} ~ is_L2 * is_male", data, groups)

    # 4. extended (+ height) — acoustic only
    m3 = None
    if "height" in data.columns and data["height"].notna().any():
        m3 = fit_lme(f"{response} ~ is_L2 * is_male + C(height)", data, groups)

    # 5. random slope
    m4 = None
    try:
        rs_model = MixedLM.from_formula(
            f"{response} ~ is_L2 + is_male",
            data, groups=data[groups],
            re_formula="~is_L2"
        )
        m4 = rs_model.fit(reml=False, method="powell")
    except Exception:
        pass

    # model comparison
    models = [
        ("null",         m0),
        ("main_effects", m1),
        ("interaction",  m2),
        ("extended",     m3),
        ("random_slope", m4),
    ]

    prev_m = m0
    for mname, m in models:
        if m is None:
            continue
        lr, df_d, p_lrt = lrt(prev_m, m) if mname != "null" else (np.nan, 0, np.nan)
        comp_rows.append({
            "representation": rep_name,
            "response":       response,
            "model":          mname,
            "AIC":            round(m.aic, 2),
            "BIC":            round(m.bic, 2),
            "loglik":         round(m.llf, 2),
            "LRT_stat":       lr,
            "LRT_df":         df_d,
            "LRT_p":          p_lrt,
        })
        if mname != "null":
            prev_m = m

    # fixed effects from interaction model
    best = m2 if m2 is not None else m1
    if best is not None:
        for param in best.params.index:
            if param == "Group Var":
                continue
            fe_rows.append({
                "representation": rep_name,
                "response":       response,
                "parameter":      param,
                "estimate":       round(best.params[param], 4),
                "se":             round(best.bse.get(param, np.nan), 4),
                "z":              round(best.tvalues.get(param, np.nan), 4),
                "p":              round(best.pvalues.get(param, np.nan), 4),
                "sig":            best.pvalues.get(param, 1.0) < 0.05,
            })

        r2_m, r2_c = marginal_r2(best)
        r2_rows.append({
            "representation": rep_name,
            "response":       response,
            "R2_marginal":    r2_m,
            "R2_conditional": r2_c,
        })
        print(f"    R2_marginal={r2_m:.4f}  R2_conditional={r2_c:.4f}")

    return icc_rows, comp_rows, fe_rows, r2_rows


# ══════════════════════════════════════════════════════════════════════════════
# ACOUSTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7.1 Acoustic models")
print("="*60)

all_icc, all_comp, all_fe, all_r2 = [], [], [], []

for response in ["F1_lob", "F2_lob"]:
    icc, comp, fe, r2 = run_model_sequence(
        vowel_df, response, "speaker", f"acoustic_{response}"
    )
    all_icc += icc; all_comp += comp; all_fe += fe; all_r2 += r2


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7.2 Neural models (PC1-PC5)")
print("="*60)

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

for name, pca_path in PCA_FILES.items():
    if not pca_path.exists():
        print(f"  [SKIP] {pca_path.name}")
        continue

    print(f"\n  Loading {name} ...")
    pca_data  = np.load(pca_path, allow_pickle=True)
    keys      = list(pca_data["keys"])
    coords_50 = pca_data["coords_50"]

    rows = []
    for i, k in enumerate(keys):
        m = key_meta.get(k)
        if m is None or m["phoneme"] not in VOWELS:
            continue
        row_dict = {
            "speaker": m["speaker"],
            "phoneme": m["phoneme"],
            "height":  VOWEL_HEIGHT.get(m["phoneme"], "mid"),
            "is_L2":   int(m["L1"] == "ru"),
            "is_male": int(m["gender"] == "m"),
        }
        for pc in range(min(5, coords_50.shape[1])):
            row_dict[f"PC{pc+1}"] = float(coords_50[i, pc])
        rows.append(row_dict)

    neu_df = pd.DataFrame(rows).dropna()
    print(f"  {len(neu_df)} vowel tokens")

    for pc in range(1, 6):
        response = f"PC{pc}"
        if response not in neu_df.columns:
            continue
        icc, comp, fe, r2 = run_model_sequence(
            neu_df, response, "speaker", name
        )
        all_icc += icc; all_comp += comp; all_fe += fe; all_r2 += r2


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Saving ...")
print("="*60)

pd.DataFrame(all_icc).to_csv(RESULTS_DIR / "6c_ICC.csv", index=False)
pd.DataFrame(all_comp).to_csv(RESULTS_DIR / "6c_model_comparison.csv", index=False)
pd.DataFrame(all_fe).to_csv(RESULTS_DIR / "6c_fixed_effects.csv", index=False)
pd.DataFrame(all_r2).to_csv(RESULTS_DIR / "6c_R2.csv", index=False)

print("\n=== ICC summary ===")
icc_df = pd.DataFrame(all_icc)
print(icc_df[["representation", "response", "ICC"]].to_string(index=False))

print("\n=== R2 summary ===")
r2_df = pd.DataFrame(all_r2)
print(r2_df[["representation", "response",
             "R2_marginal", "R2_conditional"]].to_string(index=False))

print("\n=== Significant fixed effects ===")
fe_df = pd.DataFrame(all_fe)
sig_fe = fe_df[fe_df["sig"] & ~fe_df["parameter"].str.contains("Intercept")]
print(sig_fe[["representation", "response",
              "parameter", "estimate", "p"]].to_string(index=False))

print("\nDone. Files saved:")
for p in sorted(RESULTS_DIR.glob("6c_*")):
    print(f"  {p.name}")