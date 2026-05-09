import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
PHONEMES_CSV  = Path(r"C:\Users\aviba\data\phonemes.csv")
WAV_DIR       = Path(r"C:\Users\aviba\Downloads\ru-fr_interference\ru-fr_interference\2\wav_et_textgrids\FRcorp_textgrids_only")
OUTPUT_CSV    = Path(r"C:\Users\aviba\data\features_acoustic.csv")

# LPC formant settings per gender (as specified in the assignment)
MAX_FORMANT   = {"f": 5000, "m": 4500}
N_FORMANTS    = 5

# f0 range (Hz)
F0_MIN        = 75
F0_MAX        = 600

DRY_RUN       = False
DRY_RUN_N     = 100  # number of tokens to process

# ── load phonemes ─────────────────────────────────────────────────────────────
df = pd.read_csv(PHONEMES_CSV)
if DRY_RUN:
    df = df.head(DRY_RUN_N)
    print(f"[DRY RUN] processing first {DRY_RUN_N} tokens")

print(f"Loaded {len(df)} phoneme tokens")

# ── wav file finder ───────────────────────────────────────────────────────────
def find_wav(speaker: str, textgrid_file: str) -> Path | None:
    """
    Find the WAV file corresponding to a TextGrid file.
    e.g. sd_fra_list1_FRcorp4.TextGrid -> SD/sd_fra_list1_FRcorp4.wav
    """
    wav_name = textgrid_file.replace(".TextGrid", ".wav").replace(".textgrid", ".wav")
    candidate = WAV_DIR / speaker / wav_name
    if candidate.exists():
        return candidate

    # fallback: search recursively
    matches = list((WAV_DIR / speaker).glob(f"**/{wav_name}"))
    if matches:
        return matches[0]

    return None

# ── feature extractor ─────────────────────────────────────────────────────────
def extract_features(
    wav_path: Path,
    onset: float,
    offset: float,
    gender: str,
) -> dict:
    """
    Extract F1, F2, F3 at midpoint, mean f0, and duration from a phoneme segment.
    Returns a dict with all features, NaN where extraction fails.
    """
    result = {
        "F1": np.nan, "F2": np.nan, "F3": np.nan,
        "f0_mean": np.nan,
        "F1_25": np.nan, "F2_25": np.nan,  # trajectory points for long vowels
        "F1_75": np.nan, "F2_75": np.nan,
    }

    duration_s = offset - onset
    if duration_s <= 0.02:  # skip very short segments < 20ms
        return result

    try:
        sound = parselmouth.Sound(str(wav_path))
        segment = sound.extract_part(
            from_time=onset,
            to_time=offset,
            preserve_times=True,
        )

        midpoint = onset + duration_s / 2
        max_formant = MAX_FORMANT.get(gender.lower(), 5000)

        # ── formants at midpoint ──────────────────────────────────────────────
        formant = call(segment, "To Formant (burg)",
                       0,            # time step (auto)
                       N_FORMANTS,
                       max_formant,
                       0.025,        # window length (s)
                       50)           # pre-emphasis

        f1 = call(formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, midpoint, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, midpoint, "Hertz", "Linear")

        result["F1"] = f1 if f1 != 0 else np.nan
        result["F2"] = f2 if f2 != 0 else np.nan
        result["F3"] = f3 if f3 != 0 else np.nan

        # ── trajectory for long vowels (> 80ms) ──────────────────────────────
        if duration_s * 1000 > 80:
            t25 = onset + duration_s * 0.25
            t75 = onset + duration_s * 0.75

            f1_25 = call(formant, "Get value at time", 1, t25, "Hertz", "Linear")
            f2_25 = call(formant, "Get value at time", 2, t25, "Hertz", "Linear")
            f1_75 = call(formant, "Get value at time", 1, t75, "Hertz", "Linear")
            f2_75 = call(formant, "Get value at time", 2, t75, "Hertz", "Linear")

            result["F1_25"] = f1_25 if f1_25 != 0 else np.nan
            result["F2_25"] = f2_25 if f2_25 != 0 else np.nan
            result["F1_75"] = f1_75 if f1_75 != 0 else np.nan
            result["F2_75"] = f2_75 if f2_75 != 0 else np.nan

        # ── f0 ────────────────────────────────────────────────────────────────
        pitch = call(segment, "To Pitch", 0, F0_MIN, F0_MAX)
        f0 = call(pitch, "Get mean", onset, offset, "Hertz")
        result["f0_mean"] = f0 if (f0 and f0 == f0) else np.nan  # nan check

    except Exception as e:
        pass  # return nans on any extraction failure

    return result


# # ── main loop ─────────────────────────────────────────────────────────────────
# records = []
# wav_cache = {}  # cache open sounds to avoid reloading same file repeatedly
# errors = 0

# for i, row in df.iterrows():
#     spk    = row["speaker"]
#     tg     = row["textgrid_file"]
#     onset  = row["onset"]
#     offset = row["offset"]
#     gender = str(row["gender"])

#     wav_key = (spk, tg)
#     if wav_key not in wav_cache:
#         wav_cache[wav_key] = find_wav(spk, tg)

#     wav_path = wav_cache[wav_key]

#     if wav_path is None:
#         errors += 1
#         feats = {"F1": np.nan, "F2": np.nan, "F3": np.nan,
#                  "f0_mean": np.nan,
#                  "F1_25": np.nan, "F2_25": np.nan,
#                  "F1_75": np.nan, "F2_75": np.nan}
#     else:
#         feats = extract_features(wav_path, onset, offset, gender)

#     records.append({**row.to_dict(), **feats})

#     if (i + 1) % 500 == 0 or i == 0:
#         print(f"  [{i+1}/{len(df)}] {spk} {row['phoneme']} "
#               f"F1={feats['F1']:.0f} F2={feats['F2']:.0f}"
#               if not np.isnan(feats.get("F1", np.nan))
#               else f"  [{i+1}/{len(df)}] {spk} {row['phoneme']} F1=nan")

# # ── save ──────────────────────────────────────────────────────────────────────
# out_df = pd.DataFrame(records)
# out_df.to_csv(OUTPUT_CSV, index=False)
# print(f"\nSaved {len(out_df)} rows -> {OUTPUT_CSV}")

# # ── missing value report ──────────────────────────────────────────────────────
# print(f"\nMissing value rates:")
# for col in ["F1", "F2", "F3", "f0_mean"]:
#     pct = out_df[col].isna().mean() * 100
#     print(f"  {col}: {pct:.1f}% missing")

# print(f"\nMissing by phoneme (F1):")
# missing_by_phoneme = out_df.groupby("phoneme")["F1"].apply(
#     lambda x: f"{x.isna().mean()*100:.0f}%"
# )
# print(missing_by_phoneme.to_string())

# if errors > 0:
#     print(f"\n[WARN] {errors} tokens had no matching WAV file")

import pandas as pd
df = pd.read_csv(r"C:\Users\aviba\data\features_acoustic.csv")
counts = df['phoneme'].value_counts()
print(counts[counts < 10])  # see what's rare
df = df[df['phoneme'].isin(counts[counts >= 10].index)]
print(f"After filtering: {len(df)} tokens, {df['phoneme'].nunique()} phoneme types")
df.to_csv(r"C:\Users\aviba\data\features_acoustic.csv", index=False)