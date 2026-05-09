"""
Stage 1 — parse_corpus (phoneme level)
========================================
Reads all TextGrid files, extracts the phones tier, joins with speaker
metadata, and outputs one row per phoneme token.

Output: data/phonemes.csv
Columns: speaker, L1, gender, age, FR_level, RU_level,
         textgrid_file, word, phoneme, onset, offset, duration_ms,
         sentence_id, repetition
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
CORPUS_ROOT  = Path(r"C:\Users\aviba\Downloads\ru-fr_interference\ru-fr_interference")
WAV_DIR      = CORPUS_ROOT / "2" / "wav_et_textgrids" / "FRcorp_textgrids_only"
META_CSV     = CORPUS_ROOT / "2" / "metadata_RUFR.csv"
OUTPUT_DIR   = Path(r"C:\Users\aviba\data")
OUTPUT_DIR.mkdir(exist_ok=True)
DRY_RUN = False
DRY_RUN_SPEAKERS = 2  # number of speakers to process

# Labels to skip — silence, noise markers, aligner artifacts
SKIP_LABELS  = {"", "sp", "SIL", "sil", "<unk>"}
# Any label containing these strings is also skipped
SKIP_PATTERNS = ["ding", "spn", "ns", "<"]
MIN_TOKEN_COUNT = 5  # drop any phoneme type with fewer than this many tokens globally

# ── load speaker metadata ─────────────────────────────────────────────────────
def load_speaker_meta(path: Path) -> dict[str, dict]:
    speakers = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            spk_id = row[1].strip().upper()
            speakers[spk_id] = {
                "L1":       row[2].strip(),
                "age":      row[3].strip(),
                "gender":   row[4].strip(),
                "FR_level": row[5].strip(),
                "RU_level": row[6].strip(),
            }
    print(f"  Loaded metadata for {len(speakers)} speakers")
    return speakers

# ── TextGrid parser ───────────────────────────────────────────────────────────
def parse_textgrid(path: Path) -> dict[str, list[dict]]:
    """
    Parse a Praat TextGrid file and return a dict of tier_name -> list of intervals.
    Each interval: {xmin, xmax, text}
    """
    tiers = {}
    current_tier = None
    current_intervals = []
    current_name = None
    in_item = False

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("name ="):
            if current_tier is not None and current_name:
                tiers[current_name] = current_intervals
            current_name = line.split("=", 1)[1].strip().strip('"')
            current_intervals = []
            current_tier = current_name

        elif line.startswith("xmin =") and current_tier:
            xmin = float(line.split("=", 1)[1].strip())
            # peek ahead for xmax and text
            if i + 2 < len(lines):
                xmax_line = lines[i + 1].strip()
                text_line  = lines[i + 2].strip()
                if xmax_line.startswith("xmax =") and text_line.startswith("text ="):
                    xmax = float(xmax_line.split("=", 1)[1].strip())
                    text = text_line.split("=", 1)[1].strip().strip('"')
                    current_intervals.append({
                        "xmin": xmin,
                        "xmax": xmax,
                        "text": text,
                    })
        i += 1

    if current_name and current_intervals:
        tiers[current_name] = current_intervals

    return tiers


# ── filename parser ───────────────────────────────────────────────────────────
def parse_filename(path: Path) -> dict | None:
    """
    Extract sentence_id and repetition from filename.
    e.g. sd_fra_list1_FRcorp4.TextGrid -> sentence_id=FRcorp4, rep=4
    """
    name = path.stem  # e.g. sd_fra_list1_FRcorp4
    m = re.search(r"FRcorp(\d+)", name, re.IGNORECASE)
    if m:
        return {
            "sentence_id": f"FRcorp{m.group(1)}",
            "repetition":  int(m.group(1)),
        }
    return {"sentence_id": name, "repetition": 0}


# ── label cleaner ─────────────────────────────────────────────────────────────
def is_valid_phoneme(label: str) -> bool:
    if label.strip() in SKIP_LABELS:
        return False
    for pat in SKIP_PATTERNS:
        if pat.lower() in label.lower():
            return False
    return True


# ── assign word to phoneme via overlap ───────────────────────────────────────
def assign_words(phone_intervals: list[dict], word_intervals: list[dict]) -> list[str]:
    """
    For each phone interval, find the word interval it falls within.
    Uses midpoint of phone to find containing word.
    """
    words = []
    for phone in phone_intervals:
        mid = (phone["xmin"] + phone["xmax"]) / 2
        found = ""
        for word in word_intervals:
            if word["xmin"] <= mid <= word["xmax"] and word["text"].strip():
                found = word["text"].strip()
                break
        words.append(found)
    return words


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    speaker_meta = load_speaker_meta(META_CSV)

    all_rows = []
    skipped_labels = defaultdict(int)
    missing_meta   = set()

    speaker_dirs = sorted([d for d in WAV_DIR.iterdir() if d.is_dir()])
    if DRY_RUN:
        speaker_dirs = speaker_dirs[:DRY_RUN_SPEAKERS]
        print(f"[DRY RUN] processing first {DRY_RUN_SPEAKERS} speakers only")

    for spk_dir in speaker_dirs:
        spk_id = spk_dir.name.upper()
        meta   = speaker_meta.get(spk_id)

        if meta is None:
            missing_meta.add(spk_id)
            meta = {"L1": "?", "age": "?", "gender": "?",
                    "FR_level": "?", "RU_level": "?"}

        textgrids = sorted(spk_dir.glob("*.TextGrid"))
        if not textgrids:
            textgrids = sorted(spk_dir.glob("*.textgrid"))

        for tg_path in textgrids:
            file_info = parse_filename(tg_path)

            try:
                tiers = parse_textgrid(tg_path)
            except Exception as e:
                print(f"  [ERROR] {tg_path.name}: {e}")
                continue

            # get phones tier — try several common names
            phones = None
            for name in ["phones", "phone", "Phone", "Phones", "segments"]:
                if name in tiers:
                    phones = tiers[name]
                    break
            if phones is None:
                print(f"  [WARN] no phones tier in {tg_path.name} "
                      f"(tiers: {list(tiers.keys())})")
                continue

            # get words tier for word assignment
            words_tier = tiers.get("words") or tiers.get("word") or []

            word_labels = assign_words(phones, words_tier)

            for phone, word_label in zip(phones, word_labels):
                label = phone["text"].strip()

                if not is_valid_phoneme(label):
                    skipped_labels[label] += 1
                    continue

                onset    = phone["xmin"]
                offset   = phone["xmax"]
                duration = round((offset - onset) * 1000, 2)  # ms

                all_rows.append({
                    "speaker":       spk_id,
                    "L1":            meta["L1"],
                    "age":           meta["age"],
                    "gender":        meta["gender"],
                    "FR_level":      meta["FR_level"],
                    "RU_level":      meta["RU_level"],
                    "textgrid_file": tg_path.name,
                    "sentence_id":   file_info["sentence_id"],
                    "repetition":    file_info["repetition"],
                    "word":          word_label,
                    "phoneme":       label,
                    "onset":         round(onset, 6),
                    "offset":        round(offset, 6),
                    "duration_ms":   duration,
                })

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "phonemes.csv"
    if all_rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} phoneme tokens -> {out_path}")
    else:
        print("[ERROR] No rows extracted — check TextGrid parsing")
        return

    # filter rare phoneme types
    import pandas as pd
    df = pd.read_csv(out_path)
    counts = df['phoneme'].value_counts()
    valid_phonemes = counts[counts >= MIN_TOKEN_COUNT].index
    df_filtered = df[df['phoneme'].isin(valid_phonemes)]

    print(f"Dropped {len(df) - len(df_filtered)} tokens from {len(counts) - len(valid_phonemes)} rare phoneme types")
    print(f"Remaining: {len(df_filtered)} tokens, {df_filtered['phoneme'].nunique()} phoneme types")

    df_filtered.to_csv(out_path, index=False)
    # ── diagnostics ───────────────────────────────────────────────────────────
    print(f"\nSkipped labels (top 20):")
    for label, count in sorted(skipped_labels.items(),
                                key=lambda x: -x[1])[:20]:
        print(f"  '{label}': {count}")

    if missing_meta:
        print(f"\n[WARN] No metadata found for: {missing_meta}")

    # quick summary
    import pandas as pd
    df = pd.read_csv(out_path)
    print(f"\nSummary:")
    print(f"  Speakers     : {df['speaker'].nunique()}")
    print(f"  Phonemes     : {df['phoneme'].nunique()}")
    print(f"  Total tokens : {len(df)}")
    print(f"  L1 breakdown : {df.groupby('L1')['speaker'].nunique().to_dict()}")
    print(f"\nTop 20 phoneme types:")
    print(df['phoneme'].value_counts().head(20))


if __name__ == "__main__":
    main()

# import pandas as pd
# df = pd.read_csv(r"C:\Users\aviba\data\phonemes.csv")

# # check word assignment worked
# print(df[["phoneme", "word", "onset", "offset"]].head(20))

# # check duration range looks sensible (ms)
# print(df["duration_ms"].describe())