import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from transformers import WhisperModel, WhisperFeatureExtractor
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
ACOUSTIC_CSV  = Path(r"C:\Users\aviba\data\features_acoustic.csv")
WAV_DIR       = Path(r"C:\Users\aviba\Downloads\ru-fr_interference\ru-fr_interference\2\wav_et_textgrids\FRcorp_textgrids_only")
OUTPUT_DIR    = Path(r"C:\Users\aviba\data")

MODEL_NAME         = "openai/whisper-base"
LAYERS             = [2, 5]
TARGET_SR          = 16_000
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_FRAME_RATE = 50.0

DRY_RUN            = False
DRY_RUN_N_WAVS     = 3  # number of wav files to process in dry run

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(ACOUSTIC_CSV)
print(f"Loaded {len(df)} phoneme tokens from {df['textgrid_file'].nunique()} wav files")

# ── load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} on {DEVICE} ...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
model = WhisperModel.from_pretrained(MODEL_NAME)
model.eval().to(DEVICE)
print(f"  Encoder layers : {model.config.encoder_layers}")
print(f"  Hidden size    : {model.config.d_model}")

# ── wav finder ────────────────────────────────────────────────────────────────
def find_wav(speaker: str, textgrid_file: str) -> Path | None:
    wav_name = textgrid_file.replace(".TextGrid", ".wav").replace(".textgrid", ".wav")
    candidate = WAV_DIR / speaker / wav_name
    if candidate.exists():
        return candidate
    matches = list((WAV_DIR / speaker).glob(f"**/{wav_name}"))
    return matches[0] if matches else None

# ── audio loader ──────────────────────────────────────────────────────────────
def load_audio(speaker: str, textgrid_file: str) -> tuple[np.ndarray, int] | None:
    wav_path = find_wav(speaker, textgrid_file)
    if wav_path is None:
        return None
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio_tensor = torch.tensor(audio).float().unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, TARGET_SR)
        audio = audio_tensor.squeeze().numpy()
    return audio.astype(np.float32), TARGET_SR

# ── frame overlap helper ──────────────────────────────────────────────────────
def get_frame_indices(onset: float, offset: float, n_frames: int) -> list[int]:
    start_frame = max(0, int(onset * WHISPER_FRAME_RATE))
    end_frame   = min(n_frames, int(np.ceil(offset * WHISPER_FRAME_RATE)))
    if end_frame <= start_frame:
        end_frame = min(n_frames, start_frame + 1)
    return list(range(start_frame, end_frame))

# ── main loop — one forward pass per wav ─────────────────────────────────────
reps     = {layer: {} for layer in LAYERS}
log_rows = []
errors   = 0

grouped   = list(df.groupby(["speaker", "textgrid_file"]))
total_wavs = len(grouped)

if DRY_RUN:
    grouped = grouped[:DRY_RUN_N_WAVS]
    print(f"\n[DRY RUN] processing first {DRY_RUN_N_WAVS} wav files")

print(f"\nProcessing {len(grouped)} wav files ...")
t_total = time.perf_counter()

for wav_idx, ((spk, tg), group) in enumerate(grouped):

    audio_data = load_audio(spk, tg)
    if audio_data is None:
        errors += len(group)
        print(f"  [WARN] wav not found: {spk}/{tg}")
        continue

    audio, sr = audio_data

    # pad or truncate to 30s for Whisper
    max_samples  = TARGET_SR * 30
    audio_padded = audio[:max_samples]
    if len(audio_padded) < max_samples:
        audio_padded = np.pad(audio_padded, (0, max_samples - len(audio_padded)))

    t0 = time.perf_counter()
    try:
        inputs = feature_extractor(
            audio_padded,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model.encoder(
                inputs.input_features.to(DEVICE),
                output_hidden_states=True,
                return_dict=True,
            )

        n_frames  = outputs.hidden_states[1].shape[1]
        elapsed   = time.perf_counter() - t0
        n_phonemes = len(group)

        # extract all phoneme tokens from this wav
        for _, row in group.iterrows():
            key = (f"{spk}__{row['phoneme']}__{row['sentence_id']}"
                   f"__{row['onset']:.4f}")
            frame_idx = get_frame_indices(row["onset"], row["offset"], n_frames)

            for layer in LAYERS:
                hidden = outputs.hidden_states[layer + 1]  # (1, T, D)
                pooled = hidden[0, frame_idx, :].mean(dim=0)
                reps[layer][key] = pooled.cpu().numpy().astype(np.float32)

            log_rows.append({
                "key":      key,
                "speaker":  spk,
                "phoneme":  row["phoneme"],
                "onset":    row["onset"],
                "offset":   row["offset"],
                "n_frames": len(frame_idx),
                "elapsed":  round(elapsed / n_phonemes, 4),  # amortised
                "status":   "ok",
            })

        print(f"  [{wav_idx+1}/{len(grouped)}] {spk} {tg} "
              f"| {n_phonemes} phonemes | {elapsed:.2f}s "
              f"| {elapsed/n_phonemes*1000:.0f}ms/phoneme")

    except Exception as e:
        errors += len(group)
        print(f"  [ERROR] {spk}/{tg}: {e}")

# ── save ──────────────────────────────────────────────────────────────────────
for layer in LAYERS:
    out_path = OUTPUT_DIR / f"features_whisper_L{layer}.npz"
    np.savez_compressed(out_path, **reps[layer])
    size_mb  = out_path.stat().st_size / 1e6
    print(f"\nSaved layer {layer} -> {out_path} "
          f"({size_mb:.1f} MB, {len(reps[layer])} vectors)")

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUTPUT_DIR / "features_whisper_log.csv", index=False)

total_elapsed = time.perf_counter() - t_total
print(f"\nDone. Total time: {total_elapsed:.1f}s")
print(f"Errors: {errors}")

if len(log_rows) > 0:
    avg_per_wav = total_elapsed / len(grouped)
    print(f"Mean time per wav: {avg_per_wav:.2f}s")
    print(f"Estimated full corpus time: "
          f"{avg_per_wav * total_wavs / 60:.1f} min")