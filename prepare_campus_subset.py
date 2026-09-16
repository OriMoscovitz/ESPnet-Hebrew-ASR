from datasets import load_dataset
from pathlib import Path
import re
import unicodedata
import random
import os

# -----------------------------
# SETTINGS
# -----------------------------

TARGET_HOURS = 5
DEV_RATIO = 0.10
RANDOM_SEED = 42

base_dir = Path("hebrew_campus_subset")
audio_dir = base_dir / "audio"

train_dir = base_dir / "data" / "train"
dev_dir = base_dir / "data" / "dev"

audio_dir.mkdir(parents=True, exist_ok=True)
train_dir.mkdir(parents=True, exist_ok=True)
dev_dir.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)


# -----------------------------
# Hebrew normalization
# -----------------------------

def normalize_hebrew(text):
    text = unicodedata.normalize("NFKC", text)

    # Remove Hebrew niqqud / cantillation
    text = re.sub(r"[\u0591-\u05C7]", "", text)

    # Replace maqaf with space
    text = text.replace("־", " ")

    # Remove geresh / gershayim
    text = text.replace("׳", "")
    text = text.replace("״", "")

    # Remove punctuation
    text = re.sub(r'[.,:;!?()"\'\-]', "", text)

    # Keep only Hebrew letters and whitespace
    text = re.sub(r"[^\u05D0-\u05EA\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Load dataset in streaming mode
# -----------------------------

ds = load_dataset(
    "imvladikon/hebrew_speech_campus",
    split="train",
    streaming=True,
)

# Important: prevents TorchCodec audio decoding
ds = ds.decode(False)


# -----------------------------
# Collect ~5 hours
# -----------------------------

target_ms = TARGET_HOURS * 60 * 60 * 1000

samples = []
total_ms = 0

print(f"Collecting approximately {TARGET_HOURS} hours...")

for item in ds:
    sentence = normalize_hebrew(item["sentence"])

    if not sentence:
        continue

    duration_ms = float(item["duration_ms"])

    # Skip extremely short/long examples
    if duration_ms < 1000:
        continue

    if duration_ms > 30000:
        continue

    audio_info = item["audio"]

    if audio_info is None or audio_info.get("bytes") is None:
        continue

    samples.append({
        "uid": item["uid"],
        "sentence": sentence,
        "duration_ms": duration_ms,
        "audio_bytes": audio_info["bytes"],
    })

    total_ms += duration_ms

    if len(samples) % 100 == 0:
        print(
            f"{len(samples)} samples | "
            f"{total_ms / 1000 / 3600:.2f} hours"
        )

    if total_ms >= target_ms:
        break


print()
print(f"Collected {len(samples)} samples")
print(f"Total duration: {total_ms / 1000 / 3600:.2f} hours")


# -----------------------------
# Shuffle and split
# -----------------------------

random.shuffle(samples)

dev_size = int(len(samples) * DEV_RATIO)

dev_samples = samples[:dev_size]
train_samples = samples[dev_size:]

print(f"Train samples: {len(train_samples)}")
print(f"Dev samples:   {len(dev_samples)}")


# -----------------------------
# Write audio files
# -----------------------------

def save_audio(samples, prefix):
    result = []

    for i, sample in enumerate(samples, start=1):
        utt_id = f"{prefix}{i:06d}"

        wav_path = audio_dir / f"{utt_id}.wav"

        with wav_path.open("wb") as f:
            f.write(sample["audio_bytes"])

        result.append({
            "utt_id": utt_id,
            "wav_path": wav_path.resolve(),
            "sentence": sample["sentence"],
        })

    return result


train_entries = save_audio(train_samples, "TR")
dev_entries = save_audio(dev_samples, "DV")


# -----------------------------
# Create ESPnet files
# -----------------------------

def write_espnet_files(entries, output_dir, speaker_id):

    # text
    with (output_dir / "text").open("w", encoding="utf-8") as f:
        for x in entries:
            f.write(f'{x["utt_id"]} {x["sentence"]}\n')

    # wav.scp
    with (output_dir / "wav.scp").open("w", encoding="utf-8") as f:
        for x in entries:
            f.write(f'{x["utt_id"]} {x["wav_path"]}\n')

    # utt2spk
    with (output_dir / "utt2spk").open("w", encoding="utf-8") as f:
        for x in entries:
            f.write(f'{x["utt_id"]} {speaker_id}\n')

    # spk2utt
    utterances = " ".join(x["utt_id"] for x in entries)

    with (output_dir / "spk2utt").open("w", encoding="utf-8") as f:
        f.write(f"{speaker_id} {utterances}\n")


write_espnet_files(
    train_entries,
    train_dir,
    "campus_train"
)

write_espnet_files(
    dev_entries,
    dev_dir,
    "campus_dev"
)


print()
print("Done.")
print(f"Train data: {train_dir.resolve()}")
print(f"Dev data:   {dev_dir.resolve()}")

# Workaround for the PyArrow/HF shutdown crash seen earlier
os._exit(0)