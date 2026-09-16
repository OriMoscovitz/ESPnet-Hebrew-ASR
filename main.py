# from espnet2.bin.asr_inference import Speech2Text
#
# speech2text = Speech2Text(
#     asr_train_config="/home/gecko/espnet/egs2/librispeech/asr1/exp/asr_train_asr_ctc_small_raw_en_bpe5000_sp/config.yaml",
#     asr_model_file="/home/gecko/espnet/egs2/librispeech/asr1/exp/asr_train_asr_ctc_small_raw_en_bpe5000_sp/valid.acc.ave_10best.pth",
#     device="cuda",
# )



# ### rename files
# from pathlib import Path
#
# # Folder containing the recordings
# folder = Path("C:\Charles Masters\Second Year\Summer\Algorithms in Speech Recognition\Projcet\heb_recordings")
#
# for i in range(3, 26):
#     old_file = folder / f"Recording ({i}).m4a"
#     new_file = folder / f"S{i - 2:03d}.m4a"
#
#     if old_file.exists():
#         old_file.rename(new_file)
#         print(f"{old_file.name} -> {new_file.name}")
#     else:
#         print(f"Missing: {old_file.name}")


### normalize text
# import re
# import unicodedata
# from pathlib import Path
#
# input_file = Path("C:\Charles Masters\Second Year\Summer\Algorithms in Speech Recognition\Projcet\heb_sentences.txt")
# output_file = Path("heb_sentences_normalized.txt")
#
#
# def normalize_hebrew(text):
#     # Normalize Unicode representation
#     text = unicodedata.normalize("NFKC", text)
#
#     # Remove Hebrew niqqud / cantillation marks
#     text = re.sub(r"[\u0591-\u05C7]", "", text)
#
#     # Replace Hebrew maqaf with a normal space
#     text = text.replace("־", " ")
#
#     # Remove Hebrew geresh / gershayim
#     text = text.replace("׳", "")
#     text = text.replace("״", "")
#
#     # Remove regular punctuation
#     text = re.sub(r'[.,:;!?()"\'\-]', "", text)
#
#     # Keep only Hebrew letters and whitespace
#     text = re.sub(r"[^\u05D0-\u05EA\s]", "", text)
#
#     # Collapse repeated spaces
#     text = re.sub(r"\s+", " ", text).strip()
#
#     return text
#
#
# with input_file.open("r", encoding="utf-8") as f:
#     lines = f.readlines()
#
# normalized_lines = [normalize_hebrew(line) for line in lines]
#
# with output_file.open("w", encoding="utf-8") as f:
#     for line in normalized_lines:
#         if line:
#             f.write(line + "\n")
#
# print(f"Created: {output_file}")
# print(f"Number of sentences: {len(normalized_lines)}")


from pathlib import Path

# ----------------------------
# CHANGE THIS PATH
# ----------------------------
audio_folder = Path("C:\Charles Masters\Second Year\Summer\Algorithms in Speech Recognition\Projcet\heb_recordings").resolve()

# Normalized transcript file
transcript_file = Path("heb_sentences_normalized.txt")

# ESPnet output directory
output_dir = Path("data/my_test")
output_dir.mkdir(parents=True, exist_ok=True)

speaker_id = "spk1"

# ----------------------------
# Read transcripts
# ----------------------------
with transcript_file.open("r", encoding="utf-8") as f:
    transcripts = [line.strip() for line in f if line.strip()]

if len(transcripts) != 23:
    raise ValueError(
        f"Expected 23 transcripts, but found {len(transcripts)}"
    )

# ----------------------------
# Check audio files
# ----------------------------
audio_files = []

for i in range(1, 24):
    utt_id = f"S{i:03d}"
    audio_file = audio_folder / f"{utt_id}.m4a"

    if not audio_file.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_file}")

    audio_files.append((utt_id, audio_file))

# ----------------------------
# Create text
# ----------------------------
with (output_dir / "text").open("w", encoding="utf-8") as f:
    for (utt_id, _), transcript in zip(audio_files, transcripts):
        f.write(f"{utt_id} {transcript}\n")

# ----------------------------
# Create wav.scp
# ----------------------------
# ffmpeg converts M4A -> mono 16 kHz WAV on the fly.
with (output_dir / "wav.scp").open("w", encoding="utf-8") as f:
    for utt_id, audio_file in audio_files:
        f.write(
            f"{utt_id} ffmpeg -loglevel error "
            f'-i "{audio_file}" '
            f"-ac 1 -ar 16000 -f wav - |\n"
        )

# ----------------------------
# Create utt2spk
# ----------------------------
with (output_dir / "utt2spk").open("w", encoding="utf-8") as f:
    for utt_id, _ in audio_files:
        f.write(f"{utt_id} {speaker_id}\n")

# ----------------------------
# Create spk2utt
# ----------------------------
utterance_ids = " ".join(utt_id for utt_id, _ in audio_files)

with (output_dir / "spk2utt").open("w", encoding="utf-8") as f:
    f.write(f"{speaker_id} {utterance_ids}\n")

print("Created ESPnet test data:")
print(output_dir.resolve())

for filename in ["text", "wav.scp", "utt2spk", "spk2utt"]:
    print(f"  {output_dir / filename}")