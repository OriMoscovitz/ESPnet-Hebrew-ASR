from pathlib import Path
from jiwer import wer, cer
from datetime import datetime

exp_dir = Path("/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/exp")

print_utt = False

reference_file = Path("/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/data/my_test/text")
# reference_file = Path("/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/hebrew_campus_subset/data/dev/text")


def read_text_file(path):
    data = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                utt_id, text = parts
                data[utt_id] = text

    return data


# Find all decode folders that contain a result text file
folders = [
    p for p in exp_dir.glob("decode*")
    if p.is_dir() and (p / "1best_recog" / "text").exists()
]

# Sort by the time the result file was created/modified
folders.sort(
    key=lambda p: (p / "1best_recog" / "text").stat().st_mtime
)

references = read_text_file(reference_file)

for experiment_number, folder in enumerate(folders, start=1):
    hypothesis_file = folder / "1best_recog" / "text"

    run_time = datetime.fromtimestamp(
        hypothesis_file.stat().st_mtime
    )

    print(f"Experiment #{experiment_number}")
    print(f"Reading {folder.name}:")
    print(f"Run time: {run_time:%Y-%m-%d %H:%M:%S}")

    hypotheses = read_text_file(hypothesis_file)

    ref_sentences = []
    hyp_sentences = []

    if print_utt:
        print("Per-utterance results:")
        print()

    for utt_id in sorted(references):
        ref = references[utt_id]
        hyp = hypotheses.get(utt_id, "")

        ref_sentences.append(ref)
        hyp_sentences.append(hyp)

        if print_utt:
            print(utt_id)
            print("REF:", ref)
            print("HYP:", hyp)
            print(f"CER: {cer(ref, hyp):.3f}")
            print(f"WER: {wer(ref, hyp):.3f}")
            print()

    print(f"Overall CER: {cer(ref_sentences, hyp_sentences):.3f}")
    print(f"Overall WER: {wer(ref_sentences, hyp_sentences):.3f}")

    ref_no_spaces = [text.replace(" ", "") for text in ref_sentences]
    hyp_no_spaces = [text.replace(" ", "") for text in hyp_sentences]

    print(f"Overall CER without spaces: {cer(ref_no_spaces, hyp_no_spaces):.3f}")

    print("=" * 60)