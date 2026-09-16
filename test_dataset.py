from datasets import load_dataset
import os

ds = load_dataset(
    "imvladikon/hebrew_speech_campus",
    split="train",
    streaming=True,
)

ds = ds.decode(False)

sample = next(iter(ds))

print("UID:", sample["uid"])
print("Sentence:", sample["sentence"])
print("Duration:", sample["duration_ms"])
print("Sample rate:", sample["sample_rate"])
print("Language:", sample["language"])

os._exit(0)