from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import tensorflow as tf


EXP_DIR = Path("/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/exp")
OUTPUT_FILE = Path(
    "/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/tensorboard_scalars.csv"
)

EXPERIMENTS = [
    "hebrew_xlsr_finetune",
    "hebrew_xlsr_finetune_30ep",
    "hebrew_xlsr_finetune_lr5e5",
    "hebrew_xlsr_finetune_lr5e5_ctc03",
]


rows = []


for experiment in EXPERIMENTS:
    tensorboard_dir = EXP_DIR / experiment / "tensorboard"

    print("=" * 70)
    print(f"Experiment: {experiment}")
    print(f"TensorBoard directory: {tensorboard_dir}")

    if not tensorboard_dir.exists():
        print("TensorBoard directory does not exist.")
        continue

    event_files = list(tensorboard_dir.rglob("events.out.tfevents.*"))

    print(f"Found {len(event_files)} event file(s).")

    for event_file in event_files:
        print(f"\nReading: {event_file}")

        ea = EventAccumulator(str(event_file))
        ea.Reload()

        tags = ea.Tags()

        print("Scalar tags:")
        for tag in tags.get("scalars", []):
            print("   ", tag)

            for event in ea.Scalars(tag):
                rows.append([
                    experiment,
                    str(event_file),
                    "scalar",
                    tag,
                    event.step,
                    event.wall_time,
                    event.value,
                ])

        print("Tensor tags:")
        for tag in tags.get("tensors", []):
            print("   ", tag)

            for event in ea.Tensors(tag):
                try:
                    value = tf.make_ndarray(event.tensor_proto)

                    if value.size == 1:
                        value = float(value.reshape(-1)[0])

                        rows.append([
                            experiment,
                            str(event_file),
                            "tensor",
                            tag,
                            event.step,
                            event.wall_time,
                            value,
                        ])

                except Exception as e:
                    print(f"Could not read tensor {tag}: {e}")


with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "experiment",
        "event_file",
        "type",
        "tag",
        "step",
        "wall_time",
        "value",
    ])

    writer.writerows(rows)


print()
print("=" * 70)
print(f"Extracted {len(rows)} values.")
print(f"Saved to: {OUTPUT_FILE}")