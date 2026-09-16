from pathlib import Path
import matplotlib.pyplot as plt

report_dir = Path("/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/Report")
report_dir.mkdir(parents=True, exist_ok=True)

experiments = [
    "ctc",
    "ctc03",
    "ctc05",
    "ctc07",
    "ctc09",
    "ctc08",
    "att",
    "hybrid03_lr5e5_ctc03",
]

cer_values = [
    0.246,
    0.409,
    0.313,
    0.299,
    0.301,
    0.301,
    4.894,
    0.162,
]

wer_values = [
    0.854,
    1.415,
    1.037,
    0.967,
    0.951,
    0.967,
    13.862,
    0.573,
]

cer_no_spaces_values = [
    0.246,
    0.379,
    0.315,
    0.303,
    0.303,
    0.303,
    3.034,
    0.162,
]

x = list(range(1, len(experiments) + 1))

plt.figure(figsize=(10, 5.2))
plt.plot(x, wer_values, marker="o", label="WER")
plt.plot(x, cer_values, marker="o", label="CER")
plt.plot(x, cer_no_spaces_values, marker="o", label="CER without spaces")

plt.xlabel("Experiment")
# plt.ylabel("Error Rate")
plt.ylabel("Error Rate (log scale)")
plt.yscale("log")
plt.title("my_test Experiment Results")
plt.xticks(x, experiments, rotation=45, ha="right")
plt.legend()
plt.tight_layout()

output_path = report_dir / "my_test_results_graph.png"
plt.savefig(output_path, dpi=170)
plt.close()

print(f"Created: {output_path}")