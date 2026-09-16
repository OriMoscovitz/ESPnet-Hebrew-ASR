from pathlib import Path

text_file = Path(
    "/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/"
    "hebrew_campus_subset/data/train/text"
)

output_file = Path(
    "/mnt/c/Users/orori/PycharmProjects/ESPnet_Project/"
    "hebrew_campus_subset/tokens.txt"
)

characters = set()

with text_file.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(maxsplit=1)

        if len(parts) != 2:
            continue

        transcript = parts[1]

        for char in transcript:
            if char == " ":
                continue

            characters.add(char)

tokens = [
    "<blank>",
    "<unk>",
    "<space>",
]

tokens.extend(sorted(characters))

tokens.append("<sos/eos>")

with output_file.open("w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token + "\n")

print(f"Created: {output_file}")
print(f"Number of tokens: {len(tokens)}")
print()
print("Tokens:")

for token in tokens:
    print(token)