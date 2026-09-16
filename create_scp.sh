mkdir -p data/train

for f in wav/S{001..020}.wav; do
    id=$(basename "$f" .wav)
    printf '%s %s\n' "$id" "$(realpath "$f")"
done > data/train/wav.scp