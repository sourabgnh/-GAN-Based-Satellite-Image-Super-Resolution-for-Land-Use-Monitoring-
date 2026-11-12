"""Audit the EuroSAT train/validation/test folders before model training.

Run from the repository root:
    python -m scripts.check_dataset

Optional:
    python -m scripts.check_dataset --root dataset --expected-size 64
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image


SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_split(root, split):
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {split_dir}")

    files_by_class = {}
    for class_dir in class_dirs:
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        files_by_class[class_dir.name] = files

    return files_by_class


def main():
    parser = argparse.ArgumentParser(description="Check EuroSAT dataset splits.")
    parser.add_argument("--root", default="dataset", help="Dataset root folder.")
    parser.add_argument(
        "--expected-size",
        type=int,
        default=64,
        help="Expected square HR image size in pixels (default: 64).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    split_data = {}
    all_classes = set()
    failed = False

    print(f"Dataset root: {root.resolve()}")

    for split in SPLITS:
        try:
            split_data[split] = collect_split(root, split)
        except (FileNotFoundError, RuntimeError) as error:
            print(f"ERROR: {error}")
            failed = True
            continue

        classes = set(split_data[split])
        all_classes.update(classes)
        total = sum(len(files) for files in split_data[split].values())
        print(f"\n{split.upper()}: {total} images")

        for class_name, files in split_data[split].items():
            print(f"  {class_name}: {len(files)}")

    if failed or len(split_data) != len(SPLITS):
        raise SystemExit(1)

    expected_classes = all_classes
    for split in SPLITS:
        missing = expected_classes - set(split_data[split])
        if missing:
            print(f"ERROR: {split} is missing classes: {sorted(missing)}")
            failed = True

    hashes_by_split = {}
    for split in SPLITS:
        hashes = set()
        for files in split_data[split].values():
            for path in files:
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        if image.size != (args.expected_size, args.expected_size):
                            print(
                                f"WARNING: {path} has size {image.size}; "
                                f"expected {(args.expected_size, args.expected_size)}"
                            )
                except Exception as error:
                    print(f"ERROR: Cannot read {path}: {error}")
                    failed = True
                    continue
                hashes.add(file_hash(path))
        hashes_by_split[split] = hashes

    for first_index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[first_index + 1:]:
            overlap = hashes_by_split[first_split] & hashes_by_split[second_split]
            if overlap:
                print(
                    f"ERROR: {len(overlap)} identical image file(s) occur in "
                    f"both {first_split} and {second_split}."
                )
                failed = True

    if failed:
        print("\nDataset audit finished with errors. Fix them before training.")
        raise SystemExit(1)

    print("\nDataset audit passed: all splits exist, images are readable, "
          "and no identical files were found across splits.")


if __name__ == "__main__":
    main()
