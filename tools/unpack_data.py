"""Decompress the shipped datasets (data/<dataset>/<table>.csv.gz) to <table>.csv.

The loaders read data/<dataset>/<table>.csv. An existing .csv is never overwritten
(for HMDA it may be the full public file, whose first 1M rows equal the shipped
slice). After decompressing, the MD5 of each new .csv is checked against
data/CHECKSUMS.md5 (checksums of the uncompressed files).

Usage: python tools/unpack_data.py [--data-dir data]
"""
import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    checksums = {}
    checksum_file = args.data_dir / "CHECKSUMS.md5"
    if checksum_file.is_file():
        for line in checksum_file.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            checksums[Path(name).name] = digest

    failures = 0
    for archive in sorted(args.data_dir.glob("*/*.csv.gz")):
        target = archive.with_suffix("")
        if target.exists():
            print(f"kept     {target} (already exists, not overwritten)")
            continue
        with gzip.open(archive, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        expected = checksums.get(target.name)
        if expected and md5(target) != expected:
            print(f"FAILED   {target}: MD5 differs from data/CHECKSUMS.md5", file=sys.stderr)
            failures += 1
        else:
            print(f"unpacked {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
