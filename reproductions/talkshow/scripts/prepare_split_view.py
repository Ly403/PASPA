#!/usr/bin/env python3
"""Build a non-destructive symlink view of the official TalkSHOW split."""

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional


SPLITS = ("train", "val", "test")
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"split manifest attempted to import {module}.{name}"
        )


def _resolve_source(
    source_root: Path, speaker: str, video: str, split: str, clip: str
) -> Path:
    flat = source_root / speaker / video / clip
    already_split = source_root / speaker / video / split / clip
    if flat.is_dir():
        return flat
    if already_split.is_dir():
        return already_split
    return flat


def build_view(
    source_root: Path,
    output_root: Path,
    split_file: Path,
    expected_missing: int,
    limit_per_speaker_split: Optional[int],
    split_sha256: Optional[str] = None,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    split_file = split_file.resolve()

    if not source_root.is_dir():
        raise RuntimeError(f"source root is not a directory: {source_root}")
    if source_root == output_root or source_root in output_root.parents:
        raise RuntimeError("output root must be outside the immutable source root")
    if limit_per_speaker_split is not None and limit_per_speaker_split < 1:
        raise RuntimeError("limit must be positive")

    split_bytes = split_file.read_bytes()
    observed_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 is not None and observed_sha256 != split_sha256:
        raise RuntimeError(
            f"split file SHA-256 is {observed_sha256}, expected {split_sha256}"
        )
    with split_file.open("rb") as handle:
        manifest = _DataOnlyUnpickler(handle).load()

    output_root.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    missing: list[str] = []
    selected: Counter[tuple[str, str]] = Counter()
    expected_destinations: set[Path] = set()

    for speaker in sorted(manifest):
        for video in sorted(manifest[speaker]):
            split_map = manifest[speaker][video]
            for split in SPLITS:
                entries = split_map.get(split, [])
                for entry in entries:
                    key = (speaker, split)
                    if (
                        limit_per_speaker_split is not None
                        and selected[key] >= limit_per_speaker_split
                    ):
                        continue
                    clip = Path(str(entry).replace("\\", "/")).name
                    source = _resolve_source(
                        source_root, speaker, video, split, clip
                    )
                    wav = source / f"{clip}.wav"
                    motion = source / f"{clip}.pkl"
                    if not source.is_dir() or not wav.is_file() or not motion.is_file():
                        missing.append(str(source))
                        continue

                    destination = output_root / speaker / video / split / clip
                    expected_destinations.add(destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.is_symlink():
                        if destination.resolve() != source.resolve():
                            raise RuntimeError(
                                f"existing link points elsewhere: {destination}"
                            )
                    elif destination.exists():
                        raise RuntimeError(
                            f"refusing to replace existing path: {destination}"
                        )
                    else:
                        os.symlink(source, destination, target_is_directory=True)

                    counts[split] += 1
                    selected[key] += 1

    actual_destinations: set[Path] = set()
    for split in SPLITS:
        for destination in output_root.glob(f"*/*/{split}/*"):
            if not destination.is_symlink():
                raise RuntimeError(
                    f"split view contains a non-symlink clip path: {destination}"
                )
            actual_destinations.add(destination)
    extras = sorted(actual_destinations - expected_destinations)
    if extras:
        raise RuntimeError(
            "split view contains links outside this requested selection; "
            f"first extra: {extras[0]}"
        )

    if limit_per_speaker_split is None and len(missing) != expected_missing:
        raise RuntimeError(
            f"expected {expected_missing} missing clips, observed {len(missing)}"
        )

    if limit_per_speaker_split is not None:
        for speaker in sorted(manifest):
            for split in SPLITS:
                if selected[(speaker, split)] != limit_per_speaker_split:
                    raise RuntimeError(
                        f"could not select {limit_per_speaker_split} valid "
                        f"{split} clips for {speaker}"
                    )

    summary = {
        "source_root": str(source_root),
        "split_file": str(split_file),
        "split_sha256": observed_sha256,
        "counts": {split: counts[split] for split in SPLITS},
        "missing_count": len(missing),
        "missing": missing,
        "limit_per_speaker_split": limit_per_speaker_split,
    }
    summary_path = output_root / "split_view_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--expected-missing", type=int, default=0)
    parser.add_argument("--limit-per-speaker-split", type=int)
    parser.add_argument(
        "--split-sha256",
        default=OFFICIAL_SPLIT_SHA256,
        help="expected split manifest SHA-256",
    )
    args = parser.parse_args()

    summary = build_view(
        source_root=args.source_root,
        output_root=args.output_root,
        split_file=args.split_file,
        expected_missing=args.expected_missing,
        limit_per_speaker_split=args.limit_per_speaker_split,
        split_sha256=args.split_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
