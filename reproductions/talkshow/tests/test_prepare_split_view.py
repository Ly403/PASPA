import pickle
import tempfile
import unittest
from pathlib import Path

from reproductions.talkshow.scripts.prepare_split_view import build_view


class SplitViewTest(unittest.TestCase):
    def test_builds_links_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "view"
            split_file = root / "split.pkl"
            manifest = {"oliver": {"video": {split: [] for split in ("train", "val", "test")}}}

            for split in ("train", "val", "test"):
                clip = f"clip_{split}"
                clip_dir = source / "oliver" / "video" / clip
                clip_dir.mkdir(parents=True)
                (clip_dir / f"{clip}.wav").write_bytes(b"wav")
                (clip_dir / f"{clip}.pkl").write_bytes(b"pkl")
                manifest["oliver"]["video"][split].append(
                    f"../data/oliver/video/{split}/{clip}"
                )

            with split_file.open("wb") as handle:
                pickle.dump(manifest, handle)

            summary = build_view(source, output, split_file, 0, None)
            self.assertEqual(
                summary["counts"], {"train": 1, "val": 1, "test": 1}
            )
            for split in ("train", "val", "test"):
                link = output / "oliver" / "video" / split / f"clip_{split}"
                self.assertTrue(link.is_symlink())
                self.assertTrue((source / "oliver" / "video" / f"clip_{split}").is_dir())

            repeated = build_view(source, output, split_file, 0, None)
            self.assertEqual(repeated["counts"], summary["counts"])

    def test_missing_count_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            split_file = root / "split.pkl"
            manifest = {
                "oliver": {
                    "video": {
                        "train": ["../data/oliver/video/train/missing"],
                        "val": [],
                        "test": [],
                    }
                }
            }
            with split_file.open("wb") as handle:
                pickle.dump(manifest, handle)

            with self.assertRaisesRegex(RuntimeError, "expected 0 missing"):
                build_view(source, root / "view", split_file, 0, None)

    def test_rejects_stale_links_from_a_larger_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "view"
            split_file = root / "split.pkl"
            manifest = {
                "oliver": {
                    "video": {
                        "train": [],
                        "val": [],
                        "test": [],
                    }
                }
            }
            for split in ("train", "val", "test"):
                for index in range(2):
                    clip = f"{split}_{index}"
                    clip_dir = source / "oliver" / "video" / clip
                    clip_dir.mkdir(parents=True)
                    (clip_dir / f"{clip}.wav").write_bytes(b"wav")
                    (clip_dir / f"{clip}.pkl").write_bytes(b"pkl")
                    manifest["oliver"]["video"][split].append(
                        f"oliver/video/{split}/{clip}"
                    )
            with split_file.open("wb") as handle:
                pickle.dump(manifest, handle)

            build_view(source, output, split_file, 0, None)
            with self.assertRaisesRegex(RuntimeError, "outside this requested"):
                build_view(source, output, split_file, 0, 1)


if __name__ == "__main__":
    unittest.main()
