import unittest
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from reproductions.talkshow.scripts.face_batching import (
        DistributedLengthBucketBatchSampler,
        collate_face_whole_video,
    )


def make_sample(audio_length, motion_length, offset, speaker):
    return {
        "aud_feat": (
            np.arange(audio_length, dtype=np.float64).reshape(1, -1) + offset
        ),
        "poses": (
            np.arange(3 * motion_length, dtype=np.float64).reshape(
                3, motion_length
            )
            + offset
        ),
        "expression": (
            np.arange(2 * motion_length, dtype=np.float64).reshape(
                2, motion_length
            )
            + offset
        ),
        "speaker": speaker,
        "betas": np.asarray([offset, offset + 1], dtype=np.float64),
        "aud_file": f"clip-{speaker}.wav",
        "source_path": Path(f"/dataset/clip-{speaker}"),
    }


@unittest.skipUnless(torch is not None, "PyTorch is installed in the face environment")
class FaceWholeVideoCollateTest(unittest.TestCase):
    def test_right_pads_sequences_and_preserves_metadata(self):
        first = make_sample(5, 4, 10, 20)
        second = make_sample(3, 2, 30, 21)

        batch = collate_face_whole_video([first, second])

        self.assertEqual(batch["aud_feat"].shape, (2, 1, 5))
        self.assertEqual(batch["poses"].shape, (2, 3, 4))
        self.assertEqual(batch["expression"].shape, (2, 2, 4))
        self.assertEqual(batch["aud_feat"].dtype, torch.float32)
        self.assertEqual(batch["poses"].dtype, torch.float32)
        self.assertEqual(batch["expression"].dtype, torch.float32)
        torch.testing.assert_close(
            batch["aud_feat"][0, :, :5],
            torch.as_tensor(first["aud_feat"], dtype=torch.float32),
        )
        torch.testing.assert_close(
            batch["aud_feat"][1, :, :3],
            torch.as_tensor(second["aud_feat"], dtype=torch.float32),
        )
        torch.testing.assert_close(
            batch["aud_feat"][1, :, 3:], torch.zeros((1, 2))
        )
        torch.testing.assert_close(
            batch["poses"][1, :, 2:], torch.zeros((3, 2))
        )
        torch.testing.assert_close(
            batch["expression"][1, :, 2:], torch.zeros((2, 2))
        )

        torch.testing.assert_close(
            batch["audio_lengths"], torch.tensor([5, 3])
        )
        torch.testing.assert_close(
            batch["motion_lengths"], torch.tensor([4, 2])
        )
        self.assertEqual(batch["audio_attention_mask"].dtype, torch.long)
        torch.testing.assert_close(
            batch["audio_attention_mask"],
            torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        )
        self.assertEqual(batch["motion_mask"].dtype, torch.bool)
        torch.testing.assert_close(
            batch["motion_mask"],
            torch.tensor(
                [[True, True, True, True], [True, True, False, False]]
            ),
        )

        torch.testing.assert_close(batch["speaker"], torch.tensor([20, 21]))
        self.assertEqual(batch["betas"].dtype, torch.float32)
        self.assertEqual(batch["betas"].shape, (2, 2))
        self.assertEqual(batch["aud_file"], ["clip-20.wav", "clip-21.wav"])
        self.assertEqual(
            batch["source_path"],
            [Path("/dataset/clip-20"), Path("/dataset/clip-21")],
        )

    def test_rejects_motion_length_mismatch(self):
        sample = make_sample(5, 4, 0, 20)
        sample["expression"] = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "motion length mismatch"):
            collate_face_whole_video([sample])

    def test_rejects_inconsistent_channels_and_keys(self):
        first = make_sample(5, 4, 0, 20)
        second = make_sample(5, 4, 0, 21)
        second["poses"] = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "pose channel counts differ"):
            collate_face_whole_video([first, second])

        second = make_sample(5, 4, 0, 21)
        second.pop("aud_file")
        with self.assertRaisesRegex(KeyError, "inconsistent keys"):
            collate_face_whole_video([first, second])

    def test_rejects_empty_batch_and_generated_fields(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_face_whole_video([])

        sample = make_sample(5, 4, 0, 20)
        sample["motion_mask"] = np.ones(4, dtype=bool)
        with self.assertRaisesRegex(KeyError, "collate-generated"):
            collate_face_whole_video([sample])


@unittest.skipUnless(torch is not None, "PyTorch is installed in the face environment")
class DistributedLengthBucketBatchSamplerTest(unittest.TestCase):
    def make_sampler(self, lengths, rank, **kwargs):
        return DistributedLengthBucketBatchSampler(
            lengths,
            batch_size=kwargs.pop("batch_size", 32),
            num_replicas=kwargs.pop("num_replicas", 8),
            rank=rank,
            seed=kwargs.pop("seed", 17),
            drop_last=kwargs.pop("drop_last", False),
            bucket_multiplier=kwargs.pop("bucket_multiplier", 8),
            shuffle=kwargs.pop("shuffle", True),
            **kwargs,
        )

    def test_requested_dataset_shape_has_one_balanced_tail(self):
        sample_count = 13_687
        lengths = [100 + (index * 37) % 2_000 for index in range(sample_count)]
        rank_batches = [
            list(self.make_sampler(lengths, rank)) for rank in range(8)
        ]

        for batches in rank_batches:
            self.assertEqual(len(batches), 54)
            self.assertTrue(all(len(batch) == 32 for batch in batches[:-1]))
            self.assertEqual(len(batches[-1]), 15)
        self.assertEqual(len(self.make_sampler(lengths, 0)), 54)

        flattened = []
        for step in range(54):
            local_sizes = {len(rank_batches[rank][step]) for rank in range(8)}
            self.assertEqual(len(local_sizes), 1)
            for rank in range(8):
                flattened.extend(rank_batches[rank][step])

        counts = Counter(flattened)
        self.assertEqual(len(flattened), 13_688)
        self.assertEqual(len(counts), sample_count)
        self.assertEqual(sum(count - 1 for count in counts.values()), 1)

    def test_is_deterministic_and_set_epoch_changes_order(self):
        lengths = [1 + index % 97 for index in range(1_024)]
        first = self.make_sampler(lengths, 3, num_replicas=4, batch_size=8)
        second = self.make_sampler(lengths, 3, num_replicas=4, batch_size=8)
        self.assertEqual(list(first), list(second))

        first.set_epoch(1)
        second.set_epoch(1)
        epoch_one_first = list(first)
        epoch_one_second = list(second)
        self.assertEqual(epoch_one_first, epoch_one_second)

        third = self.make_sampler(lengths, 3, num_replicas=4, batch_size=8)
        self.assertNotEqual(epoch_one_first, list(third))

    def test_buckets_neighboring_lengths_into_the_same_global_batch(self):
        lengths = list(reversed(range(1, 65)))
        rank_batches = [
            list(
                DistributedLengthBucketBatchSampler(
                    lengths,
                    4,
                    2,
                    rank,
                    seed=0,
                    bucket_multiplier=8,
                    drop_last=False,
                    shuffle=False,
                )
            )
            for rank in range(2)
        ]

        for step in range(8):
            batch_lengths = [
                lengths[index]
                for rank in range(2)
                for index in rank_batches[rank][step]
            ]
            self.assertEqual(max(batch_lengths) - min(batch_lengths), 7)

    def test_drop_last_discards_only_incomplete_global_batch(self):
        lengths = list(range(1, 271))
        rank_batches = [
            list(
                self.make_sampler(
                    lengths,
                    rank,
                    num_replicas=4,
                    batch_size=8,
                    drop_last=True,
                )
            )
            for rank in range(4)
        ]
        for batches in rank_batches:
            self.assertEqual(len(batches), 8)
            self.assertTrue(all(len(batch) == 8 for batch in batches))

        selected = {
            index
            for batches in rank_batches
            for batch in batches
            for index in batch
        }
        self.assertEqual(len(selected), 256)

    def test_short_dataset_pads_only_to_world_size(self):
        lengths = list(range(1, 120))
        rank_batches = [list(self.make_sampler(lengths, rank)) for rank in range(8)]
        self.assertTrue(all(len(batches) == 1 for batches in rank_batches))
        self.assertTrue(all(len(batches[0]) == 15 for batches in rank_batches))

        selected = [
            index
            for batches in rank_batches
            for index in batches[0]
        ]
        self.assertEqual(len(selected), 120)
        self.assertEqual(len(set(selected)), 119)

    def test_validates_constructor_and_epoch(self):
        with self.assertRaisesRegex(ValueError, "num_replicas"):
            self.make_sampler([1], 0, num_replicas=0)
        with self.assertRaisesRegex(ValueError, "rank"):
            self.make_sampler([1], 8)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.make_sampler([1, 0], 0)

        sampler = self.make_sampler([1], 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sampler.set_epoch(-1)


if __name__ == "__main__":
    unittest.main()
