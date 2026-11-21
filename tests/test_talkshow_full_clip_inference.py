from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from diffusion.pipeline.emage_inference import CausalEmageInferencePipeline


class _Processor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        waveforms,
        *,
        sampling_rate,
        padding,
        return_tensors,
    ):
        del sampling_rate, padding, return_tensors
        self.calls += 1
        values = torch.as_tensor(waveforms, dtype=torch.float32)
        return SimpleNamespace(
            input_values=values,
            attention_mask=torch.ones_like(values, dtype=torch.long),
        )


class _WavLM:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, values, attention_mask):
        del attention_mask
        self.calls += 1
        # A small deterministic hidden sequence is enough to exercise the
        # full-clip interpolation and subsequent block slicing.
        hidden = values[:, ::4].unsqueeze(-1).repeat(1, 1, 8)
        return SimpleNamespace(last_hidden_state=hidden)


class _SpeechModel:
    def __init__(self) -> None:
        self.calls = 0
        self.input_lengths = []

    def __call__(self, features, *, pid):
        del pid
        self.calls += 1
        self.input_lengths.append(features.shape[1])
        return features[..., :1].repeat(1, 1, 100)


def _pipeline(scope: str) -> CausalEmageInferencePipeline:
    pipeline = CausalEmageInferencePipeline.__new__(
        CausalEmageInferencePipeline
    )
    processor = _Processor()
    wavlm = _WavLM()
    pipeline.config = SimpleNamespace(
        data=SimpleNamespace(audio_sr=16000),
        audio=SimpleNamespace(
            audio_encoder={"name": "wavlm", "quantization": False}
        ),
        train=SimpleNamespace(task="audio2motion"),
    )
    pipeline.context_encoder = [processor, wavlm]
    pipeline.speech_model = _SpeechModel()
    pipeline.device = torch.device("cpu")
    pipeline.pose_fps = 30
    pipeline.audio_feature_scope = scope
    return pipeline


class TalkSHOWFullClipInferenceTest(unittest.TestCase):
    def test_full_clip_scope_encodes_once_then_slices_features(self) -> None:
        pipeline = _pipeline("full_clip")
        waveform = torch.linspace(-1.0, 1.0, 3200).unsqueeze(0)

        prepared = pipeline._prepare_audio_context(
            waveform,
            source_frames=6,
            target_frames=8,
        )
        first = pipeline.context_encoding(prepared, 0, 4, 4)
        second = pipeline.context_encoding(prepared, 4, 8, 4)

        self.assertEqual(prepared.shape, (1, 8, 8))
        self.assertEqual(first.shape, (1, 4, 8))
        self.assertEqual(second.shape, (1, 4, 8))
        self.assertEqual(pipeline.context_encoder[0].calls, 1)
        self.assertEqual(pipeline.context_encoder[1].calls, 1)
        torch.testing.assert_close(first, prepared[:, :4])
        torch.testing.assert_close(second, prepared[:, 4:8])
        torch.testing.assert_close(
            prepared[:, 6:],
            torch.zeros_like(prepared[:, 6:]),
        )

    def test_block_scope_preserves_raw_waveform_path(self) -> None:
        pipeline = _pipeline("block")
        waveform = torch.randn(1, 1600)
        prepared = pipeline._prepare_audio_context(
            waveform,
            source_frames=3,
            target_frames=4,
        )
        self.assertEqual(prepared.ndim, 2)
        self.assertEqual(pipeline.context_encoder[0].calls, 0)

        encoded = pipeline.context_encoding(prepared, 0, 2, 2)
        self.assertEqual(encoded.shape, (1, 2, 8))
        self.assertEqual(pipeline.context_encoder[0].calls, 1)

    def test_full_clip_face_head_runs_once_before_padding(self) -> None:
        pipeline = _pipeline("full_clip")
        waveform = torch.linspace(-1.0, 1.0, 3200).unsqueeze(0)

        features, expression = pipeline._prepare_audio_conditions(
            waveform,
            torch.tensor([20]),
            source_frames=6,
            target_frames=8,
        )

        self.assertEqual(features.shape, (1, 8, 8))
        self.assertEqual(expression.shape, (1, 8, 100))
        self.assertEqual(pipeline.context_encoder[0].calls, 1)
        self.assertEqual(pipeline.context_encoder[1].calls, 1)
        self.assertEqual(pipeline.speech_model.calls, 1)
        self.assertEqual(pipeline.speech_model.input_lengths, [6])
        torch.testing.assert_close(
            expression[:, :6],
            features[:, :6, :1].repeat(1, 1, 100),
        )
        torch.testing.assert_close(
            expression[:, 6:],
            torch.zeros_like(expression[:, 6:]),
        )

    def test_block_scope_does_not_precompute_face_head(self) -> None:
        pipeline = _pipeline("block")
        features, expression = pipeline._prepare_audio_conditions(
            torch.randn(1, 1600),
            torch.tensor([20]),
            source_frames=3,
            target_frames=4,
        )
        self.assertEqual(features.ndim, 2)
        self.assertIsNone(expression)
        self.assertEqual(pipeline.speech_model.calls, 0)

    def test_validation_can_reuse_lmdb_features_and_face_context(self) -> None:
        pipeline = _pipeline("full_clip")
        features = torch.randn(2, 6, 8)
        expression_cache = torch.randn(2, 6, 100)

        prepared, expression = pipeline._prepare_audio_conditions(
            features,
            torch.tensor([20, 21]),
            source_frames=6,
            target_frames=8,
            precomputed_expression=expression_cache,
        )

        self.assertEqual(pipeline.context_encoder[0].calls, 0)
        self.assertEqual(pipeline.context_encoder[1].calls, 0)
        self.assertEqual(pipeline.speech_model.calls, 0)
        torch.testing.assert_close(prepared[:, :6], features)
        torch.testing.assert_close(expression[:, :6], expression_cache)
        torch.testing.assert_close(
            prepared[:, 6:],
            torch.zeros_like(prepared[:, 6:]),
        )
        torch.testing.assert_close(
            expression[:, 6:],
            torch.zeros_like(expression[:, 6:]),
        )

    def test_short_motion_padding_uses_safe_fallback(self) -> None:
        noise = torch.arange(9.0).reshape(1, 9, 1)
        padded = CausalEmageInferencePipeline._pad_noise_frames(noise, 55)
        self.assertEqual(padded.shape, (1, 64, 1))
        torch.testing.assert_close(padded[:, :9], noise)


if __name__ == "__main__":
    unittest.main()
