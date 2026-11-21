import textwrap
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = F = None

from reproductions.talkshow.scripts.patch_upstream import REPLACEMENTS


def build_forward_harness():
    replacement = next(
        new
        for old, new, *_ in REPLACEMENTS["nets/spg/s2g_face.py"]
        if "frame_num=time_steps" in old
    )
    source = (
        "def run(self, in_spec, time_steps):\n"
        + textwrap.indent(textwrap.dedent(replacement), "    ")
        + "    return feature\n"
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(source, "<face-audio-forward-patch>", "exec"), namespace)
    return namespace["run"]


class IdentityMap:
    def __call__(self, value):
        return value


class FakeEncoder:
    def __init__(self, hidden_states, expects_frame_num):
        self.hidden_states = hidden_states
        self.expects_frame_num = expects_frame_num
        self.frame_num = None
        self.input_values = None
        self.eval_called = False
        self.grad_enabled = None

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_values, frame_num=None):
        if self.expects_frame_num:
            if frame_num is None:
                raise AssertionError("Wav2Vec2 frame_num was not forwarded")
        elif frame_num is not None:
            raise AssertionError("WavLM does not accept frame_num")
        self.frame_num = frame_num
        self.input_values = input_values
        self.grad_enabled = torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=self.hidden_states)


@unittest.skipUnless(torch is not None, "PyTorch is installed in the face environment")
class FaceAudioForwardPatchTest(unittest.TestCase):
    def setUp(self):
        self.run = build_forward_harness()

    def test_wav2vec2_path_keeps_native_frame_num_contract(self):
        target_frames = 4
        encoder = FakeEncoder(
            torch.randn(1, target_frames, 3), expects_frame_num=True
        )
        harness = SimpleNamespace(
            audio_encoder_name="wav2vec2",
            audio_do_normalize=False,
            freeze_audio_encoder=False,
            audio_encoder=encoder,
            audio_feature_map=IdentityMap(),
        )
        waveform = torch.arange(16, dtype=torch.float32).reshape(1, 1, -1)

        feature = self.run(harness, waveform, target_frames)

        self.assertEqual(feature.shape, (1, 3, target_frames))
        self.assertEqual(encoder.frame_num, target_frames)
        torch.testing.assert_close(encoder.input_values, waveform.reshape(1, -1))

    def test_frozen_wavlm_normalizes_and_aligns_to_motion_frames(self):
        encoder = FakeEncoder(torch.randn(1, 7, 3), expects_frame_num=False)
        harness = SimpleNamespace(
            audio_encoder_name="wavlm",
            audio_do_normalize=True,
            freeze_audio_encoder=True,
            audio_encoder=encoder,
            audio_feature_map=IdentityMap(),
        )
        waveform = torch.arange(32, dtype=torch.float32).reshape(1, 1, -1)

        feature = self.run(harness, waveform, time_steps=5)

        self.assertEqual(feature.shape, (1, 3, 5))
        self.assertTrue(encoder.eval_called)
        self.assertFalse(encoder.grad_enabled)
        torch.testing.assert_close(
            encoder.input_values.mean(dim=-1), torch.zeros(1), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            encoder.input_values.var(dim=-1, unbiased=False),
            torch.ones(1),
            atol=1e-5,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
