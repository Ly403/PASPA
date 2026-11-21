#!/usr/bin/env python3
"""Apply fail-closed modern-runtime fixes to a pinned TalkSHOW checkout."""

import argparse
import json
from pathlib import Path


EXPECTED_COMMIT = "9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
PATCH_VERSION = 5


REPLACEMENTS = {
    "nets/__init__.py": [
        (
            "from .smplx_face import TrainWrapper as s2g_face\n"
            "from .smplx_body_vq import TrainWrapper as s2g_body_vq\n"
            "from .smplx_body_pixel import TrainWrapper as s2g_body_pixel\n"
            "from .body_ae import TrainWrapper as s2g_body_ae\n"
            "from .LS3DCG import TrainWrapper as LS3DCG\n"
            "from .base import TrainWrapperBaseClass\n"
            "\n"
            "from .utils import normalize, denormalize",
            "\"\"\"TalkSHOW model package with stage-local imports.\"\"\"\n",
        ),
    ],
    "nets/init_model.py": [
        (
            "from nets import *\n"
            "\n"
            "\n"
            "def init_model(model_name, args, config):\n"
            "\n"
            "    if model_name == 's2g_face':\n"
            "        generator = s2g_face(\n"
            "            args,\n"
            "            config,\n"
            "        )\n"
            "    elif model_name == 's2g_body_vq':\n"
            "        generator = s2g_body_vq(\n"
            "            args,\n"
            "            config,\n"
            "        )\n"
            "    elif model_name == 's2g_body_pixel':\n"
            "        generator = s2g_body_pixel(\n"
            "            args,\n"
            "            config,\n"
            "        )\n"
            "    elif model_name == 's2g_body_ae':\n"
            "        generator = s2g_body_ae(\n"
            "            args,\n"
            "            config,\n"
            "        )\n"
            "    elif model_name == 's2g_LS3DCG':\n"
            "        generator = LS3DCG(\n"
            "            args,\n"
            "            config,\n"
            "        )\n"
            "    else:\n"
            "        raise ValueError\n"
            "    return generator\n"
            "\n"
            "\n",
            "def init_model(model_name, args, config):\n"
            "    if model_name == 's2g_face':\n"
            "        from nets.smplx_face import TrainWrapper\n"
            "    elif model_name == 's2g_body_vq':\n"
            "        from nets.smplx_body_vq import TrainWrapper\n"
            "    elif model_name == 's2g_body_pixel':\n"
            "        from nets.smplx_body_pixel import TrainWrapper\n"
            "    elif model_name == 's2g_body_ae':\n"
            "        from nets.body_ae import TrainWrapper\n"
            "    elif model_name == 's2g_LS3DCG':\n"
            "        from nets.LS3DCG import TrainWrapper\n"
            "    else:\n"
            "        raise ValueError(f'unknown model: {model_name}')\n"
            "    return TrainWrapper(args, config)\n",
        ),
    ],
    "scripts/train.py": [
        (
            "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n",
            "# CUDA visibility is controlled by the launcher.\n",
        ),
    ],
    "trainer/Trainer.py": [
        (
            "        time_stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))\n",
            "        time_stamp = os.environ.get('TALKSHOW_RUN_DATE') or time.strftime('%Y-%m-%d',time.localtime(time.time()))\n",
        ),
    ],
    "nets/smplx_body_pixel.py": [
        (
            "        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))\n",
            "        model_ckpt = torch.load(\n"
            "            model_path, map_location=torch.device('cpu'), weights_only=False\n"
            "        )\n",
        ),
    ],
    "data_utils/dataloader_torch.py": [
        (
            'from transformers import Wav2Vec2Processor\n',
            '',
        ),
        (
            '            am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")\n',
            "            # Faceformer needs only a non-None raw-waveform sentinel.\n"
            "            # Body MFCC must stay on the direct torchaudio path; using the\n"
            "            # sentinel there would decode every clip twice.\n"
            "            am = object() if config.Model.model_type == 'face' else None\n",
        ),
        (
            "            f = open(self.split+config.Data.pklname, 'rb+')\n",
            "            f = open(self.split+config.Data.pklname, 'rb')\n",
        ),
        (
            "            import pickle\n"
            "\n"
            "            f = open(self.split+config.Data.pklname, 'wb')\n"
            "            pickle.dump(self.dataset, f)\n"
            "            f.close()\n",
            "            if os.environ.get('TALKSHOW_WRITE_DATA_CACHE', '0') == '1':\n"
            "                import pickle\n"
            "                with open(self.split + config.Data.pklname, 'wb') as f:\n"
            "                    pickle.dump(self.dataset, f)\n",
        ),
    ],
    "data_utils/mesh_dataset.py": [
        (
            ".astype(np.float)",
            ".astype(np.float32)",
            17,
        ),
        (
            "        f = open(motion_fn, 'rb+')\n",
            "        f = open(motion_fn, 'rb')\n",
        ),
    ],
    "data_utils/utils.py": [
        (
            "import torchaudio as ta\n"
            "import torchaudio.functional as ta_F\n"
            "import torchaudio.transforms as ta_T\n",
            "try:\n"
            "    import torchaudio as ta\n"
            "    import torchaudio.functional as ta_F\n"
            "    import torchaudio.transforms as ta_T\n"
            "except (ImportError, OSError):\n"
            "    ta = ta_F = ta_T = None\n",
        ),
    ],
    "nets/spg/s2g_face.py": [
        (
            "from transformers import Wav2Vec2Processor\n",
            "from transformers import WavLMModel\n",
        ),
        (
            "from torchaudio.sox_effects import apply_effects_tensor\n",
            "try:\n"
            "    from torchaudio.sox_effects import apply_effects_tensor\n"
            "except (ImportError, OSError):\n"
            "    apply_effects_tensor = None\n",
        ),
        (
            "import torchaudio as ta\n",
            "try:\n"
            "    import torchaudio as ta\n"
            "except (ImportError, OSError):\n"
            "    ta = None\n",
        ),
        (
            "                 num_classes=0,\n"
            "                 ):\n",
            "                 num_classes=0,\n"
            "                 audio_encoder='wav2vec2',\n"
            "                 audio_model_path=None,\n"
            "                 audio_do_normalize=False,\n"
            "                 freeze_audio_encoder=False,\n"
            "                 ):\n",
        ),
        (
            '            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  # "vitouphy/wav2vec2-xls-r-300m-phoneme""facebook/wav2vec2-base-960h"\n'
            "            self.audio_encoder.feature_extractor._freeze_parameters()\n"
            "            self.audio_feature_map = nn.Linear(768, in_dim)\n",
            "            self.audio_encoder_name = str(audio_encoder).lower()\n"
            "            self.audio_do_normalize = bool(audio_do_normalize)\n"
            "            self.freeze_audio_encoder = bool(freeze_audio_encoder)\n"
            "            if self.audio_encoder_name == 'wav2vec2':\n"
            "                model_source = audio_model_path or os.environ.get(\n"
            "                    'TALKSHOW_WAV2VEC_PATH', 'facebook/wav2vec2-base-960h'\n"
            "                )\n"
            "                load_kwargs = {} if os.path.isdir(model_source) else {\n"
            "                    'revision': '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'\n"
            "                }\n"
            "                self.audio_encoder = Wav2Vec2Model.from_pretrained(\n"
            "                    model_source, **load_kwargs\n"
            "                )\n"
            "                self.audio_encoder.feature_extractor._freeze_parameters()\n"
            "                audio_hidden_size = 768\n"
            "            elif self.audio_encoder_name == 'wavlm':\n"
            "                if not audio_model_path:\n"
            "                    raise ValueError(\n"
            "                        'WavLM requires Model.audio_model_path'\n"
            "                    )\n"
            "                self.audio_encoder = WavLMModel.from_pretrained(\n"
            "                    audio_model_path\n"
            "                )\n"
            "                if getattr(self.audio_encoder.config, 'add_adapter', False):\n"
            "                    raise ValueError('WavLM checkpoints with adapters are unsupported')\n"
            "                audio_hidden_size = self.audio_encoder.config.hidden_size\n"
            "            else:\n"
            "                raise ValueError(\n"
            "                    f'unsupported face audio encoder: {self.audio_encoder_name}'\n"
            "                )\n"
            "            if self.freeze_audio_encoder:\n"
            "                self.audio_encoder.requires_grad_(False)\n"
            "                self.audio_encoder.eval()\n"
            "            self.audio_feature_map = nn.Linear(audio_hidden_size, in_dim)\n"
            "            # Transformers can leave this checkpoint-missing parameter\n"
            "            # partially uninitialized when loading the legacy Wav2Vec2 model.\n"
            "            masked_spec_embed = getattr(\n"
            "                self.audio_encoder, 'masked_spec_embed', None\n"
            "            )\n"
            "            if (\n"
            "                masked_spec_embed is not None\n"
            "                and not torch.isfinite(masked_spec_embed).all()\n"
            "            ):\n"
            "                with torch.no_grad():\n"
            "                    masked_spec_embed.uniform_()\n",
        ),
        (
            "            hidden_states = self.audio_encoder(in_spec.reshape(in_spec.shape[0], -1), frame_num=time_steps).last_hidden_state\n"
            "            feature = self.audio_feature_map(hidden_states).transpose(1, 2)\n",
            "            waveform = in_spec.reshape(in_spec.shape[0], -1)\n"
            "            if self.audio_encoder_name == 'wav2vec2':\n"
            "                hidden_states = self.audio_encoder(\n"
            "                    waveform, frame_num=time_steps\n"
            "                ).last_hidden_state\n"
            "                feature = self.audio_feature_map(hidden_states).transpose(1, 2)\n"
            "            else:\n"
            "                if self.audio_do_normalize:\n"
            "                    waveform = waveform - waveform.mean(dim=-1, keepdim=True)\n"
            "                    waveform = waveform / torch.sqrt(\n"
            "                        waveform.var(dim=-1, keepdim=True, unbiased=False) + 1e-7\n"
            "                    )\n"
            "                if self.freeze_audio_encoder:\n"
            "                    self.audio_encoder.eval()\n"
            "                    with torch.no_grad():\n"
            "                        hidden_states = self.audio_encoder(\n"
            "                            waveform\n"
            "                        ).last_hidden_state\n"
            "                else:\n"
            "                    hidden_states = self.audio_encoder(\n"
            "                        waveform\n"
            "                    ).last_hidden_state\n"
            "                feature = self.audio_feature_map(hidden_states).transpose(1, 2)\n"
            "                if time_steps is not None:\n"
            "                    feature = F.interpolate(\n"
            "                        feature,\n"
            "                        size=time_steps,\n"
            "                        mode='linear',\n"
            "                        align_corners=True,\n"
            "                    )\n",
        ),
    ],
    "nets/spg/wav2vec.py": [
        (
            "        self.config.output_attentions = True\n",
            "        # Keep the configured attention backend; attention maps are unused here.\n",
        ),
        (
            "        hidden_states = self.feature_projection(hidden_states)\n",
            "        hidden_states, _ = self.feature_projection(hidden_states)\n",
        ),
        (
            "                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)\n",
            "                mask_time_indices = torch.from_numpy(mask_time_indices).to(hidden_states.device)\n"
            "                hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)\n",
        ),
        (
            "            hidden_states[0],\n",
            "            hidden_states,\n",
        ),
    ],
    "nets/smplx_face.py": [
        (
            "            num_classes=self.num_classes,\n"
            "        ).to(self.device)\n",
            "            num_classes=self.num_classes,\n"
            "            audio_encoder=getattr(\n"
            "                self.config.Model, 'audio_encoder', 'wav2vec2'\n"
            "            ),\n"
            "            audio_model_path=getattr(\n"
            "                self.config.Model, 'audio_model_path', None\n"
            "            ),\n"
            "            audio_do_normalize=getattr(\n"
            "                self.config.Model, 'audio_do_normalize', False\n"
            "            ),\n"
            "            freeze_audio_encoder=getattr(\n"
            "                self.config.Model, 'freeze_audio_encoder', False\n"
            "            ),\n"
            "        ).to(self.device)\n",
        ),
        (
            "        MSELoss = torch.mean(torch.abs(pred_poses[:, :, :6] - gt_poses[:, :, :6]))\n",
            "        # Default 6 preserves the released training behavior; 3 is the corrected jaw width.\n"
            "        jaw_width = 6 if os.environ.get('TALKSHOW_LEGACY_FACE_LOSS', '1') == '1' else b_e\n"
            "        MSELoss = torch.mean(torch.abs(pred_poses[:, :, :jaw_width] - gt_poses[:, :, :jaw_width]))\n",
        ),
    ],
}


def replace_exact(
    path: Path, old: str, new: str, expected_count: int, check: bool
) -> None:
    text = path.read_text(encoding="utf-8")
    if check:
        if text.count(new) < expected_count:
            raise RuntimeError(f"patch verification failed: {path}")
        return
    count = text.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"expected {expected_count} patch targets in {path}, found {count}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.upstream_dir.resolve()
    marker = root / ".paspa_talkshow_patch.json"

    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected upstream commit: {commit}")

    if args.check:
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if marker_data != {
            "upstream_commit": EXPECTED_COMMIT,
            "patch_version": PATCH_VERSION,
        }:
            raise RuntimeError("patch marker does not match this launcher")

    if not args.check:
        for relative, replacements in REPLACEMENTS.items():
            text = (root / relative).read_text(encoding="utf-8")
            for replacement in replacements:
                old = replacement[0]
                expected_count = replacement[2] if len(replacement) == 3 else 1
                observed_count = text.count(old)
                if observed_count != expected_count:
                    raise RuntimeError(
                        f"expected {expected_count} patch targets in "
                        f"{root / relative}, found {observed_count}"
                    )

    for relative, replacements in REPLACEMENTS.items():
        path = root / relative
        for replacement in replacements:
            old, new = replacement[:2]
            expected_count = replacement[2] if len(replacement) == 3 else 1
            replace_exact(path, old, new, expected_count, args.check)

    if not args.check:
        marker.write_text(
            json.dumps(
                {
                    "upstream_commit": EXPECTED_COMMIT,
                    "patch_version": PATCH_VERSION,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
