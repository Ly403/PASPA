# TalkSHOW reproduction

This directory reproduces the training code released with the CVPR 2023
TalkSHOW paper.  It deliberately keeps the TalkSHOW pipeline separate from
PASPA: the upstream code is cloned at a pinned commit and receives a
small, audited compatibility patch for current CUDA/PyTorch environments.

The supported training chain is:

1. the native TalkSHOW face model (`jaw_pose + expression`), with the released
   Wav2Vec 2.0 backbone or an optional frozen WavLM Large backbone;
2. the body-and-hands VQ-VAE;
3. the audio-conditioned body-and-hands PixelCNN, initialized from the VQ-VAE
   checkpoint produced in step 2.

Face and body training are independent, so the face stage and body VQ-VAE can
run concurrently on two GPUs.  By default, each launcher preserves the
released single-GPU protocol.  Face training also has an explicit opt-in
WavLM-only DDP path; the body launchers remain single-GPU.

## 1. Obtain the licensed assets

The SHOW dataset is distributed through the authors' authenticated download
page.  Registration and acceptance of its terms are required, so there is no
honest anonymous `wget` command to put here.

1. Follow the dataset link in the
   [TalkSHOW repository](https://github.com/yhw-yhw/TalkSHOW).
2. Download `SHOW_dataset_v1.0.zip`.
3. Verify the archive before extracting it:

   ```bash
   echo "b5ddcd055eef117f49c929494d3a52fc8755f9a776d83b3bbd254dddc96d6b68  SHOW_dataset_v1.0.zip" \
     | sha256sum --check
   mkdir -p SHOW_dataset_v1.0
   unzip SHOW_dataset_v1.0.zip -d SHOW_dataset_v1.0
   DATA_ROOT="$PWD/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0"
   find "$DATA_ROOT" -name '*.tar.gz' -exec tar -xzf '{}' -C "$DATA_ROOT" \;
   ```

The resulting immutable source root must have this layout:

```text
ExpressiveWholeBodyDatasetReleaseV1.0/
  {oliver,chemistry,seth,conan}/<video>/<clip>/<clip>.{wav,pkl}
```

TalkSHOW evaluation additionally needs the authors' SMPL-X support bundle,
linked from the
[upstream TalkSHOW README](https://github.com/yhw-yhw/TalkSHOW#5-visualization).
Register at the official SMPL-X site and accept its license before using that
bundle.  The evaluation asset directory must contain:

```text
SMPLX_NEUTRAL_2020.npz
smplx_extra_joints.yaml
SMPLX_to_J14.pkl
```

## 2. Create the official split without modifying the source data

The upstream split utility moves the source clips.  The replacement below
creates a symlink-only view and never writes inside the downloaded dataset:

```bash
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --expected-missing 55
```

For the currently distributed archive the official manifest contains 17,165
clips, of which 55 WAV files are absent.  A complete validated view therefore
contains 13,687 train, 1,715 validation, and 1,708 test pairs.  The command
fails if the observed missing-file count differs.  Before loading it, the
builder verifies the repository split file against its pinned SHA-256 and uses
a restricted unpickler that cannot import classes.

For a quick four-speaker smoke dataset:

```bash
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_smoke_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --limit-per-speaker-split 1
```

## 3. Prepare a modern runtime and the pinned upstream source

Create an environment first, install the CUDA-enabled PyTorch build supplied
for the machine into that environment, and then install the lightweight
TalkSHOW dependencies:

```bash
python3.12 -m venv /path/to/talkshow_env
source /path/to/talkshow_env/bin/activate
# Install the site-supported CUDA PyTorch wheel or module here.
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -m pip install -r reproductions/talkshow/requirements-face-modern.txt
bash reproductions/talkshow/bootstrap_upstream.sh \
  /local-ssd/talkshow_upstream
```

The compatibility path is tested with Python 3.12.3, PyTorch 2.6.0/CUDA 12.8,
and Transformers 4.57.6 on NVIDIA H200.  Install the PyTorch build supplied for
your CUDA image rather than replacing it through this requirements file.

Body preprocessing additionally requires a `torchaudio` build with the exact
PyTorch C++ ABI used by the host.  With a normal stable PyTorch installation,
install the same stable release of `torchaudio`.  NVIDIA's PyTorch 25.01 image
ships `torch 2.6.0a0+ecf3bae40a.nv25.01`, for which the public stable wheel is
not ABI-compatible.  The tested build for that image is:

```bash
git clone --depth 1 --branch v2.6.0 \
  https://github.com/pytorch/audio.git /local-ssd/torchaudio-v2.6.0
cd /local-ssd/torchaudio-v2.6.0
test "$(git rev-parse HEAD)" = d8831425203385077a03c1d92cfbbe3bf2106008
BUILD_SOX=0 BUILD_FFMPEG=0 BUILD_RNNT=0 USE_CUDA=0 MAX_JOBS=8 \
  python -m pip install --no-deps --no-build-isolation .
python -c 'import torch, torchaudio; print(torch.__version__, torchaudio.__version__)'
```

Validate the audio path on one real dataset WAV before training:

```bash
python - <<'PY'
import torch
import torchaudio

wav = "/path/to/one/TalkSHOW/clip.wav"
audio, sample_rate = torchaudio.load(wav)
features = torchaudio.transforms.MFCC(
    sample_rate=sample_rate, n_mfcc=64
)(audio)
assert torch.isfinite(features).all()
print(audio.shape, sample_rate, features.shape)
PY
```

The bootstrap pins upstream TalkSHOW commit
`9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b`.  It does not vendor or silently
track changing upstream code.

The model is pinned to Hugging Face revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.  To prefetch it on a login node
and run compute offline:

```bash
hf download facebook/wav2vec2-base-960h \
  --revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156 \
  --include config.json model.safetensors \
  --local-dir /shared/talkshow_assets/wav2vec2-base-960h
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Pass
`--wav2vec-path /shared/talkshow_assets/wav2vec2-base-960h` to the face
launcher.  The path must be visible from the compute node; node-local SSD
prefetched on a login node is usually not shared.  If no local path is passed,
the same pinned revision is resolved through the normal Hugging Face cache.

For integration with this codebase, the face model can instead use the same
1024-dimensional WavLM Large snapshot configured in
`configs/audio/identity.yaml`.  If that local directory is unavailable,
prepare the pinned Hugging Face snapshot:

```bash
hf download patrickvonplaten/wavlm-libri-clean-100h-large \
  --revision e70e3a062ec399c46008ee55d1fb52c7ba338d5c \
  --include config.json preprocessor_config.json pytorch_model.bin \
    tokenizer_config.json special_tokens_map.json vocab.json added_tokens.json \
  --local-dir /shared/talkshow_assets/wavlm-libri-clean-100h-large
```

WavLM mode reads the checkpoint's preprocessing normalization, maps its
configured hidden size to 256, and linearly aligns the approximately 50 Hz
hidden states to the 30 FPS face targets using the main codebase's interpolation
convention.  It freezes WavLM to match the main codebase's fixed feature
extractor.  This is an integration variant, not the released TalkSHOW training
configuration.

## 4. Smoke-test and train the face stage

One-epoch smoke run (one train clip per speaker is sufficient):

```bash
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_smoke_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_smoke \
  --smoke
```

Full official-length run:

```bash
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_full
```

To train the WavLM Large variant, add these arguments to either command and
use a distinct run ID:

```text
--audio-encoder wavlm
--audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large
```

`--audio-model-path` must be a local Hugging Face model directory, not a raw
`WavLM-Large.pt` file.  The legacy `--wav2vec-path DIR` option remains an alias
for `--audio-encoder wav2vec2 --audio-model-path DIR`.

`--smplx-assets /path/to/smplx_assets` may be supplied at training time to
record absolute asset paths in the run config, but face training itself does
not read those files.  They become mandatory for face evaluation.

Without the new distributed arguments, the wrapper preserves the released
optimizer and training protocol: batch size 1, SGD with learning rate 0.001
and momentum 0.9, gradient clipping at 5, and 100 epochs.  It isolates the
output directory by run ID and disables the upstream's unused multi-gigabyte
dataset-cache side effect.

By default it preserves the released face-loss behavior for comparable
reproduction.  The upstream implementation supervises the first six output
channels against the first six pose channels even though the output ordering
is `jaw[3] + expression[100]`; this means three expression channels are also
compared with left-eye pose.  Set `--corrected-face-loss` only for an explicitly
non-paper-compatible ablation that supervises the three jaw channels instead.

The smoke launcher saves `ckpt-0.pth` and runs a finite-value/checkpoint
contract check before returning success.  A full run saves the same released
milestones: epochs 29, 49, and 99.

For the opt-in eight-GPU WavLM Large run (per-GPU batch 32, 1000 epochs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_8gpu_b32_e1000 \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --num-gpus 8 \
  --batch-size-per-gpu 32 \
  --epochs 1000 \
  --corrected-face-loss
```

This uses single-node `torchrun` DDP and mask-aware, length-bucketed padding.
It requires the documented WavLM Large checkpoint's per-time-step feature
LayerNorm and fails closed for padding-unsafe feature norms.  It remains
entirely FP32: no BF16/FP16 autocast or gradient scaler is enabled.  SGD,
learning rate 0.001, momentum 0.9, gradient clipping, and the frozen WavLM
backbone are unchanged.  The 13,687-clip training split produces 53 full
steps plus one balanced tail step per epoch: 54 steps per epoch and 54,000
steps over 1000 epochs.  The tail has 15 samples per GPU and repeats one
sample globally so that all ranks stay synchronized.  The final checkpoint is
`ckpt-999.pth` and keeps the released wrapper state without a `module.` prefix.

Run the same distributed command with `--epochs 1` and a separate run ID
before the long job.  Whole-video WavLM memory depends strongly on clip length,
so batch 32 may still need to be reduced to 16 or 8.  The upstream eager
dataset is also constructed independently by all eight processes, which can
require close to eight times the single-process CPU memory and increase
startup I/O.  The DDP path sorts clip paths deterministically; before the first
step, it also compares the sample count and ordered dataset SHA-256 across
ranks and fails if they differ.

Face checkpoints can be resumed at a completed epoch boundary with
`--resume-from`.  `--epochs` remains the total target, and a new run ID is
required so the previous run is never overwritten.  For example, resume an
eight-GPU `ckpt-499.pth` at epoch 500 and finish epoch 999 by repeating the
same command above with:

```text
--run-id face_wavlm_large_8gpu_b32_e1000_resume500
--epochs 1000
--resume-from /shared/old-run/experiments/.../ckpt-499.pth
```

Keep the audio encoder, GPU count, per-GPU batch, dataset, speaker order, and
face-loss mode unchanged.  New DDP checkpoints preserve every rank's Python,
NumPy, CPU Torch, and CUDA RNG state in addition to the model, SGD optimizer,
progress counters, dataset fingerprint, and training contract.  Older DDP
checkpoints can still resume their model/optimizer/progress, but emit warnings
because they cannot reproduce the uninterrupted RNG stream exactly.  Released
single-GPU Trainer/launcher checkpoints that include model and optimizer state
are also accepted by the single-GPU path and start at the saved epoch plus
one.  Weight-only/head-only files are rejected.  `--resume-from` is
epoch-boundary only and cannot be combined with `--smoke`.
Checkpoints produced after manually invoking the upstream trainer's old,
off-by-one `--resume` path may have inconsistent epoch/step counters and are
rejected by the fail-closed checks.

The DDP dataset fingerprint uses clip IDs relative to `data_root` plus speaker,
audio-length, and motion-length metadata.  It therefore survives relocating an
otherwise identical split, but it is not a hash of every floating-point sample.
When an adjacent run-local `face.json` exists, single-GPU resume requires the
same absolute `Data.data_root` and semantic model/loss options.  A missing old
config emits a warning, in which case the user is responsible for preserving
the dataset identity.

Use shared persistent storage for `--output-dir`.  The split view and pinned
upstream checkout are reconstructible and may live on node-local SSD.

The end-to-end smoke was validated on H200 with all four speaker identities:
four optimizer steps completed with finite loss and gradients, and the saved
checkpoint reloaded with `global_steps=4` and 508 finite tensors.

## 5. Smoke-test and train the body stages

The body stages use the same pinned checkout, split view, and Python
environment.  The launcher refuses to run unless `CUDA_VISIBLE_DEVICES`
contains exactly one GPU ID.  This matters because the released PixelCNN code
otherwise enables `DataParallel` when it sees multiple GPUs and changes the
training protocol.

Smoke-test the VQ-VAE first:

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/talkshow/train_body_stage.sh \
  --stage vq \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_smoke_split \
  --output-dir /shared/talkshow_smoke \
  --run-id body_vq_smoke \
  --smoke

VQ_SMOKE_CKPT="$(
  find /shared/talkshow_smoke/body_vq_smoke -name ckpt-0.pth -print -quit
)"
test -n "$VQ_SMOKE_CKPT"
```

Then smoke-test PixelCNN with that exact VQ-VAE checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/talkshow/train_body_stage.sh \
  --stage pixel \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_smoke_split \
  --output-dir /shared/talkshow_smoke \
  --run-id body_pixel_smoke \
  --vq-checkpoint "$VQ_SMOKE_CKPT" \
  --smoke
```

Both smoke commands run one epoch with batch size 1, require at least one
optimizer step, reload the resulting `ckpt-0.pth`, and reject any checkpoint
containing non-finite model tensors.  On the tested H200 setup, each stage
completed 129 real optimizer steps over the four-speaker smoke split.  The VQ
and PixelCNN checkpoints contained 992 and 828 finite tensors, respectively.

For the full released-length body training:

```bash
CUDA_VISIBLE_DEVICES=0 bash reproductions/talkshow/train_body_stage.sh \
  --stage vq \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id body_vq_full

VQ_FULL_CKPT="$(
  find /shared/talkshow_runs/body_vq_full -name ckpt-99.pth -print -quit
)"
test -n "$VQ_FULL_CKPT"

CUDA_VISIBLE_DEVICES=0 bash reproductions/talkshow/train_body_stage.sh \
  --stage pixel \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id body_pixel_full \
  --vq-checkpoint "$VQ_FULL_CKPT"
```

The full launchers preserve the released four-speaker order, 88-frame motion
windows, codebook size 2048, single-GPU batch size 128, Adam learning rate
`1e-4`, gradient clipping at 5, and 100 epochs.  Keep both final checkpoints:
the PixelCNN checkpoint does not embed the VQ-VAE body and hand decoders, so
official inference requires both files.

The released repository's automated body evaluator was added after the paper
and reports LVD, error, diversity, FGD, feature distance, and beat consistency.
These are not the same as the human body-and-hands scores in CVPR 2023 Table 2;
results from that script should be labelled “released evaluator metrics,” not
paper Table 2 reproduction.
