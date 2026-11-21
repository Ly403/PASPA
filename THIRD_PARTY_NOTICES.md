# Third-party notices

The Apache-2.0 license in the repository root applies to PASPA-authored source
code unless a file or directory states otherwise. It does not replace or
relicense third-party code, datasets, body models, pretrained models, or other
assets. Notices retained in individual source files take precedence for those
files.

The following list summarizes the principal projects and assets described in
the paper's license appendix.

| Project or asset | License or terms | Use in this repository |
|---|---|---|
| [SANA](https://github.com/NVlabs/Sana) | Apache-2.0 | Base flow-matching training framework |
| [GlobalDiff](https://github.com/Human3DAIGC/GlobalDiff) | Apache-2.0 | BEAT2 data processing, FK/loss components, network components, and audio-to-expression weights |
| [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | MIT | Gated DeltaNet implementation and Triton kernels extended for block-causal execution |
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Apache-2.0 | Reference autoregressive diffusion inference pipeline |
| [EMAGE / Character Policy](https://github.com/haiyangliu0/character-policy) | Apache-2.0 | BEAT2 assets and evaluation tools |
| [DiffSHEG](https://github.com/JeremyCJM/DiffSHEG) | BSD-3-Clause | SHOW evaluation components and pretrained feature extractors |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | MIT | Frozen speech feature extractor |
| [BEAT2 / BEAT-X](https://pantomatrix.github.io/EMAGE/) | Apache-2.0 | Training and evaluation dataset; not redistributed here |
| [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW) and SHOW | Non-commercial scientific research terms; no repository-level license file is provided by TalkSHOW | Optional face/body reproduction and SHOW experiments; upstream source and data are obtained separately by users |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | SMPL-X non-commercial scientific research license | Body model and compatible human representation; model files are not redistributed here |

Setup scripts may download additional checkpoints or model assets. Those files
remain governed by the terms published by their respective authors and are not
covered by the PASPA Apache-2.0 license merely because they are placed under
`tools/`.

The released BEAT2 checkpoints and generated-motion archives may contain or be
derived from third-party data and model components. Users must also comply with
the corresponding dataset and model terms. Nothing in this notice grants rights
to BEAT2/BEAT-X, SHOW, SMPL-X, a person's identity or likeness, or any other
third-party asset beyond the rights granted by its owner.
