# TalkSHOW 主模型训练 Pipeline

本文说明如何从 SHOW 原始数据生成 LMDB，并使用 TalkSHOW face network 训练、恢复和评估 PASPA。所有命令均在仓库根目录执行。TalkSHOW 与 BEAT2 使用独立的数据、rest-joint table 和 checkpoint，不会改变 BEAT2 训练流程。

## 1. 三档配置

| `TALKSHOW_PROFILE` | Sequence | Block | Stride | 默认每卡 batch | 默认 epochs |
|---|---:|---:|---:|---:|---:|
| `block16_seq64` | 64 | 16 | 20 | 72 | 1000 |
| `block32_seq128` | 128 | 32 | 20 | 72 | 1000 |
| `block64_seq256` | 256 | 64 | 20 | 72 | 1000 |

默认 profile 是 `block64_seq256`。三档均使用 4 张 GPU 和 FP32；可通过 `CUDA_VISIBLE_DEVICES`、`NUM_PROCESSES`、`TRAIN_BATCH_SIZE`、`NUM_EPOCHS` 和 `NUM_WORKERS` 覆写。Sequence 长度不同，因此三档必须使用各自的 LMDB，不能只改 YAML 复用另一档数据。

## 2. 准备资产

先安装主项目并准备 WavLM/EMAGE 资产：

```bash
bash environment_setup.sh paspa
conda activate paspa
```

设置计算节点可见的绝对路径：

```bash
export SHOW_SOURCE_ROOT=/absolute/path/to/ExpressiveWholeBodyDatasetReleaseV1.0
export SHOW_SPLIT_ROOT=/absolute/path/to/talkshow_split
export SHOW_LMDB_DIR=/absolute/path/to/show_lmdb
export TALKSHOW_FACE_CKPT=/absolute/path/to/face/ckpt-999.pth
export WAVLM_DIR=/absolute/path/to/wavlm-libri-clean-100h-large
export EMAGE_EVALTOOLS_DIR=/absolute/path/to/emage_evaltools
export SHOW_REST_JOINT_PATH="$SHOW_LMDB_DIR/rest_joint_position_show.mat"
```

`EMAGE_EVALTOOLS_DIR` 至少应包含：

```text
AESKConv_240_100.bin
mean_vel_smplxflame_30.npy
smplx_models/smplx/SMPLX_NEUTRAL_2020.npz
```

主模型要求使用 `reproductions/talkshow/train_face.sh` 生成的 corrected-loss WavLM Large face checkpoint。完整训练方法见 [reproductions/talkshow/README_zh.md](../../reproductions/talkshow/README_zh.md)。face network 与主干内部都使用身份 `0..3`；LMDB 和 rest table 保留 raw PID `21..24`，dataset 在送入两个网络前统一减 21。

可先检查 face/WavLM 契约：

```bash
python tools/show/verify_talkshow_face_condition.py \
  --checkpoint "$TALKSHOW_FACE_CKPT" \
  --wavlm-dir "$WAVLM_DIR" \
  --required-face-loss corrected
```

## 3. 创建官方 split view

该步骤只需执行一次。它创建符号链接视图，不会移动或修改原始数据：

```bash
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root "$SHOW_SOURCE_ROOT" \
  --output-root "$SHOW_SPLIT_ROOT" \
  --split-file tools/show/split_more_than_2s.pkl \
  --expected-missing 55
```

当前数据发行版应得到 train 13,687、validation 1,715、test 1,708 条有效音频/动作对。若已有使用同一 manifest 成功生成的 `split_view_summary.json`，无需因为 sequence 长度变化而重建 split view。

## 4. 生成对应长度的 LMDB

选择一档并设置长度，例如：

```bash
export TALKSHOW_PROFILE=block64_seq256
export SEQ_SIZE=256
```

生成 train/validation/test：

```bash
set -euo pipefail
mkdir -p "$SHOW_LMDB_DIR"

for SPLIT in train val test; do
  CUDA_VISIBLE_DEVICES=0 python scripts/create_lmdb_show.py "$SHOW_SPLIT_ROOT" \
    --split "$SPLIT" \
    --seq_size "$SEQ_SIZE" \
    --stride_size 20 \
    --local_rotation 0 \
    --model_type wavlm \
    --model_id "$WAVLM_DIR" \
    --require_split_summary \
    --feature_scope full_clip \
    --max_av_duration_delta_frames 3 \
    --duration_mismatch_policy skip \
    --pid_offset 1 \
    --lmdb_path "$SHOW_LMDB_DIR/show_${SPLIT}_seq_size_${SEQ_SIZE}_stride_size_20_global.lmdb"
done
```

音画差超过 3 帧的坏 clip 会被整条跳过，脚本结束时会汇总跳过数量。train 与 validation 按 sequence/stride 切窗；test 每个有效源 clip 只保存一条完整变长序列，不会切成训练窗口。

分别把 `SEQ_SIZE` 设为 `64`、`128`、`256` 即可生成三档数据。每档都需要独立的三个 LMDB。

## 5. 生成 rest-joint table

rest table 只依赖四位人物的 shape，因此可由任一合格的 train LMDB 生成，并由三档共同使用：

```bash
python scripts/build_rest_joint_position_show.py \
  --lmdb_path "$SHOW_LMDB_DIR/show_train_seq_size_${SEQ_SIZE}_stride_size_20_global.lmdb" \
  --smplx_model_path "$EMAGE_EVALTOOLS_DIR/smplx_models/smplx" \
  --expected_pids 21 22 23 24 \
  --expected-pid-offset 1 \
  --output "$SHOW_REST_JOINT_PATH"
```

不要覆盖 BEAT2 的 `diffusion/data/datasets/beat2/rest_joint_position.mat`。

## 6. 静态契约检查

对目标 sequence 的三个 LMDB 执行检查：

```bash
for SPLIT in train val test; do
  python tools/show/verify_lmdb_contract.py \
    --lmdb-path "$SHOW_LMDB_DIR/show_${SPLIT}_seq_size_${SEQ_SIZE}_stride_size_20_global.lmdb" \
    --expected-split "$SPLIT" \
    --expected-seq-size "$SEQ_SIZE" \
    --expected-pids 21 22 23 24 \
    --expected-pid-offset 1 \
    --expected-max-av-duration-delta-frames 3 \
    --rest-joint-path "$SHOW_REST_JOINT_PATH" \
    --face-checkpoint "$TALKSHOW_FACE_CKPT" \
    --wavlm-dir "$WAVLM_DIR" \
    --required-face-loss corrected
done
```

训练入口默认会在启动 GPU 任务前再执行一次完整 preflight。仅在已经确认数据不变时才设置 `RUN_CONTRACT_PREFLIGHT=0`。

## 7. 训练与恢复

四卡默认训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TALKSHOW_PROFILE=block64_seq256 \
TALKSHOW_MODE=train \
bash train_scripts/run_talkshow.sh
```

另外两档只需同时更换 `TALKSHOW_PROFILE` 与相应 LMDB：

```bash
TALKSHOW_PROFILE=block16_seq64 TALKSHOW_MODE=train \
bash train_scripts/run_talkshow.sh

TALKSHOW_PROFILE=block32_seq128 TALKSHOW_MODE=train \
bash train_scripts/run_talkshow.sh
```

例如改为 8 卡、每卡 batch 32：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 TRAIN_BATCH_SIZE=32 NUM_EPOCHS=1000 \
TALKSHOW_PROFILE=block64_seq256 TALKSHOW_MODE=train \
bash train_scripts/run_talkshow.sh
```

恢复训练时使用顶层 `resume_from`，并保持 profile、LMDB、face checkpoint、WavLM、GPU 数与 batch size 一致：

```bash
TALKSHOW_PROFILE=block64_seq256 TALKSHOW_MODE=train \
bash train_scripts/run_talkshow.sh \
  resume_from=/absolute/path/to/output/checkpoints/epoch_N_step_M.pth
```

`NUM_EPOCHS` 是整个 run 的目标 epoch，不是额外增加的 epoch。

## 8. Validation 与 test

纯评估使用 `model.load_from`，不恢复 optimizer、scheduler、epoch 或 RNG：

```bash
TALKSHOW_PROFILE=block64_seq256 TALKSHOW_MODE=validation \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/epoch_N_step_M.pth

TALKSHOW_PROFILE=block64_seq256 TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/epoch_N_step_M.pth
```

若明确要用 test LMDB 运行 validation 流程，可显式覆写：

```bash
TALKSHOW_PROFILE=block64_seq256 \
TALKSHOW_MODE=validation \
TALKSHOW_VALIDATION_LMDB="$SHOW_LMDB_DIR/show_test_seq_size_256_stride_size_20_global.lmdb" \
TALKSHOW_VALIDATION_SPLIT=test \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/epoch_N_step_M.pth
```

未设置 `TALKSHOW_VALIDATION_LMDB` 时仍读取当前 profile 默认的 validation LMDB。validation/test batch size 固定为 1，专用分布式 sampler 会过滤为凑齐 world size 产生的重复输出。

批量评估 checkpoint：

```bash
python scripts/eval_talkshow_pth_dir.py --help
```

离线评估协议：

- [TalkSHOW 原仓库协议](OFFICIAL_EVAL_zh.md)
- [DiffSHEG SHOW 协议](DIFFSHEG_EVAL_zh.md)

两套协议的特征提取器和 FGD 数值尺度不同，结果不能直接互相比较。
