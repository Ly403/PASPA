# TalkSHOW Face Network 中文训练指南

本目录用于复现 TalkSHOW 发布的 face generation 训练流程。训练得到的是一个参数化人脸网络：

```text
16 kHz 原始音频 + 说话人 ID
                ↓
 Wav2Vec2（论文复现）或 WavLM Large（主工程对接）
                ↓
jaw_pose（3 维）+ SMPL-X expression（100 维）
```

输出是每帧 103 维 SMPL-X 脸部参数，不是 RGB 图像、视频或 mesh 顶点。

> 注意：该网络仍是 TalkSHOW 原生 `s2g_face`，只是可以选择 WavLM
> 音频骨干。主项目通过 `TalkSHOWFaceConditionModel` 加载其 face head，
> 不会将该 checkpoint 当作 `SimpleSpeechModel` 权重。

## 1. 训练流程概览

完整流程为：

1. 下载已授权的 SHOW 数据集；
2. 创建不修改原始数据的官方 train/val/test 软链接视图；
3. 创建独立 Python 环境并安装依赖；
4. 获取固定 commit 的 TalkSHOW 上游代码并应用兼容补丁；
5. 准备 Wav2Vec2 或主工程同款 WavLM Large 权重；
6. 先运行一轮 smoke training；
7. 再启动默认单卡训练，或显式启用八卡大 batch 训练。

训练使用 TalkSHOW 的四名说话人：

```text
oliver
chemistry
seth
conan
```

默认训练协议（不传新增参数时保持不变）：

| 项目 | 配置 |
|---|---:|
| GPU | 单卡 |
| Batch size | 1 |
| 数据方式 | whole-video |
| Epoch | 100 |
| Optimizer | SGD |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Gradient clipping | 5 |
| Motion FPS | 30 |
| Audio sample rate | 16 kHz |

WavLM 另提供一个增量式的大 batch/DDP 入口。它不会替换上面的默认路径；
只有显式传入 `--num-gpus > 1` 或 `--batch-size-per-gpu > 1` 时才会启用。

音频骨干有两种模式：

- `wav2vec2`：默认值，保留 TalkSHOW 发布代码的复现路径；
- `wavlm`：用于和本项目统一音频表征的新训练路径，不属于论文原始配置。

## 2. 准备 SHOW 数据集

SHOW 数据集需要从作者提供的认证页面下载并接受相应许可。下载文件名应为：

```text
SHOW_dataset_v1.0.zip
```

可以使用以下命令校验和解压：

```bash
echo "b5ddcd055eef117f49c929494d3a52fc8755f9a776d83b3bbd254dddc96d6b68  SHOW_dataset_v1.0.zip" \
  | sha256sum --check

mkdir -p SHOW_dataset_v1.0
unzip SHOW_dataset_v1.0.zip -d SHOW_dataset_v1.0

DATA_ROOT="$PWD/SHOW_dataset_v1.0/ExpressiveWholeBodyDatasetReleaseV1.0"
find "$DATA_ROOT" -name '*.tar.gz' -exec tar -xzf '{}' -C "$DATA_ROOT" \;
```

解压后的原始目录应类似：

```text
ExpressiveWholeBodyDatasetReleaseV1.0/
  oliver/<video>/<clip>/<clip>.wav
  oliver/<video>/<clip>/<clip>.pkl
  chemistry/...
  seth/...
  conan/...
```

不要运行 TalkSHOW 原始的 `apply_split.py` 或
`dataset_preprocess.py`。它们会移动甚至删除原始数据。本目录提供的
`prepare_split_view.py` 只创建软链接。

### 2.1 创建完整官方 split

建议把软链接视图放在本地 SSD，并保证它位于原始数据目录之外：

```bash
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --expected-missing 55
```

当前发行版的官方 manifest 共包含 17,165 个 clip，其中有 55 个缺少
WAV。成功生成的视图应包含：

```text
train: 13687
val:    1715
test:   1708
```

如果实际缺失数量不是 55，脚本会停止。此时应先检查数据是否完整或版本
是否不同，不要直接修改 `--expected-missing` 掩盖问题。

### 2.2 创建 smoke 数据集

smoke 数据集为每个说话人、每个 split 选择一个有效 clip：

```bash
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_smoke_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --limit-per-speaker-split 1
```

face 训练只读取 `train` split，因此该 smoke run 通常执行 4 个 optimizer
step。

## 3. 创建独立训练环境

建议为 TalkSHOW reproduction 使用独立环境，避免修改主项目的依赖：

```bash
python3.12 -m venv /path/to/talkshow_env
source /path/to/talkshow_env/bin/activate
```

先根据机器的 CUDA 和驱动安装匹配的 PyTorch。不要让下面的 requirements
文件替换机器已经适配好的 CUDA PyTorch：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -m pip install -r reproductions/talkshow/requirements-face-modern.txt
```

推荐在训练前确认：

```bash
python - <<'PY'
import torch
import librosa
import smplx
import transformers

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY
```

face 训练本身不需要 SMPL-X 模型资产；SMPL-X support bundle 只在评估和
渲染时需要。

## 4. 准备固定版本的 TalkSHOW 上游代码

`reproductions/talkshow` 是训练 wrapper，不包含完整的 TalkSHOW 网络实现。
使用下面的脚本创建固定 commit、已应用现代环境兼容补丁的 checkout：

```bash
bash reproductions/talkshow/bootstrap_upstream.sh \
  /local-ssd/talkshow_upstream
```

固定的 TalkSHOW commit 为：

```text
9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b
```

不要直接把一个任意 TalkSHOW checkout 传给训练脚本。训练 launcher 会检查
commit 和 `.paspa_talkshow_patch.json`，不满足条件时会拒绝运行。

WavLM 支持对应 `patch_version: 5`。如果已有目录是旧补丁版本，请换一个新的
目标目录重新运行 `bootstrap_upstream.sh`；不要手工改 marker 冒充新版本。

## 5. 准备音频模型

### 5.1 推荐用于本项目：WavLM Large

主项目的 `configs/audio/identity.yaml` 使用本地
`wavlm-libri-clean-100h-large`。TalkSHOW face 现在可以使用同一个 Hugging
Face 模型目录：

```text
patrickvonplaten/wavlm-libri-clean-100h-large
revision: e70e3a062ec399c46008ee55d1fb52c7ba338d5c
hidden size: 1024
```

如果 `configs/audio/identity.yaml` 中的 `local_model_path` 已经可用，直接把
同一路径传给 `--audio-model-path`，不需要再次下载。否则在联网节点准备：

```bash
hf download patrickvonplaten/wavlm-libri-clean-100h-large \
  --revision e70e3a062ec399c46008ee55d1fb52c7ba338d5c \
  --include config.json preprocessor_config.json pytorch_model.bin \
    tokenizer_config.json special_tokens_map.json vocab.json added_tokens.json \
  --local-dir /shared/talkshow_assets/wavlm-libri-clean-100h-large

test -s /shared/talkshow_assets/wavlm-libri-clean-100h-large/config.json
test -s /shared/talkshow_assets/wavlm-libri-clean-100h-large/preprocessor_config.json
test -s /shared/talkshow_assets/wavlm-libri-clean-100h-large/pytorch_model.bin
```

这里必须传 Hugging Face snapshot 目录，不能传 Microsoft 原始发布的单个
`WavLM-Large.pt` 文件。训练入口会检查 `config.json`、模型权重和
`preprocessor_config.json` 是否存在。

WavLM Large 接入的具体行为如下：

1. loader 继续读取 16 kHz 单声道原始 waveform；
2. 按该模型的 `preprocessor_config.json` 执行与主项目 processor 一致的
   utterance normalization；
3. `WavLMModel` 产生约 50 FPS、1024 维的 hidden state；
4. `Linear(1024, 256)` 将其映射到 TalkSHOW face hidden size；
5. 使用与主项目 WavLM ETL 相同的线性插值约定，精确对齐到 30 FPS 的
   GT 帧数；
6. WavLM 整体冻结，只训练映射层和 TalkSHOW face head。

整体冻结是为了与主项目的固定 WavLM feature extractor 保持同一表征，也能
显著降低训练显存。该模式统一了 face/body 所用的音频模型，但当前 TalkSHOW
独立训练仍在内部执行一次 WavLM forward。接入主项目时使用新提供的
`TalkSHOWFaceConditionModel`，它会只加载 post-WavLM head，让 face/body
共享主项目已经计算的同一份 hidden state，不会在联合推理中再跑第二次 WavLM。

### 5.2 严格复现 TalkSHOW：Wav2Vec2

默认模式仍使用固定 revision：

```text
facebook/wav2vec2-base-960h
revision: 22aad52d435eb6dbaf354bdad9b0da84ce7d6156
```

如果计算节点不能联网，先在可联网节点下载：

```bash
hf download facebook/wav2vec2-base-960h \
  --revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156 \
  --include config.json model.safetensors \
  --local-dir /shared/talkshow_assets/wav2vec2-base-960h
```

后续可通过旧参数 `--wav2vec-path` 传入该目录；它等价于：

```text
--audio-encoder wav2vec2
--audio-model-path /shared/talkshow_assets/wav2vec2-base-960h
```

计算节点离线运行时可设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

所有本地模型路径都必须对实际计算节点可见。

## 6. 先运行 smoke training

使用 WavLM Large 和修正后的 face loss：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_smoke_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_smoke \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --corrected-face-loss \
  --smoke
```

严格复现 Wav2Vec2 时，把上面的两个音频参数改成：

```text
--audio-encoder wav2vec2
--audio-model-path /shared/talkshow_assets/wav2vec2-base-960h
```

也可以继续只写兼容参数
`--wav2vec-path /shared/talkshow_assets/wav2vec2-base-960h`。

smoke 模式会：

- 训练 1 个 epoch；
- 每个 batch 执行真实的 forward、backward 和 optimizer step；
- 保存 `ckpt-0.pth`；
- 检查 checkpoint 的 epoch、global steps、基本 payload 合约和所有浮点
  tensor；
- checkpoint tensor 中存在 NaN/Inf、checkpoint 缺失或训练步数为零时返回
  失败。

该校验不直接检查每一步的 loss/gradient，也不评价生成质量；它的作用是确认
训练流程完成了至少一个 step，并产出了数值有限、可重新加载的 checkpoint。

成功后应看到类似：

```text
verified checkpoint epoch=0, global_steps=4, ...
TalkSHOW face training completed: .../ckpt-0.pth
```

## 7. 启动完整训练

smoke 通过后再使用完整 split：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_full \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --corrected-face-loss
```

WavLM 模式故意要求显式的本地 `--audio-model-path`，避免训练时悄悄解析到
不同版本。只有默认 Wav2Vec2 模式可以不传模型路径；此时代码会从 Hugging
Face cache/网络解析上面锁定的 revision。

不要从 Wav2Vec2 checkpoint resume WavLM run，反过来也不可以。两种模型的
backbone 参数和 `audio_feature_map` 输入维度不同，应使用不同 `run-id`。

### 7.1 八卡、每卡 batch 32、1000 epoch

下面是本次新增的 WavLM Large 多卡训练配置：

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

launcher 会用 `torchrun --standalone --nproc_per_node 8` 启动一机八进程 DDP。
`--batch-size-per-gpu` 是每个进程的最大 batch，不是八卡总 batch。此配置的
最大全局 batch 为 `8 × 32 = 256`。

这个入口只支持文档指定的 WavLM Large（feature extractor 使用逐时刻
LayerNorm）、`whole_video=true` 和冻结音频骨干；不满足时会直接停止，避免
padding 改变有效帧表征。它没有改变数值精度：模型、输入、loss 和梯度仍为
FP32，没有启用 BF16、FP16、autocast 或 GradScaler。优化器也仍是 SGD，
学习率 0.001、momentum 0.9、gradient clipping 5；WavLM 继续整体冻结。
默认单卡命令仍调用上游 `scripts/train.py`，其行为不受影响。

whole-video clip 长度不同，不能直接使用普通 `DataLoader(batch_size=32)`。
新入口会先按原始音频长度分桶，再在 batch 内右侧补零；WavLM 使用
`attention_mask`，face head 每层重新应用 motion mask，loss 只统计各 clip
的有效帧。checkpoint 保存的是原始 TalkSHOW wrapper state，不会带 DDP 的
`module.` key 前缀。

完整 train split 为 13,687 个 clip；在不丢弃尾 batch 的情况下，八卡配置
每个 epoch 的实际调度为：

| 项目 | 数值 |
|---|---:|
| 前 53 个 step | 每卡 32，共 256 |
| 最后 1 个 step | 每卡 15，共 120 |
| 每 epoch optimizer steps | 54 |
| 1000 epoch optimizer steps | 54,000 |
| 最终 checkpoint | `ckpt-999.pth` |

为了让八个 rank 的最后一步大小一致，尾 batch 会从该 batch 内确定性重复
1 个样本：每个 epoch 实际处理 13,688 个样本，但不会丢掉原始 clip。分桶和
shuffle 会随 epoch 改变。这里的 `batch size 32` 因此是每卡最大值，尾 batch
为每卡 15。

这个训练协议与默认单卡复现并不等价。默认 100 epoch、batch 1 共
1,368,700 次 optimizer update；八卡配置虽然将每个 clip 的训练遍数提高到
约 1000 次，但因为全局 batch 更大，update 数变为 54,000。是否收敛以及最终
质量仍需根据 loss、定期 checkpoint 和下游评估判断，不能只比较 epoch。

建议在正式跑 1000 epoch 前，先用完整 split 做一次 `--epochs 1` 的八卡
资源测试，并使用新的 `run-id`。单卡只占约 2.4 GB 并不能保证每卡 batch 32
一定可行：WavLM 的显存会随 clip 长度增长，batch 又会补齐到桶内最长 clip。
若 H100 上 OOM，先把 `--batch-size-per-gpu` 降到 16 或 8；卡数仍可保持 8。

还要检查 CPU 内存和数据盘吞吐。TalkSHOW 上游 loader 会把数据 eagerly
读入内存，而 `torchrun` 的八个进程各自构建一份 dataset，因此主存占用可能
接近单进程的八倍，启动时也会并发扫描数据。本次修改未改变该 loader 行为，
以免影响既有单卡实验。多卡入口会按 clip 路径稳定排序；开始第一个 step 前，
trainer 还会比对各 rank 的样本数和顺序 SHA-256。不一致时会停止，不会带着
错位的 sampler 继续训练。

### 7.2 从 checkpoint 继续训练

face launcher 支持从完整 epoch checkpoint 继续训练：

```text
--resume-from /path/to/ckpt-N.pth
```

`--epochs` 表示整个 run 的总目标 epoch 数，不是额外增加的 epoch 数。例如，
`ckpt-499.pth` 表示 epoch 499 已经完成；下面的命令会从 epoch 500 开始，
一直训练到 epoch 999：

```bash
RESUME_CKPT=/shared/talkshow_runs/old-run/experiments/.../ckpt-499.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_8gpu_b32_e1000_resume500 \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --num-gpus 8 \
  --batch-size-per-gpu 32 \
  --epochs 1000 \
  --corrected-face-loss \
  --resume-from "$RESUME_CKPT"
```

续训必须保持原 run 的以下选项一致：

- `--audio-encoder` 与 WavLM 模型/预处理方式；
- `--num-gpus`、`--batch-size-per-gpu`、length bucket 和是否丢弃尾 batch；
- 数据集 clip 列表、顺序、音频/动作长度，以及四名 speaker 的顺序；
- seed、optimizer、face head 结构；
- 是否使用 `--corrected-face-loss`。

新 DDP checkpoint 会保存模型、SGD optimizer、`global_steps`、
`samples_seen`、数据布局指纹、训练合约，以及每个 rank 的 Python、NumPy、CPU
Torch 和 CUDA RNG 状态。恢复时会先校验 world size、batch size、
steps/samples per epoch、数据布局指纹和训练合约，再从 `checkpoint epoch + 1`
开始。参数不一致时直接停止，避免把改变训练协议的操作误当成 resume。

这里的数据布局指纹覆盖稳定 clip key、顺序以及音频/动作长度，不会哈希每个
浮点值；因此仍应使用原数据集，不能把它当作完整数据内容校验。
clip key 会先转换为相对 `data_root` 的路径，因此同一 split 可以换到新的挂载
位置；相对目录布局必须保持一致。

本次修改之前生成的大 batch/DDP checkpoint 已经包含 optimizer 和调度计数，
因此仍可通过 `--resume-from` 恢复；但它没有数据指纹、训练合约和各 rank RNG，
launcher 会打印兼容性警告，续训结果不保证与不中断训练逐位一致。新 checkpoint
具备所需 RNG 状态；即便如此，更换 PyTorch/CUDA/cuDNN 版本或 GPU 型号仍可能
带来数值差异。

默认单卡、batch 1 的完整训练 checkpoint 也可以恢复，例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_single_resume50 \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --epochs 100 \
  --corrected-face-loss \
  --resume-from /shared/old-run/experiments/.../ckpt-49.pth
```

单卡路径会严格恢复 wrapper 中的网络和 SGD optimizer，并修正上游
TalkSHOW resume 会重复最后一个 epoch 的问题。旧单卡 checkpoint 没有 RNG
元数据，因此不能承诺逐位等价续训。如果 checkpoint 旁存在旧 run-local
`face.json`，单卡续训会强制当前绝对 `Data.data_root` 及关键模型/loss 配置
一致；旧配置文件缺失时会打印警告，此时数据一致性由用户保证。无论配置文件
是否存在，都会校验
`global_steps == 已完成epoch数 × 当前每epoch batch数`。

resume 还有以下边界：

- 仅支持 epoch 末保存的 `ckpt-*.pth`，不支持从某个 epoch 的 batch 中间恢复；
- `--resume-from` 不能与 `--smoke` 同时使用；
- 必须使用一个新的 `--run-id`。launcher 仍拒绝复用已有 run 目录，从而不会
  覆盖原 checkpoint 和日志；
- `--epochs` 必须大于 checkpoint 的下一 epoch。八卡/DDP 的
  `ckpt-499.pth` 最低可以配置 `--epochs 501`；默认单卡路径还沿用发布版
  保存规则，非 smoke 的目标 epoch 必须是 30 或 50 的倍数，因此同一
  checkpoint 的单卡目标最低应为 550；
- Wav2Vec2 checkpoint 不能恢复到 WavLM run，反过来也不可以；
- checkpoint 必须包含 TalkSHOW wrapper 的网络和 `generator_optim`；只导出的
  face head/weight-only 文件不能用于 resume；
- 兼容范围是本 launcher 或上游 Trainer 正常训练保存的完整 checkpoint。若
  曾手工使用上游有 epoch off-by-one 问题的旧 `--resume` 再保存 checkpoint，
  其 `epoch/global_steps` 可能不一致，新的 fail-closed 校验会拒绝；
- resume checkpoint 只用于续训；如果只想加载网络权重并把 optimizer/epoch
  清零，那属于 finetune/init-from，目前不是该参数的语义。

## 8. 选择哪一种 face loss

### 推荐：修正后的 loss

训练实用模型时建议加入：

```text
--corrected-face-loss
```

模型输出排列为：

```text
jaw[0:3] + expression[0:100]
```

修正版 loss 为：

```text
L = L1(pred_jaw, gt_jaw)
  + MSE(pred_expression, gt_expression)
```

### 严格复现发布代码

如果目标是严格保留 TalkSHOW 发布训练代码的行为，则省略
`--corrected-face-loss`。

原始实现会比较预测和 GT 的前 6 个通道。由于预测的前 6 维是
`jaw[3] + expression[:3]`，而 GT 前 6 维是 `jaw[3] + left_eye[3]`，
这会让前三个 expression 系数额外拟合左眼姿态。该行为为复现而保留，
不建议用于新的实用模型。

## 9. 输出目录与 checkpoint

以以下参数为例：

```text
--output-dir /shared/talkshow_runs
--run-id face_full
```

输出目录大致为：

```text
/shared/talkshow_runs/face_full/
├── config/
│   └── face.json
├── cache/
└── experiments/
    └── YYYY-MM-DD-talkshow_face_face_full-face/
        ├── train.log
        ├── face.json
        ├── ckpt-29.pth
        ├── ckpt-49.pth
        └── ckpt-99.pth
```

默认 100 epoch 训练最后的 checkpoint 是：

```text
ckpt-99.pth
```

八卡 1000 epoch 命令最后的 checkpoint 是 `ckpt-999.pth`。它仍在第 30
个 epoch 保存 `ckpt-29.pth`，之后每 50 个 epoch 保存一次，并保证保存最终
epoch。

run-local `config/face.json` 会保存
`Model.audio_encoder`、`audio_model_path`、`audio_model_source`、
`audio_model_revision`、`audio_hidden_size`、`audio_do_normalize` 和
`freeze_audio_encoder`，并记录音频 `config.json`/preprocessor 的规范化
SHA-256、WavLM 权重文件（含 index 引用分片）的逐文件 SHA-256、face loss
模式和父 resume checkpoint。评估时应使用该 run 随附的配置，不能只看
checkpoint 文件名猜测音频骨干。

checkpoint 外层包含：

```text
generator
epoch
global_steps
```

其中 `generator` 内还包含网络权重及 SGD optimizer state。
大 batch/DDP checkpoint 还会记录 `samples_seen` 和 `distributed` 元数据，
其中包括 `steps_per_epoch` 与 `samples_per_epoch`。新生成的 DDP checkpoint
还包含 `dataset_fingerprint`、`resume_contract` 和 `rng_states`；已有的
`generator/epoch/global_steps` 合约保持不变，主项目仍可从同一个 checkpoint
提取 TalkSHOW face head。

当前从头启动的新 run 使用 `resume_contract schema 2`，把上述 WavLM 文件摘要
一起写入 checkpoint。旧 schema 1 checkpoint 仍能续训；恢复时会明确警告
旧文件无法证明当时的 WavLM 权重，其他训练契约保持严格比较，续训 lineage
继续保存为 schema 1，不会用当前目录摘要冒充旧冻结权重的来源。

## 10. 已知限制

### 10.1 通过专用适配器接入主项目

TalkSHOW face checkpoint：

```text
输入：raw waveform + 4-person one-hot
骨干：Wav2Vec2 或冻结的 WavLM Large
输出：jaw 3D + expression 100D
```

它与 `SimpleSpeechModel` 的输入、架构、输出和 state-dict key 不同，
因此主项目提供
`TalkSHOWFaceConditionModel`，可以跳过 checkpoint 中冻结的 WavLM，只加载
post-WavLM face head，并向扩散模型返回后 100 维 expression。完整接入方法见
主项目的 [TalkSHOW pipeline](../../configs/talkshow_profile/README_zh.md)。

### 10.2 当前没有 validation 和 best-checkpoint

上游 Trainer 只遍历 train split，没有 early stopping、验证指标或
best-checkpoint 选择。`ckpt-99.pth`（默认训练）或 `ckpt-999.pth`（上面的
八卡配置）只是最后一个训练 checkpoint，不一定是泛化效果最好的 checkpoint。

### 10.3 resume 仅支持完整 epoch 边界

launcher 已支持 `--resume-from`，但 checkpoint 仍然只在 epoch 末保存，不含
batch 内游标、未完成梯度或 DataLoader 预取状态。因此不能从 epoch 中间恢复。
完整用法与旧 checkpoint 的兼容边界见
[7.2 从 checkpoint 继续训练](#72-从-checkpoint-继续训练)。

### 10.4 仅支持四名已知说话人

身份条件固定为四类。若增加新人物，需要同时修改 speaker ID 映射、
`num_classes` 和相应 checkpoint 结构。

### 10.5 非因果、非流式

两种模式都使用双向语音 Transformer 上下文并处理完整音频，不适合直接用于
在线流式生成。

## 11. 常见问题

### `unexpected TalkSHOW commit`

不要复用任意的上游目录。换一个不存在的新目录重新运行：

```bash
bash reproductions/talkshow/bootstrap_upstream.sh \
  /local-ssd/talkshow_upstream_new
```

### `patch marker does not match this launcher`

说明上游目录不是由当前版本的 `bootstrap_upstream.sh` 生成，或者补丁版本
不同。重新创建一个干净 checkout。

### `refusing to reuse an existing run directory`

每次实验使用新的 `--run-id`，例如：

```text
face_smoke_v2
face_corrected_seed1
```

### Wav2Vec2 下载失败

提前下载权重并通过 `--wav2vec-path` 使用共享存储上的绝对路径，同时设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### WavLM 模型目录校验失败

`--audio-model-path` 必须指向完整的 Hugging Face snapshot 目录，至少包含：

```text
config.json
preprocessor_config.json
pytorch_model.bin 或 model.safetensors（也支持分片权重）
```

如果手中只有原始 `WavLM-Large.pt`，需要先转换成 Transformers 格式；不能
直接把该文件路径传给训练脚本。

### 训练前显存不足

默认路径使用 batch size 1；新增多卡路径可把每卡 batch 从 32 降到 16、8
或 1。WavLM Large 的 Transformer 在约 50 FPS 上运行，显存和计算量明显
高于原来的 Wav2Vec2-base；本实现已冻结 WavLM 以省去梯度和 optimizer
state，但仍应先用完整 split 的单 epoch 运行确认最长分桶能够执行。

### 初始化很慢或占用较多内存

上游 loader 会扫描并加载大量 motion/audio。兼容补丁默认禁用了原始的大型
pickle cache，因此新 run 第一次启动仍可能较慢，并占用较多主存。

## 12. 最小命令清单

以下清单假设 SHOW 数据已经解压，并且第 3 节所述 Python/PyTorch 环境已经
准备好。

```bash
# 1. 创建 smoke split
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_smoke_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --limit-per-speaker-split 1

# 2. 创建完整 split
python reproductions/talkshow/scripts/prepare_split_view.py \
  --source-root /path/to/ExpressiveWholeBodyDatasetReleaseV1.0 \
  --output-root /local-ssd/talkshow_split \
  --split-file tools/show/split_more_than_2s.pkl \
  --expected-missing 55

# 3. 创建 patched upstream
bash reproductions/talkshow/bootstrap_upstream.sh \
  /local-ssd/talkshow_upstream

# 4. 下载与主工程一致的 WavLM Large
hf download patrickvonplaten/wavlm-libri-clean-100h-large \
  --revision e70e3a062ec399c46008ee55d1fb52c7ba338d5c \
  --include config.json preprocessor_config.json pytorch_model.bin \
    tokenizer_config.json special_tokens_map.json vocab.json added_tokens.json \
  --local-dir /shared/talkshow_assets/wavlm-libri-clean-100h-large

# 5. 运行 smoke training
CUDA_VISIBLE_DEVICES=0 \
bash reproductions/talkshow/train_face.sh \
  --upstream-dir /local-ssd/talkshow_upstream \
  --data-root /local-ssd/talkshow_smoke_split \
  --output-dir /shared/talkshow_runs \
  --run-id face_wavlm_large_smoke \
  --audio-encoder wavlm \
  --audio-model-path /shared/talkshow_assets/wavlm-libri-clean-100h-large \
  --corrected-face-loss \
  --smoke

# 6. 运行八卡完整训练
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

若要做论文复现，把第 4 步改为第 5.2 节的 Wav2Vec2 权重，并把两个训练
命令中的音频参数换成 `--wav2vec-path /path/to/wav2vec2-base-960h`。
