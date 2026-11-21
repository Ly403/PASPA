# TalkSHOW 官方协议离线评估

本说明用于评估主项目在 TalkSHOW test split 上生成的完整 SMPL-X 结果。执行脚本
是：

```text
scripts/talkshow_official_eval.py
```

它参考并固定到 TalkSHOW 官方仓库版本：

```text
9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b
```

该脚本是新增的独立离线路径，不替换 `distributed_emage`，也不修改 BEAT2 的
训练、测试或指标逻辑。

## 1. 先区分两种评估

| 入口 | 用途 | 是否为 TalkSHOW 原始协议 |
|---|---|---|
| `metrics=distributed_emage` | 训练期间快速观察趋势 | 否，使用 EMAGE 指标 |
| `scripts/talkshow_official_eval.py` | 最终离线评估 | 公式和预处理对齐；生成 provenance 需另行审计 |

最终报告请使用 `scripts/talkshow_official_eval.py`。

## 2. 为什么必须准备两份预测

TalkSHOW 原始 `scripts/test_body.py` 对每条音频一次生成两个 body 样本：

```text
B = 2
```

官方 diversity 是两个样本在 SMPL-X joint 空间中的无偏方差。单个
`res_*.npz` 只能得到 B=1，`torch.var` 会产生无效值，所以新 evaluator
强制要求按顺序传两次 `--pred-dir`：

```text
第一个 --pred-dir = sample0
第二个 --pred-dir = sample1
```

两份预测必须来自：

- 同一个主扩散 checkpoint；
- 同一个 TalkSHOW profile；
- 同一个 test LMDB；
- 同一套 face checkpoint、WavLM、CFG 和 sampler 配置；
- 两个不同的随机种子。

官方 BC 使用 sample0 的 body；face 指标也只使用 sample0。官方
`feat_dist` 则保留了上游固有的 batch-major 展开和前 N 行截断配对：多 clip
时并不是简单的“只取 sample0”，而且结果依赖 clip 顺序。传原始 split view
时，脚本按官方 `test_body.sh` 的 speaker 顺序及该 split view 上
`os.listdir` 的 video/sequence 顺序遍历，以复现这一行为。参数和 clip 顺序
都具有语义，脚本不会自行交换 sample0/sample1。

## 3. 准备依赖

评估环境需能导入 TalkSHOW 原仓库及其依赖，包括 PyTorch、SMPL-X、
librosa、torchaudio、SciPy 和 TalkSHOW 自身模块。还需要两项官方资产：

1. TalkSHOW 预训练包中的 `experiments/feature_extractor.pth`；
2. neutral SMPL-X 模型 `SMPLX_NEUTRAL.npz`。

推荐目录形态：

```text
/path/to/TalkSHOW/
├── data_utils/
├── evaluation/
├── nets/
├── scripts/
├── experiments/
│   └── feature_extractor.pth
└── visualise/
    └── smplx/
        └── SMPLX_NEUTRAL.npz
```

如果 SMPL-X 不放在 TalkSHOW 的 `visualise/` 下，评估时显式传
`--smplx-path`。该参数可以指向包含 `smplx/SMPLX_NEUTRAL.npz` 的目录，也
可以直接指向 neutral NPZ。

脚本默认校验定义评估协议及 FGD encoder 的核心 TalkSHOW 源文件。CRLF 和 LF
会先归一化，所以 Windows 换行不会造成误报。源码与固定版本不同会直接停止；
只有人工审查过差异后，才应使用 `--allow-unpinned-talkshow`。输出 JSON 会
记录源码、feature extractor 和 SMPL-X neutral asset 的 SHA-256。

## 4. 从同一 checkpoint 生成两个独立 test 输出

以下命令仍使用对应 profile 的 canonical launcher。正式评估时保持训练用的
所有环境变量不变。

先设置 checkpoint 和 profile：

```bash
export TALKSHOW_PROFILE=block64_seq256
export TALKSHOW_EVAL_CKPT=/absolute/path/to/run/output/checkpoints/epoch_X_step_Y.pth
```

生成 sample0：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
RUN_NAME=talkshow_eval_sample0 \
TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from="$TALKSHOW_EVAL_CKPT" \
  train.seed=1001 \
  train.restore_rng_state=false \
  _train=false _validation=false _test=true
```

生成 sample1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_PROCESSES=8 \
RUN_NAME=talkshow_eval_sample1 \
TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from="$TALKSHOW_EVAL_CKPT" \
  train.seed=2002 \
  train.restore_rng_state=false \
  _train=false _validation=false _test=true
```

纯 test 使用 `model.load_from`，只导入模型权重，不进入训练 resume 分支。
`train.restore_rng_state=false` 进一步明确禁止从 checkpoint 恢复训练 RNG；
两个 seed 因而对应两份独立采样。训练续跑仍使用顶层 `resume_from`，其默认
行为没有变化。

`TALKSHOW_MODE=test` 让这两次纯 test 只检查和实例化当前 profile 的 test
LMDB，不读取 train/validation LMDB，也不构造训练 optimizer/scheduler。未设置
该变量时，canonical launcher 仍保持原训练行为。

`model.load_from` 测试作为独立运行创建输出目录，不会在原实验中恢复 epoch、
step 或训练状态。以两次日志中实际打印的 `work_dir` 为准，分别记录：

```bash
export SAMPLE0_OUTPUT=/absolute/path/to/sample0/work_dir
export SAMPLE1_OUTPUT=/absolute/path/to/sample1/work_dir
```

两个 evaluator 输入目录应为：

```text
$SAMPLE0_OUTPUT/npz/test
$SAMPLE1_OUTPUT/npz/test
```

每个目录内必须是一条 source clip 对应一份：

```text
res_<speaker>__<sequence>.npz
gt_<speaker>__<sequence>.npz
```

不能传 validation 窗口、带 `rank*`/`bs*` 前缀的旧输出，也不能传旧版转换器
生成的 body-isolated B=1 文件；这类文件没有完整 face GT，再次评估还会重复
执行 body 隔离变换。

唯一的诊断例外是通过统一入口设置 `TALKSHOW_MODE=validation`，并显式把
`TALKSHOW_VALIDATION_LMDB` 指向对应档位的完整 test LMDB。该模式把文件写到
`npz/validation`，但读取的是完整 test clip，而不是 validation 窗口。为了比较
缓存 WavLM 路径与纯 test 在线 WavLM 路径，可以将两个不同 seed 产生的
`npz/validation` 目录显式作为两次 `--pred-dir`，并将 sample0 的同一目录作为
`--gt-dir`。这种结果只用于路径诊断；正式最终报告仍推荐使用两次纯 test 的
`npz/test`。

## 5. 选择音频来源

### 5.1 推荐：原始 TalkSHOW test WAV

TalkSHOW 官方脚本直接对原始 clip WAV 提取 onset。为了尽量做到端到端一致，
推荐把生成 LMDB 时使用的官方 split view 作为：

```bash
export SHOW_EVAL_AUDIO_ROOT=/absolute/path/to/talkshow_split_view
```

脚本根据 NPZ 的 `<speaker>__<sequence>` 自动解析：

```text
<root>/<speaker>/<video>/test/<sequence>/<sequence>.wav
```

每个 clip 必须且只能匹配一个 WAV。

默认还会比较 prediction clip 集与 split view 中官方 dataloader 实际可读取的
完整 test clip 集。固定 split manifest 的完整 view 应有 1708 条带 WAV/PKL
的 test entry；之后还会照抄官方 `MultiVidData.get_dataset()` 的规则，排除
motion 少于 88 帧的 entry。`split_view_summary.json` 的 manifest hash、
`missing_count=55`、无 smoke limit、原始 entry 数、短片段列表及最终可评估
数量都会写入报告。

例如此前发现的 62 帧异常片段会被官方 DataLoader 排除，新 evaluator 也会
自动排除；它不会被误报成主模型漏生成的 subset。

如果 test LMDB 因此前明确审计过的坏样本策略少了少量 clip，可以显式追加：

```text
--allow-source-subset
```

此时缺失的全部 clip ID 会写入 JSON，报告明确是 subset，不能当成完整官方
test 数值。默认不允许静默漏 clip。

### 5.2 兼容模式：主项目保存的对齐 WAV

若原始 split view 不可用，可以改传：

```text
$SAMPLE0_OUTPUT/wav/test
```

其中包含 `gt_sr16000_<speaker>__<sequence>.wav`。这些 WAV 已按 motion
长度裁剪或补零，脚本会严格检查单声道、16 kHz 和
`16000 * T // 30` 个采样点。

这一模式仍使用官方 onset/BC 代码，但若原始音频和动作存在不超过 ETL 阈值的
尾部长度差，最后几帧的 onset 可能与直接读取原始 TalkSHOW WAV 略有不同。
此外该模式只能按 clip ID 字典序遍历，不能复现上游 `feat_dist` 依赖的原始
DataLoader 顺序；因此 `feat_dist` 也不保证与官方数值一致。论文最终数值必须
优先使用 `--source-audio-root`。

## 6. 先做无 GPU 输入预检

推荐先执行轻量预检。它只依赖 Python 标准库和 NumPy，不导入 torch：

```bash
python scripts/talkshow_official_eval.py \
  --pred-dir "$SAMPLE0_OUTPUT/npz/test" \
  --pred-dir "$SAMPLE1_OUTPUT/npz/test" \
  --gt-dir "$SAMPLE0_OUTPUT/npz/test" \
  --source-audio-root "$SHOW_EVAL_AUDIO_ROOT" \
  --talkshow-root /absolute/path/to/TalkSHOW \
  --validate-input-only
```

若只能使用保存的对齐音频，把：

```text
--source-audio-root "$SHOW_EVAL_AUDIO_ROOT"
```

替换为：

```text
--audio-dir "$SAMPLE0_OUTPUT/wav/test"
```

预检会 fail closed 地检查：

- `--pred-dir` 恰好两份且不是同一目录；
- 两份 prediction、GT 和音频的 clip 集完全相同；
- 使用原始 split view 时 prediction 默认覆盖完整 source test 集；
- `poses=[T,165]`、`expressions=[T,100]`、`betas=[300]`；
- `T >= 88`，与官方 body/face 配置的 `generate_length=88` 过滤一致；
- 所有参数有限且 FPS 严格为 30；
- 两份 prediction 与 GT 的 T 完全一致，不做 `min(T)` 截短；
- 每个 prediction 的 betas 与 GT 完全一致；
- 第二份输出中的本地 GT 若存在，必须与 canonical GT 完全一致；
- GT 必须保留非零 expression，防止误传预处理后的 body-only 文件；
- 对齐音频必须和动作长度精确一致。

若两份 body 在全部 clip 上完全相同，预检会明确报警。脚本仍可计算，diversity
会是 0；此时应先检查是否忘记设置不同 seed 或忘记关闭 checkpoint RNG 恢复。

NPZ 本身没有内嵌 checkpoint、Hydra config 和 seed 的可信关联，因此 evaluator
无法仅凭两个目录证明它们“同 checkpoint、不同 seed”。JSON 会明确记录
`generation_provenance_verified=false`；正式归档时必须同时保存第 4 节两次
命令、日志和 resolved config。

## 7. 运行完整官方评估

使用原始 TalkSHOW WAV：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/talkshow_official_eval.py \
  --pred-dir "$SAMPLE0_OUTPUT/npz/test" \
  --pred-dir "$SAMPLE1_OUTPUT/npz/test" \
  --gt-dir "$SAMPLE0_OUTPUT/npz/test" \
  --source-audio-root "$SHOW_EVAL_AUDIO_ROOT" \
  --talkshow-root /absolute/path/to/TalkSHOW \
  --feature-extractor /absolute/path/to/feature_extractor.pth \
  --smplx-path /absolute/path/to/TalkSHOW/visualise \
  --device cuda:0 \
  --output "$SAMPLE0_OUTPUT/talkshow_official_metrics.json"
```

使用主项目保存的对齐 WAV：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/talkshow_official_eval.py \
  --pred-dir "$SAMPLE0_OUTPUT/npz/test" \
  --pred-dir "$SAMPLE1_OUTPUT/npz/test" \
  --gt-dir "$SAMPLE0_OUTPUT/npz/test" \
  --audio-dir "$SAMPLE0_OUTPUT/wav/test" \
  --talkshow-root /absolute/path/to/TalkSHOW \
  --feature-extractor /absolute/path/to/feature_extractor.pth \
  --smplx-path /absolute/path/to/TalkSHOW/visualise \
  --device cuda:0 \
  --output "$SAMPLE0_OUTPUT/talkshow_official_metrics.json"
```

若已经固定了 feature extractor 文件，还可以增加：

```text
--expected-feature-extractor-sha256 <64位SHA256>
--expected-smplx-sha256 <64位SHA256>
```

评估器本身是单进程、单 GPU 的离线指标程序。两次主模型 test 可以继续使用
8 卡；最终 SMPL-X/FGD 汇总只需选择一张卡。官方 `FGD.py` 会在 CPU 内存中
保留全部 clip 的 GT joints、两个 generated joints 和音频 onset；1708 条完整
test 粗略需要约 5 GB 主存，最终 BC 汇总也会花一些时间。这是上游实现的原始
行为，不是脚本泄漏。

## 8. 输出指标

JSON 的 `metrics` 包含：

| 字段 | 官方含义 | 趋势 |
|---|---|---|
| `fgd` | body autoencoder latent 的 Fréchet distance | 越低越好 |
| `feat_dist` | latent feature MAE | 越低越好 |
| `bc` | audio onset 与 motion beat consistency | 越高越好 |
| `body_lvd` | 前 22 个 joints 的速度差 | 越低越好 |
| `body_error` | 全部 SMPL-X joints 的位置误差 | 越低越好 |
| `diverse` | 两个 body 样本的 joint 方差 | 表示多样性，不是单调准确率 |
| `jaw_l1` | jaw joints 误差 | 越低越好 |
| `landmark_l1` | face landmarks 误差 | 越低越好 |
| `face_lvd` | jaw + face landmarks 的速度差 | 越低越好 |

官方 `test_body.py` 中 MAAC 的计算和打印均被注释，因此严格报告不输出 MAAC。

报告还包含：

- clip 数和总帧数；
- sample0/sample1 的绝对路径与顺序；
- 两个推理 work dir 中可发现的 resolved/Hydra config 路径与 hash；
- 音频来源协议；
- TalkSHOW commit 和逐文件归一化 hash；
- feature extractor 与 SMPL-X neutral asset 的 SHA-256；
- clip manifest hash；
- 完整 split 覆盖、原始/对齐音频和资产 expected-hash 的 alignment audit；
- evaluator 自身 hash、主仓库 commit 和关键 Python 包版本；
- 输入警告。

## 9. 与官方代码对齐的具体位置

评估器保留以下原始行为：

1. 将主项目 `global/body/jaw/eyes/hands + expression` 重排成 TalkSHOW 的
   `jaw/eyes/global/body/hands + expression`；
2. body 用官方 `c_index_3d` 取 129 维；
3. FGD 接收 `[2,T,129]` prediction 和 `[1,T,265]` raw GT；
4. body joint 指标调用官方 `part2full`、`poses2pred` 和 `get_joints`；
5. body 的 jaw/expression 清零，并使用官方固定 lower-body pose；
6. face 只保留 jaw/expression，其余 162 个 rotation 通道清零；
7. body/face loss 按完整 source clip 等权平均；
8. 在 source split view 中排除少于 88 帧的 clip，复现官方 DataLoader；
9. BC 调用官方
   `get_mfcc_ta(..., fps=30, sr=16000, encoder_choice="onset")`；
10. translation 完全忽略。

`c_index_3d` 与 `part2full/poses2pred` 之间存在官方代码自身留下的不对称索引
行为。新 evaluator 为保证数值一致而原样保留，没有“修正”它。

## 10. 本机能验证什么

仓库内的轻量测试：

```bash
python -m unittest tests.test_talkshow_official_eval -v
```

会验证通道重排、129-D body 索引、body/face 隔离、B=2 输入合同、clip
配对、严格帧长、原始音频解析和源码换行 hash。完整数值评估仍需训练服务器上的
PyTorch、官方 feature extractor 和 SMPL-X neutral model；本地缺少这些资产
时不能伪造端到端数值。

## 11. 批量遍历主扩散 checkpoint

若要对同一 profile 的 checkpoint 目录逐个生成 B=2 输出、运行本页官方评估并
按最低 FGD 排序，使用：

```bash
python scripts/eval_talkshow_pth_dir.py \
  /absolute/path/to/run/output/checkpoints \
  --profile block64_seq256 \
  --talkshow-root /absolute/path/to/TalkSHOW \
  --feature-extractor /absolute/path/to/experiments/feature_extractor.pth \
  --source-audio-root /absolute/path/to/talkshow_split_view \
  --smplx-path /absolute/path/to/TalkSHOW/visualise \
  --parallel-workers 4 \
  --gpus-per-worker 1 \
  --cuda-visible-devices 0,1,2,3 \
  --num-workers 2 \
  --epoch-range 500 800
```

该入口复用本页同一个 `talkshow_official_eval.py`，不是 BEAT 的
`emage_eval.py`。它只加载当前 profile 的 test LMDB，每个 checkpoint 使用
两个不同 seed。每个 GPU worker 会连续完成两次采样和该 checkpoint 的官方
评估，主线程只汇总 worker 返回的结果，并把全部指标、错误和最佳 checkpoint
写入增量 CSV/JSON。

`--epoch-range START END` 为包含首尾的闭区间；省略时仍评估目录下的全部
合法 checkpoint。单卡 worker 会自动注入 `LOCAL_RANK/RANK/WORLD_SIZE` 等
单进程分布式变量。每次 test 完成后还会写 completion marker；若
Accelerate/DDP 未在默认 10 秒宽限期内退出，watchdog 会清理该 worker 的整个
进程组，再继续官方指标计算。详细参数与 CSV 的
`watchdog_terminated_samples` 字段见下方完整说明。
完整参数与输出目录说明见
`scripts/eval_talkshow_pth_dir.py` 文件开头的用法说明。
