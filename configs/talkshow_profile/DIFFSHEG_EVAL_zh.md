# TalkSHOW：DiffSHEG-SHOW 协议评估

本说明用于把主项目在 TalkSHOW test split 上生成的完整 NPZ，转换到
DiffSHEG 的 SHOW 表达，并计算论文表格中的：

```text
FMD / FED / expression diversity / FGD / BA / PCM / gesture diversity
```

入口是：

```text
scripts/diffsheg_show_eval.py
```

它是独立离线 evaluator，不会修改训练期间的 `distributed_emage`、现有
`scripts/emage_eval.py`、TalkSHOW 官方 evaluator，也不会影响 BEAT2。

参考源：

- [DiffSHEG 官方仓库](https://github.com/JeremyCJM/DiffSHEG)；
- [DiffSHEG CVPR 2024 论文](https://arxiv.org/abs/2401.04747)。

## 1. 先说明对齐边界

DiffSHEG 公共仓库公开了以下关键实现：

- `datasets/show.py` 中 SHOW 的 129 维 gesture、103 维 expression 映射和
  `talkshow_mean_std.npy` 标准化；
- `models/motion_autoencoder.py` 中 FMD/FED/FGD 使用的 88 帧
  `HalfEmbeddingNet`；
- `datasets/data_tools.py` 中 Fréchet distance 公式；
- `trainers/ddpm_show_trainer.py` 中 PCM/PCK、diversity 和在线 FGD 逻辑；
- SHOW 生成配置 `n_poses=88`、`overlap_len=10`；
- 论文 Appendix E 对 Fréchet 指标明确要求使用 non-overlapping clips。

但是，仓库测试结束后调用的
`0_BEAT_ori/codes/audio2pose/test_talkshow_ddpm.py` 没有公开。因此本脚本是
根据上述公开组件重建的协议，不声称与缺失文件逐行一致。输出 JSON 固定写入：

```text
protocol.status = reconstructed_from_public_components
public_offline_entry_available = false
```

当前明确采用：

- 88 帧窗口；
- stride 88，即评估窗口互不重叠；生成阶段的 10 帧 FOPPAS overlap 不带入
  evaluator；
- 不足 88 帧的尾部不进入 AE 和 diversity；PCM 按论文的 `T × J` 定义对每条
  完整 clip 的所有帧各计算一次；
- 默认按 canonical clip ID 字典序遍历；
- AE 全程 FP32，不启用 autocast；
- diversity 按连续 50 个窗口分组；最后不足 50 但至少 2 个的组仍参与，单独
  剩余 1 个窗口时丢弃并记录。

这些选择都会写进 JSON。正式论文对比前，最稳妥的做法是把 DiffSHEG 发布的
SHOW 预测也转换成同样的 canonical NPZ，再由本脚本同时重算双方，而不是把
本脚本的单次结果直接与论文表格小数硬拼。

## 2. 与仓库其他 evaluator 的关系

| 入口 | 指标空间 | 用途 |
|---|---|---|
| `distributed_emage` | EMAGE | 训练期间快速观察 |
| `scripts/emage_eval.py --dataset talkshow` | EMAGE | 与使用同一 EMAGE evaluator 的方法比较 |
| `scripts/talkshow_official_eval.py` | TalkSHOW 原论文 body/face 协议 | 与 TalkSHOW/ReCoM 的对应协议比较 |
| `scripts/diffsheg_show_eval.py` | DiffSHEG 的 SHOW AE/参数空间 | 与 DiffSHEG 协议比较 |

四个入口中的 `FGD` 不是同一特征提取器，数值尺度也不相同。尤其
DiffSHEG 表格中约 `0.00x` 的 FGD，不能与 TalkSHOW 官方 evaluator 中几十量级
的 FGD 比大小。

## 3. 准备 DiffSHEG 资产

建议固定到本脚本调研时使用的公共 commit：

```bash
git clone https://github.com/JeremyCJM/DiffSHEG.git /absolute/path/to/DiffSHEG
cd /absolute/path/to/DiffSHEG
git checkout 3ebf3058f48cba3da9146afb7623e9ec1ab9e9a5
```

解压仓库自带的统计量：

```bash
cd /absolute/path/to/DiffSHEG/assets
tar -xzf data.tar.gz
mv data ..
```

确认文件存在：

```text
/absolute/path/to/DiffSHEG/data/SHOW/talkshow_mean_std.npy
```

还需要三套 SHOW autoencoder 权重：

```text
data/SHOW/ae_weights/
├── gesture.pth.tar
├── expression.pth.tar
└── gesture_expression.pth.tar
```

DiffSHEG 作者在
[issue #14](https://github.com/JeremyCJM/DiffSHEG/issues/14) 讨论中提供的
AE 压缩包名为 `diffsheg_ae_weights.7z`。对应下载链接是：

```text
https://drive.google.com/file/d/1Wm2WMlacwStFaciCh7UlhQeyA3E2yEnj/view
```

解压后把上述三个文件放到 `data/SHOW/ae_weights/`。脚本会按文件内容计算并在
结果 JSON 中保存 SHA-256；比较不同模型时必须使用相同的三份权重。

如果不想采用上述目录，也可以在运行时分别传：

```text
--stats /path/to/talkshow_mean_std.npy
--weights-dir /path/to/ae_weights
```

此时不需要 `--diffsheg-root`。

### 3.1 生成官方 DiffSHEG 推理缺少的 test LMDB

DiffSHEG 的 `runner.py --mode test_arbitrary_len` 固定读取下面两份缓存：

```text
<DiffSHEG>/data/SHOW/cached_data/talkshow_test_cache
<DiffSHEG>/data/SHOW/cached_aud_hubert/test/hubert_large_ls960_ft
```

公共仓库没有提供这两份缓存，也没有公开原始 SHOW cache builder。本项目新增
`scripts/prepare_diffsheg_show_lmdb.py`，可以从已经生成的非破坏性 TalkSHOW
split view 同时构造两份目录型 LMDB。它不会读取本项目的 64/128/256 训练
LMDB；test 中每个源 clip 都保持为一条完整的变长 record，供 DiffSHEG 自己的
`test_arbitrary_len` 再切成 88 帧窗口。

先确认：

```text
<SHOW_SPLIT_ROOT>/<speaker>/<video>/test/<clip>/<clip>.pkl
<SHOW_SPLIT_ROOT>/<speaker>/<video>/test/<clip>/<clip>.wav
<DiffSHEG>/data/SHOW/talkshow_mean_std.npy
```

然后在本项目根目录执行。建议使用 DiffSHEG 自己的 legacy 环境，因为其 reader
依赖 `pyarrow==3.0.0` 的 `pyarrow.deserialize`：

```bash
conda activate diffsheg

export SHOW_SPLIT_ROOT=/absolute/path/to/talkshow_split_view
export DIFFSHEG_ROOT=/absolute/path/to/DiffSHEG
export HUBERT_LARGE_DIR=/absolute/path/to/facebook-hubert-large-ls960-ft

python scripts/prepare_diffsheg_show_lmdb.py \
  --split-view-root "$SHOW_SPLIT_ROOT" \
  --diffsheg-root "$DIFFSHEG_ROOT" \
  --hubert-model "$HUBERT_LARGE_DIR" \
  --local-files-only \
  --device cuda:0 \
  --require-split-summary \
  --max-av-duration-delta-frames 3 \
  --invalid-clip-policy skip
```

DiffSHEG 同时固定了 `lmdb==0.96`，但其公开
`datasets/show.py` 使用了只在新版 python-lmdb 中才允许省略 `db` 参数的
`txn.stat()`。如果启动官方推理时报
`stat() missing 1 required positional argument: 'db'`，将该文件中的：

```python
with self.lmdb_env.begin() as txn:
    self.n_samples = txn.stat()["entries"]
```

改为：

```python
self.n_samples = self.lmdb_env.stat()["entries"]
```

`Environment.stat()` 在 `lmdb==0.96` 和新版中都会读取当前环境的 unnamed main
database，正是这两份缓存使用的数据库。不要为了绕过该错误改 LMDB 内容或重新
编号 key。本项目的 prepare 脚本已使用这个兼容写法。

若允许 Hugging Face 在线下载，可不传 `--hubert-model` 和
`--local-files-only`，默认模型就是 `facebook/hubert-large-ls960-ft`。这里
必须是 hidden size 1024 的 **HuBERT Large**，不能换成本项目训练使用的 WavLM
Large，也不能使用 HuBERT Base；这是为了匹配 DiffSHEG 已发布 checkpoint 的
输入层，不会改变本项目自身的 WavLM 数据流程。

命令会生成：

```text
<DiffSHEG>/data/SHOW/
├── cached_data/talkshow_test_cache/
├── cached_aud_hubert/test/hubert_large_ls960_ft/
└── diffsheg_show_test_cache_summary.json
```

summary 保存五位 LMDB key 到原始 clip 的完整映射、speaker ID、帧数、HuBERT
长度、跳过原因、统计量/模型元数据哈希和预处理参数。完成后脚本还会重新打开
两份缓存，检查 entry 数、连续 key、首尾 tuple schema，以及两边 key 是否严格
配对。

TalkSHOW 原生身份在这份 **DiffSHEG 专用缓存**中必须保留为：

```text
oliver=20, chemistry=21, seth=22, conan=23
```

因为 DiffSHEG trainer 会在网络前执行 `speaker - 20`。这与本项目主干 LMDB
使用的 raw PID 21--24 是两个隔离的数据契约，不需要迁移本项目现有 LMDB，也
不会影响 BEAT2。

`--invalid-clip-policy skip` 会跳过已知的坏音画长度样本，并在最后的 summary
中逐条列出；它必须和显式阈值一起使用。如果希望发现一个坏样本就停止，改成：

```text
--max-av-duration-delta-frames 3 --invalid-clip-policy error
```

脚本使用临时目录构建并在校验通过后才发布，且不会覆盖已有缓存。如果目标路径
已有旧缓存或失败运行留下的目录，请先人工确认内容，再将其移动到备份位置后
重跑。

需要明确的是：LMDB tuple、pose 顺序、ID 和 HuBERT 分段严格照 DiffSHEG 公共
读取/特征代码实现；但作者没有公开原始 cache builder。因此 30 FPS mel 的
时间步是依据其公开 18 kHz mel 配方重建为 600-sample hop，并会在 summary 中
标记为 reconstruction。该缓存可解决官方 checkpoint 无法启动的问题，但不能
宣称其经典 mel 数值与作者私有缓存逐元素相同。

## 4. BA 额外依赖

完整论文指标还包含 BA。它需要：

- TalkSHOW 原仓库及其 Python 环境；
- neutral SMPL-X 模型；
- 每条 test clip 的音频。

推荐目录：

```text
/absolute/path/to/TalkSHOW/
├── data_utils/
├── evaluation/
├── visualise/
│   └── smplx/
│       └── SMPLX_NEUTRAL.npz
└── ...
```

默认 `--smplx-path` 是 `<talkshow-root>/visualise`。如果模型在别处，显式传：

```text
--smplx-path /absolute/path/containing/smplx
```

BA 复用 TalkSHOW 的
`evaluation.FGD.EmbeddingSpaceEvaluator.get_BCscore`。脚本先把主项目的
165 维 pose 转成 TalkSHOW 顺序，再隔离 DiffSHEG 的 129 维 gesture、补入
TalkSHOW 固定下半身、通过 SMPL-X 得到 joints，最后与音频 onset 对齐。

如果所有预测都没有可检测的 motion beat，原实现的分母会为 0。新入口会明确
停止并报告 BA undefined，不会偷偷把它改成 0。

## 5. 先生成一份完整 test 输出

DiffSHEG 的 diversity 是不同输入序列之间的 pairwise distance，不是
TalkSHOW 官方协议中“同一音频生成两个样本”的 variance。因此本 evaluator
只需要一份 test 输出，不要传两次 `--pred-dir`。

三档模型分别使用对应 test launcher。例：64/256 档：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TALKSHOW_PROFILE=block64_seq256 TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/checkpoint.pth train.seed=1001
```

32/128 档：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TALKSHOW_PROFILE=block32_seq128 TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/checkpoint.pth train.seed=1001
```

16/64 档：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TALKSHOW_PROFILE=block16_seq64 TALKSHOW_MODE=test \
bash train_scripts/run_talkshow.sh \
  model.load_from=/absolute/path/to/checkpoint.pth train.seed=1001
```

三档 launcher 虽然训练时的 sequence length 不同，但纯 test 都生成完整源
clip。DiffSHEG evaluator 会统一重新切成 88 帧窗口，因此三档都使用同一个
评估入口和同一套 AE 权重，不需要为 64/128/256 各准备一份 evaluator。

以日志中实际打印的 `work_dir` 为准，输入目录为：

```text
<work_dir>/npz/test/
├── res_<speaker>__<sequence>.npz
├── gt_<speaker>__<sequence>.npz
└── ...
```

默认 `--gt-dir` 等于 `--pred-dir`。不能传 LMDB，也不能传旧版转换器生成的
body-only NPZ。

## 6. 先做轻量输入检查

推荐先检查所有 clip 的配对、维度、帧率、长度、统计量、AE 文件路径、音频和
SMPL-X asset。此模式不会 import torch，也不会加载三套大权重：

```bash
export RUN_DIR=/absolute/path/to/work_dir
export DIFFSHEG_ROOT=/absolute/path/to/DiffSHEG
export TALKSHOW_ROOT=/absolute/path/to/TalkSHOW
export SHOW_SPLIT_ROOT=/absolute/path/to/talkshow_split_view

python scripts/diffsheg_show_eval.py \
  --pred-dir "$RUN_DIR/npz/test" \
  --diffsheg-root "$DIFFSHEG_ROOT" \
  --talkshow-root "$TALKSHOW_ROOT" \
  --source-audio-root "$SHOW_SPLIT_ROOT" \
  --validate-input-only \
  --output "$RUN_DIR/diffsheg_show_validation.json"
```

检查成功会给出：

```text
[done] validation report: ...
```

## 7. 跑完整七项指标

推荐用原始 TalkSHOW test WAV 计算 BA：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/diffsheg_show_eval.py \
  --pred-dir "$RUN_DIR/npz/test" \
  --diffsheg-root "$DIFFSHEG_ROOT" \
  --talkshow-root "$TALKSHOW_ROOT" \
  --source-audio-root "$SHOW_SPLIT_ROOT" \
  --device cuda:0 \
  --batch-size 64 \
  --output "$RUN_DIR/diffsheg_show_metrics.json"
```

程序依次执行：

1. 扫描一次所有 NPZ，验证 prediction/GT 一一对应；
2. 计算 PCM 和两项 diversity；
3. 加载 `gesture_expression.pth.tar`，计算 FMD，然后释放；
4. 加载 `expression.pth.tar`，计算 FED，然后释放；
5. 加载 `gesture.pth.tar`，计算 FGD，然后释放；
6. 加载 SMPL-X，计算 BA；
7. 原子写入 JSON。

三套 AE 不会同时驻留 GPU。`--batch-size 64` 表示每次有 64 个 prediction
窗口，程序会把对应 64 个 GT 拼在同一轮前向中，所以实际 AE 输入第一维是
128。显存不足时降低到 32 或 16；这只改变吞吐，不改变指标。

若原始 split view 不在机器上，可使用本次 test 保存的对齐音频：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/diffsheg_show_eval.py \
  --pred-dir "$RUN_DIR/npz/test" \
  --diffsheg-root "$DIFFSHEG_ROOT" \
  --talkshow-root "$TALKSHOW_ROOT" \
  --audio-dir "$RUN_DIR/wav/test" \
  --device cuda:0 \
  --output "$RUN_DIR/diffsheg_show_metrics.json"
```

这种情况下 FMD/FED/FGD/PCM/diversity 不变，但音频边界经过主项目对齐，
BA 可能略有差异，JSON 会记录 warning。

## 8. 只算 motion/AE 指标

如果当前环境没有 TalkSHOW、SMPL-X 或音频，使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/diffsheg_show_eval.py \
  --pred-dir "$RUN_DIR/npz/test" \
  --diffsheg-root "$DIFFSHEG_ROOT" \
  --skip-ba \
  --device cuda:0 \
  --batch-size 64 \
  --output "$RUN_DIR/diffsheg_show_metrics_no_ba.json"
```

该命令仍会输出：

```text
FMD / FED / FGD / PCM / expression diversity / gesture diversity
```

只有 `BA` 缺失。不要把这个六项报告描述成“完整 DiffSHEG 指标”。

## 9. 指标具体含义

| JSON 字段 | DiffSHEG 表格 | 计算对象 | 趋势 |
|---|---|---|---|
| `fmd` | FMD | 标准化的 232 维 gesture+expression，经 holistic AE 的 latent Fréchet distance | 越低越好 |
| `fed` | FED | 标准化的 103 维 jaw+expression，经 expression AE 的 latent Fréchet distance | 越低越好 |
| `expression_diversity` | Expression Diversity | 生成 expression 窗口之间的平均 pairwise L1 | 与 GT 分布接近为佳 |
| `fgd` | FGD | 标准化的 129 维 gesture，经 gesture AE 的 latent Fréchet distance | 越低越好 |
| `ba` | BA | TalkSHOW SMPL-X motion beat 与 audio onset 的对齐分数 | 越高越好 |
| `pcm` | PCM | 把标准化 gesture reshape 成 43 个 3 维 axis-angle joint，joint L2 误差小于 `1.0` 的比例 | 越高越好 |
| `gesture_diversity` | Gesture Diversity | 生成 gesture 窗口之间的平均 pairwise L1 | 与 GT 分布接近为佳 |

论文的 PCM 公式对第 `j` 个 body joint 做 L2，并在 SHOW 上设置
`δ=1.0`。DiffSHEG 公共 trainer 的在线 validation 代码却把每个 axis-angle
标量单独处理，将同类量命名为 PCK 并使用 `0.5`。本脚本的主字段 `pcm` 采用
论文的 3 维 joint L2 和 `1.0`，同时在 JSON diagnostics 中额外保存：

```text
gesture_scalar_accuracy_at_1_0
gesture_pck_at_0_5_public_validation_code
```

这项只用于排查公共代码与论文文字的差异，不替代 `pcm`。

## 10. 固定评估 subset 或顺序

默认会要求 `res_*.npz` 与 `gt_*.npz` 的 clip 集完全一致，并按 clip ID
字典序运行。若要让多个方法严格使用同一个 subset 和顺序，可准备文本：

```text
oliver__000001-00_00_00-00_00_10
oliver__000002-00_00_10-00_00_20
...
```

然后追加：

```text
--clip-manifest /absolute/path/to/diffsheg_eval_clip_ids.txt
```

空行和 `#` 开头的注释会被忽略。manifest 中任何 ID 缺 prediction 或 GT 都会
直接报错；目录中不在 manifest 的额外文件不会参与评估。最终 JSON 保存顺序
说明和 `clip_manifest_sha256`。

## 11. 如何比较 checkpoint

对每个 checkpoint：

1. 用同一档 test launcher、相同 CFG、相同 sampler 和固定
   `TALKSHOW_TEST_SEED` 生成一份完整 test；
2. 用完全相同的 stats、三套 AE、clip manifest、窗口 stride 和音频来源跑本
   evaluator；
3. 保存每个输出 JSON；
4. 按需要分别选择 FMD/FED/FGD 最低或 BA/PCM 最高的 checkpoint。

不要把不同随机 seed 的指标混在一起挑 checkpoint。若要报告扩散采样的均值和
方差，应预先固定一组 seed，对每个 checkpoint 都运行同一组 seed，再对每次
完整 JSON 的指标求均值；不要先把不同 seed 的 NPZ 混进一个目录。

## 12. 常见错误

### 找不到 `talkshow_mean_std.npy`

确认已经解压 `DiffSHEG/assets/data.tar.gz`，或者显式传 `--stats`。

### 找不到三个 `.pth.tar`

这三份是评估 AE，不是 DiffSHEG 的生成模型 checkpoint。目录和文件名必须为：

```text
gesture.pth.tar
expression.pth.tar
gesture_expression.pth.tar
```

### checkpoint shape 不匹配

必须使用 SHOW、`n_poses=88`、`vae_length=300` 的三套 AE。BEAT AE、生成模型
checkpoint 或自行训练的不同 latent size 权重都不兼容。

### prediction/GT clip set mismatch

`--pred-dir` 必须指向一次 canonical test 的 `npz/test`，其中同时有一一对应的
`res_` 和 `gt_`。如果 GT 在另一目录，追加 `--gt-dir`。

### BA 报分母为 0

说明当前预测在 TalkSHOW 阈值下没有形成可用 motion beat，BA 对这批输入没有
定义。先检查预测是否静止、pose 通道是否完整以及是否误传 body-only NPZ；不
应通过给分母加 epsilon 把它伪装成有效论文指标。

### 想与 TalkSHOW、ReCoM 或 SemTalk 比较

先确认对方使用的 evaluator。若其使用 TalkSHOW 原始 FGD/BC，请改用
`scripts/talkshow_official_eval.py`；若使用 EMAGE，则改用
`scripts/emage_eval.py --dataset talkshow`。只有明确使用 DiffSHEG 的三套
SHOW AE 时，才使用本页入口。
