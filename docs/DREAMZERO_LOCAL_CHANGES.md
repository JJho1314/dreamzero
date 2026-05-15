# DreamZero Local Changes Summary

本文档总结当前工作区相对原始 DreamZero 代码做过的主要改动，重点是为了支持 LIBERO 数据训练、加速训练数据读取，以及在本地 A6000 上进行标准 LIBERO 评测。

## 1. LIBERO 数据与训练配置

### 新增 LIBERO 数据配置

相关文件：

- `groot/vla/configs/data/dreamzero/libero_relative.yaml`
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `scripts/data/convert_lerobot_to_gear.py`

主要改动：

- 新增 `libero_sim` embodiment tag，并允许 `convert_lerobot_to_gear.py` 处理该 tag。
- 在 DreamZero 数据配置中加入 `libero_sim` 的 modality 定义：
  - `video.primary_image`
  - `video.wrist_image`
  - `state.eef_pose`
  - `state.gripper`
  - `action.eef`
  - `action.gripper`
  - `annotation.task`
- LIBERO 动作配置为 delta-EEF + gripper，因此 `libero_relative.yaml` 中关闭了二次 relative action：
  - `relative_action: false`
  - `relative_action_per_horizon: false`
  - `relative_action_keys: []`
- 默认使用 4 个 LIBERO 子集：
  - goal
  - object
  - spatial
  - libero_10
- `libero_sim` 默认 FPS 配置为 `20`。

### LIBERO embodiment 映射

相关文件：

- `groot/vla/configs/model/dreamzero/transform/base.yaml`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain_libero.yaml`

主要改动：

- 新增 `libero_sim: 14`，在 transform / prompt 分支上与 `oxe_droid: 17` 分离，避免 LIBERO 训练吃到 DROID 的文字布局描述。
- 新增 `dreamzero_cotrain_libero.yaml` 作为 LIBERO 专用 transform 配置：
  - `num_views: 2`
  - `libero_prompt_style: raw`
  - `libero_sim: 14`
- 注意：当前 DiT 内部 action/state projector 仍沿用原始实现，会把 `embodiment_id` 覆盖为 category `0`，所以这里的独立 id 主要用于 transform 侧分支和训练链路隔离，不会额外新增一套可训练 projector 参数。

## 2. 图像与序列 Transform 改动

相关文件：

- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain.yaml`

主要改动：

- 对 LIBERO 两视角输入新增专门的水平拼接逻辑：
  - 原来多视角默认走 2x2 grid，LIBERO 两视角会浪费一半黑块。
  - 现在 `num_views == 2` 时拼成：

    ```text
    primary | wrist
    ```

  - 这样保持最终 token 数不变，但有效图像区域更大。
- 为 LIBERO 新增独立 prompt style，避免误用 DROID 的多视角描述：
  - `raw`: 直接使用 LIBERO task 文本，当前 LIBERO 专用配置默认使用这个模式。
  - `simple`: `A robot ...`
  - `layout`: 明确描述 horizontal split。
  - DROID 分支仍走 DROID 自己的 layout prompt。
- 新增 `max_chunk_size` 和 `num_frames` 传入 DreamTransform。
- 对视频长度做 pad/trim，保证 batch 内样本帧数稳定。
- 对 state/action token 数做 pad/trim：
  - state 目标长度：`max_chunk_size * state_horizon`
  - action 目标长度：`max_chunk_size * action_horizon`
- collate 时对 `np.stack` 失败增加 shape 诊断，方便定位 batch 中哪个 key 形状不一致。

## 3. 帧缓存与训练数据读取加速

相关文件：

- `groot/vla/common/utils/misc/video_utils.py`
- `groot/vla/data/transform/video.py`
- `scripts/data/precompute_frame_cache.py`
- `scripts/data/precompute_fastwam_frame_cache.sbatch`
- `scripts/train/libero_training.sh`

主要改动：

- `get_frames_by_timestamps()` 支持从预解码 `.npy` 帧缓存读取。
- 通过环境变量启用缓存：

  ```bash
  DREAMZERO_FRAME_CACHE_ROOT=/path/to/cache
  DREAMZERO_FRAME_CACHE_SOURCE_ROOT=/path/to/source/videos
  ```

- 缓存路径按源视频相对路径映射，例如：

  ```text
  source_root/a/b/video.mp4
  -> cache_root/a/b/video.npy
  ```

- 新增 `scripts/data/precompute_frame_cache.py`：
  - 扫描 source root 下所有 `.mp4`
  - 用 decord 解码
  - resize 到指定方形尺寸，默认 `160`，也支持 `224`
  - 保存为 `.npy`
- 新增 `scripts/data/precompute_fastwam_frame_cache.sbatch`，用于 HPC 上批量生成 LIBERO frame cache。
- `VideoCrop` 和 `VideoResize` 支持通过环境变量禁用：

  ```bash
  DREAMZERO_DISABLE_VIDEO_CROP=true
  DREAMZERO_DISABLE_VIDEO_RESIZE=true
  ```

- `libero_training.sh` 默认启用 frame cache：
  - 默认 cache 尺寸由 `FRAME_CACHE_SIZE` 控制。
  - 当 `ACTION_HEAD_CONFIG=wan_flow_matching_action_tf_wan22_224` 且未手动设置 `FRAME_CACHE_SIZE` 时，默认使用 `224`。
  - 其他情况下默认使用 `160`。
  - 有 cache 时训练侧 image resolution 设置为 `${FRAME_CACHE_SIZE}x${FRAME_CACHE_SIZE}`
  - 同时跳过原始 crop/resize
  - 训练时避免反复视频解码和 resize

### 224 分辨率 Action Head 配置

相关文件：

- `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_224.yaml`

主要改动：

- 新增两视角 LIBERO 的 224 配置：
  - 单视角输入：`224x224`
  - transform 水平拼接后 DiT/VAE 输入：`224x448`
  - VAE38 spatial downsample 后 latent：`14x28`
  - DiT patch stride `(1, 2, 2)` 后每帧 token 数：`7x14 = 98`
- 使用方式：

  ```bash
  ACTION_HEAD_CONFIG=wan_flow_matching_action_tf_wan22_224 \
  FRAME_CACHE_SIZE=224 \
  DREAMZERO_LIBERO_VIEW_SIZE=224
  ```

## 4. 训练脚本与 HPC 作业

相关文件：

- `scripts/train/libero_training.sh`
- `scripts/train/dz_libero_train.sbatch`

主要改动：

- 新增统一 LIBERO 训练入口 `scripts/train/libero_training.sh`。
- 支持通过环境变量控制常用训练参数：

  ```bash
  PER_DEVICE_BS
  GLOBAL_BATCH_SIZE
  MAX_STEPS
  SAVE_STEPS
  SAVE_STRATEGY
  DATALOADER_NUM_WORKERS
  DATASET_SHARD_SAMPLING_RATE
  USE_GRADIENT_CHECKPOINTING
  USE_FRAME_CACHE
  FRAME_CACHE_SIZE
  ACTION_HEAD_CONFIG
  TRANSFORM_CONFIG
  DEEPSPEED_CFG
  ```

- 默认每 `5000` 步保存一次：

  ```bash
  SAVE_STEPS=5000
  SAVE_STRATEGY=steps
  ```

- 默认训练设置是 full fine-tuning：

  ```bash
  TRAIN_ARCHITECTURE=full
  TUNE_PROJECTOR=true
  TUNE_DIFFUSION_MODEL=true
  SAVE_LORA_ONLY=false
  ```

- 新增 Slurm 脚本 `scripts/train/dz_libero_train.sbatch`：
  - 8 GPU
  - ACD partition
  - 512G memory
  - 默认 `PER_DEVICE_BS=4`
  - 默认 `GLOBAL_BATCH_SIZE=64`
  - 默认 `MAX_STEPS=30000`

## 5. 训练稳定性与日志行为

### W&B step 对齐 Trainer global_step

相关文件：

- `groot/vla/experiment/base.py`

主要改动：

- 新增 `TrainerStepWandbCallback`。
- 替换 HuggingFace 默认 W&B callback，使 W&B 的 step 使用 `TrainerState.global_step`。
- 解决 W&B step 和 trainer step 不一致导致曲线难读的问题。

### Gradient checkpointing 可控

相关文件：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

主要改动：

- action head 初始化 diffusion model 后，会将 `config.use_gradient_checkpointing` 写入 DiT model：

  ```python
  self.model.gradient_checkpointing = self.use_gradient_checkpointing
  ```

- 修复关闭 gradient checkpointing 后 forward 分支返回 tuple 时的处理逻辑。
- checkpointing 与非 checkpointing 分支都兼容 block 返回 `(x, kv_cache)` 的情况。

### Batch action loss shape 修复

相关文件：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

主要改动：

- 修复 action loss 中 `has_real_action` broadcast 维度：

  ```python
  has_real_action[:, None, None]
  ```

- 避免 batch size > 1 时出现 action loss tensor shape mismatch。

### Torch compile 可禁用

相关文件：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `eval_utils/eval_libero_dreamzero.py`

主要改动：

- 新增环境变量：

  ```bash
  DISABLE_DREAMZERO_TORCH_COMPILE=true
  ```

- 用于评测或调试时跳过 text/image/VAE 的 `torch.compile`，减少动态 shape / compile cache 带来的不稳定。

## 6. LIBERO 标准评测脚本

相关文件：

- `eval_utils/eval_libero_dreamzero.py`
- `eval_utils/run_libero_dreamzero_standard.py`

主要改动：

- 新增 DreamZero 专用 LIBERO 评测入口。
- 对齐 FastWAM 风格设置：
  - warmup steps: `30`
  - max steps: `800`
  - default trials: `50`
  - default seed: `7`
  - replan steps: `10`
- 图像处理：
  - LIBERO obs 里读取 `agentview_image` 和 `robot0_eye_in_hand_image`
  - 做 LIBERO 所需翻转
  - center crop + resize 到 `DREAMZERO_LIBERO_VIEW_SIZE`，默认 `160`
  - 送入 DreamZero 的两视角 transform
- 动作处理：
  - 从 DreamZero 输出中取 `action.eef` 和 `action.gripper`
  - 拼成 LIBERO env 所需 7 维 action
- 支持每个 task 单独写 JSON：

  ```text
  eval_results/.../libero_spatial/gpu0_task0_results.json
  ```

- 支持两张 GPU 拆任务并行：

  ```bash
  python eval_utils/run_libero_dreamzero_standard.py \
    --model-path /path/to/checkpoint \
    --suites libero_spatial \
    --gpus 0,1 \
    --num-trials 50 \
    --num-envs 8 \
    --subproc-env-step \
    --detach
  ```

### 多环境并行推理

相关文件：

- `eval_utils/eval_libero_dreamzero.py`

主要改动：

- 支持 `--num-envs N`，同一个 model process 内一次 rollout 多个 LIBERO env。
- 支持两种环境步进：
  - `--parallel-env-step`: thread pool
  - `--subproc-env-step`: subprocess env pool
- 实测 thread pool 版本容易触发底层仿真崩溃，当前推荐：

  ```bash
  --num-envs 8 --subproc-env-step
  ```

- subprocess env worker 使用 `spawn`，避免继承 CUDA context。

## 7. 当前常用运行命令

### 训练 LIBERO

```bash
PER_DEVICE_BS=4 \
GLOBAL_BATCH_SIZE=64 \
MAX_STEPS=30000 \
SAVE_STEPS=5000 \
USE_FRAME_CACHE=true \
ACTION_HEAD_CONFIG=wan_flow_matching_action_tf_wan22 \
TRANSFORM_CONFIG=dreamzero_cotrain_libero \
bash scripts/train/libero_training.sh
```

### HPC 提交训练

```bash
sbatch scripts/train/dz_libero_train.sbatch
```

### 评测一个 checkpoint 的 spatial

```bash
/data/LFT-W02_data/.conda/envs/dreamzero/bin/python eval_utils/run_libero_dreamzero_standard.py \
  --model-path /data/LFT-W02_data/junjie/VLA_WM/dreamzero/checkpoints/dreamzero_libero_wan22_full_297534/checkpoint-30000 \
  --suites libero_spatial \
  --gpus 0,1 \
  --num-trials 50 \
  --seed 7 \
  --replan-steps 10 \
  --num-steps-wait 30 \
  --num-envs 8 \
  --subproc-env-step \
  --master-port 29940 \
  --output-dir eval_results/dreamzero_spatial_ckpt30000_numenv8_subproc \
  --detach
```

## 8. 当前注意事项

- `eval_results/`、`weights/`、checkpoint 目录属于运行产物或大文件，不建议提交进 git。
- `--parallel-env-step` 已保留，但不推荐；推荐 `--subproc-env-step`。
- LIBERO 训练当前按 EEF pose / delta-EEF action 链路处理，不是 joint action。
- `libero_10` 当前已按需求停止，不再继续评测。
- 本地已有 spatial 完整结果：
  - 20k：`456/500 = 91.2%`
  - 25k：`462/500 = 92.4%`
  - 30k：`458/500 = 91.6%`
