# LIBERO Preprocessing Changes

This branch contains the LIBERO-specific preprocessing and closed-loop alignment changes used for DreamZero LIBERO training/evaluation.

## What Changed

### Two-view image layout

LIBERO has two cameras: primary and wrist. The transform now concatenates them horizontally:

```text
primary | wrist
```

This replaces the generic 2x2 multi-view layout that wasted half of the image on black pixels for two-view data.

Files:

- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain_libero.yaml`

### Per-view image preprocessing

LIBERO per-view preprocessing is kept aligned with the FastWAM/LingBotVA LIBERO path:

```text
ToTensor -> Resize -> ToNumpy
```

The LIBERO branch does not use the generic DreamZero random crop or color jitter transforms. When frame cache is enabled, resize is already baked into the cached frames and `libero_training.sh` disables online crop/resize.

Files:

- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `scripts/train/libero_training.sh`

### LIBERO-specific prompt path

LIBERO uses raw task language by default, instead of DROID's multi-view layout description. This avoids mixing DROID prompt semantics into LIBERO training.

Files:

- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain_libero.yaml`

### Independent action/state projector id

Raw `embodiment_id` is still used for embodiment/prompt routing, but action/state projectors now receive a compact `action_projector_id`.

Current LIBERO config:

```yaml
action_projector_num_embeddings: 2
action_projector_tag_to_index:
  libero_sim: 0
  oxe_droid: 1
```

This prevents sparse raw ids such as `libero_sim: 14` and `oxe_droid: 17` from being used directly by category-specific action/state MLP rows.

Old checkpoints with one projector row are loaded by copying row 0 into the expanded rows, so fine-tuning starts from the previous behavior instead of random projector rows.

Files:

- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml`
- `groot/vla/configs/model/dreamzero/transform/dreamzero_cotrain_libero.yaml`

### Gripper normalization and env conversion

The LIBERO LeRobot data stores action gripper as RLDS-style openness:

```text
0 = closed
1 = open
```

LIBERO/robosuite env actions use signed command:

```text
+1 = close
-1 = open
```

Training now normalizes `action.gripper` with `min_max` so the 0/1 semantics are preserved. Evaluation uses a shared adapter to convert model output to LIBERO env convention.

Files:

- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `eval_utils/libero_action_adapter.py`
- `eval_utils/eval_libero_dreamzero.py`
- `eval_utils/run_libero_dreamzero_standard.py`

## Data Sanity Check

Run this before training on a converted LIBERO dataset:

```bash
python scripts/data/check_libero_action_semantics.py \
  --dataset-path /path/to/libero_goal_no_noops_1.0.0_lerobot \
  --max-rows 200000
```

Expected output for RLDS-style LIBERO data should show `action.gripper` values in `[0, 1]`, typically only `0` and `1`, and converted env gripper values in `{-1, +1}`.

## Training Entry

Current LIBERO training entry:

```bash
PER_DEVICE_BS=4 \
GLOBAL_BATCH_SIZE=64 \
MAX_STEPS=30000 \
SAVE_STEPS=5000 \
MAX_CHUNK_SIZE=1 \
NUM_FRAMES=9 \
USE_FRAME_CACHE=true \
ACTION_HEAD_CONFIG=wan_flow_matching_action_tf_wan22_224 \
DECOUPLE_VIDEO_ACTION_NOISE=true \
TRANSFORM_CONFIG=dreamzero_cotrain_libero \
bash scripts/train/libero_training.sh
```

## Evaluation Notes

The standard LIBERO evaluation wrapper forwards gripper conversion options to the task runner:

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

Default gripper conversion uses threshold `0.5` and binarizes the gripper command. For ablations:

```bash
--gripper-threshold 0.5
--no-binarize-gripper
```

## Related Detailed Notes

See `docs/DREAMZERO_LOCAL_CHANGES.md` for the full local change log, including frame cache, 224-resolution config, evaluation scripts, and known LIBERO evaluation results.
