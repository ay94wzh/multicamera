# Pure Diffusion Policy Baseline — 4-cam train, 4-cam & front-only eval

A self-contained copy of the origin training stack (`policy_robosuite/`, untouched)
modified for a **pure diffusion policy** (`policy_class='dp'`) baseline on your fixed
camera rig:

- **Train**: 4 cameras — left, right, top, bottom — as RGB image input, **no Plucker
  embedding**.
- **Eval the same checkpoint twice**: (a) with the same 4 cameras, (b) with the
  **front camera only**.
- **Origin task, custom cameras only**: verified on the **classic robosuite `Lift`**
  task, whose setup (table, floor, walls) is already static by default — no task
  definition is modified and no background is pinned. The fork's `*_rand` tasks stay
  exactly as the origin code has them (random backgrounds and all).

## Why the same checkpoint works for 4 cams and 1 cam

In `models/dp.py`, camera features are **mean-pooled** over views (`(B,N,64) -> (B,64)`)
instead of concatenated (`64*N`), and the camera-count check is relaxed. The shared
`RgbEncoder` is applied per view, so the model has exactly one vision encoder, one UNet,
and no camera-count-dependent tensors — one `state_dict` serves any N.

## File map

| File | Role |
|------|------|
| `train.py` | train loop + `--eval_only` (eval a saved checkpoint, no wandb) |
| `utils.py` | dataset: RIG pose files (one rig per demo), C=3 RGB, `fix_scene_setup()` (no-op on classic tasks) |
| `eval.py` | evaluator: RIG poses per episode, N-camera video composition (1 / 2 / 2x2) |
| `models/dp.py` | DiffusionPolicy with mean-pooled, N-agnostic conditioning |
| `cam_embedding.py` | verbatim copy (only used when `--use_plucker 1`) |

Rig pose files come from `custom_camera_demos/gen_camera_poses.py` (new `rig_train.json`,
`rig_test.json`, `front_eval.json` outputs, format `{"format":"rig","camera_names":[...],
"poses":[[N x 4x4] per demo]}`).

## Commands

```bash
conda activate know_your_camera   # mujoco==3.1.6 already pinned

# 1. rig pose files (4-cam rig + front-only rig)
cd custom_camera_demos
python gen_camera_poses.py --task lift --train_demos 200 --test_demos 50 --eval_episodes 50

# 2. record demos on the classic task (static setup by origin; object placement
#    and the end-effector pose stay random per demo). 210 = 200 train + 10 val
#    (load_data asserts num_episodes + 10 <= demos, utils.py:517)
python gen_demos_with_cameras.py --task lift --num_demos 210 \
    --action_spaces eef_delta --output_files eef_delta.hdf5 \
    --poses_json camera_poses/custom_cameras.json --image_size 128

# 3. train (image-only: prob_drop_proprio=1.0)
cd ../custom_dp_baseline
WANDB_MODE=offline python train.py --policy_class dp --name dp4cam_lift_seed0 \
    --dataset_dir /home/zihan-wang/CamPoseOpensource/custom_camera_demos/demos --dataset_suffix eef_delta \
    --camera_poses_dir /home/zihan-wang/CamPoseOpensource/custom_camera_demos/camera_poses \
    --train_poses_file rig_train.json --test_poses_file rig_test.json --pose_files rig_test.json front_eval.json \
    --num_side_cam 4 --use_plucker 0 --use_cam_pose 0 --num_episodes 200 --batch_size 16 \
    --transform crop --prob_drop_proprio 1.0 --num_epochs 30001 --eval_every 1000 \
    --eval_start_epoch 20000 --eval_episodes 50 --save_every 1000 --seed 0

# 4a. eval the same checkpoint with the 4 cameras
WANDB_MODE=offline python train.py --eval_only 1 --policy_class dp --name dp4cam_lift_seed0 \
    --dataset_dir /home/zihan-wang/CamPoseOpensource/custom_camera_demos/demos --dataset_suffix eef_delta \
    --camera_poses_dir /home/zihan-wang/CamPoseOpensource/custom_camera_demos/camera_poses \
    --train_poses_file rig_train.json --test_poses_file rig_test.json --pose_files rig_test.json \
    --num_side_cam 4 --use_plucker 0 --use_cam_pose 0 --num_episodes 200 --batch_size 16 \
    --transform crop --prob_drop_proprio 1.0 --eval_episodes 50

# 4b. eval the same checkpoint with the front camera only
WANDB_MODE=offline python train.py --eval_only 1 --policy_class dp --name dp4cam_lift_seed0 \
    ... same args ... --pose_files front_eval.json --eval_episodes 50
```

## Results

Everything lands under `<ckpt_dir>` (= `checkpoints/<name>` next to this folder):

- `eval_epoch_<N>_rig_test/` and `eval_epoch_<N>_front_eval/` — one dir per pose file,
  each with `success_by_seed.json` (per-episode success) and `*.mp4` videos
  (2x2 grid 512x512 for 4 cams, single 256x256 for front).
- `eval_only_epoch_<N>_<pose_name>/` — same layout for `--eval_only` runs.
- `dataset_stats.json`, `config.json`, `epoch_*.pth` checkpoints (last 3 kept).

## Notes

- **Front-only eval is a distribution shift**: training conditions on the mean of 4
  views; the front-only eval feeds the mean of 1 view (the front camera was never
  seen during training). Expect degraded success — that is the point of the baseline.
- `--name` must be identical between training and `--eval_only` runs (it anchors
  `ckpt_dir`). The last 7 chars of the name are treated as the seed for the wandb
  group; keep the `_seed0` suffix.
- `WANDB_MODE=offline` avoids network dependency; `--eval_only` skips wandb entirely.
- **Train without in-loop eval**: pass `--eval_every 0` to disable the in-loop
  evaluation (the origin loop always evaluates at epoch 0 and every
  `--eval_every` epochs; `0` skips all of it). Train first, then evaluate the
  saved checkpoint with `--eval_only` (steps 4a/4b) — the pose files are
  loaded fresh there.
- `--batch_size 16` is sized for 4 x 256x256 images; rendering
  (4 renders per sample, `num_workers=0`) is the throughput bottleneck.
- **GPU memory**: the origin DP model is ~264M params — the optimizer step alone
  needs ~4.3 GiB (weights + grads + Adam states), so training requires roughly
  `4.3 + 1.5 * batch_size` GiB at 256x256 (batch 16 ≈ 28 GiB, e.g. A100 40G; the
  origin's own default `batch_size 70` implies a >48 GiB GPU). A 6-8 GiB GPU can
  only fit `--batch_size 1..2`. `--eval_only` needs no Adam states and fits
  anywhere (measured peak ~2.6 GiB). Verified on the repo's 6 GiB GPU: eval-only
  round-trip works; the training loop was verified end-to-end on the earlier
  `dp4cam_tiny_seed0` run (2 epochs, both pose files, eval-only reproduces the
  in-loop evals).
- Demos are recorded on the classic `Lift` task, whose setup is static by origin;
  no background pinning is involved. Training replays recorded states via
  `set_state_from_flattened`, so it is always consistent with the recorded frames.
- `fix_scene_setup(env)` exists only for the fork's `*_rand` tasks (randomized
  table-top / floor planes): it pins `plane_sampler` / `table_plane_sampler` /
  `floor_plane_sampler` in place (`x_range=[0,0]`, `y_range=[0,0]`,
  `rotation=[0,0]`) **and sets `env.hard_reset = False`** (the fork's default
  `hard_reset=True` re-runs `_load_model` on every `env.reset()`, recreating the
  samplers with their random ranges). **On classic tasks (e.g. `Lift`) it is a
  no-op**: no such samplers exist, `hard_reset` stays `True`, and the env behaves
  exactly as origin robosuite. The `*_rand` tasks themselves are never touched.
- `--original 1` (kept from the origin) is an optional alternate look; only
  relevant for `*_rand` tasks.
