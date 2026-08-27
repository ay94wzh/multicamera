#!/usr/bin/env python3
"""Visualize a recorded demo to verify its plausibility.

Reads the 5 camera images already stored in the HDF5 (obs/<name>_image) — no
re-rendering. Produces:

  <output_dir>/demo_<i>_grid.mp4   -> 3x2 grid video: left/right/top on row 1,
                                      bottom/front + live info panel on row 2
  <output_dir>/demo_<i>_plot.png   -> plausibility plot: cube/EE heights over
                                      time (with the success threshold), gripper
                                      width, and a top-down XY view

Also prints a per-demo summary (length, success, first success step, max lift).
"""

import argparse
import os

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw

CAMERAS = ["left", "right", "top", "bottom", "front"]
TABLE_Z = 0.8          # tabletop height (classic Lift, table_offset [0,0,0.8])
SUCCESS_LIFT = 0.04    # fork's env success: cube z > table + 0.04 (lift.py _check_success)
PLANNER_LIFT = 0.22    # mp_lift_abs.py's own (stricter) success detection


def load_demo(hdf5_path, demo_idx):
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys())
        if f"demo_{demo_idx}" not in f["data"]:
            raise SystemExit(
                f"demo_{demo_idx} not found in {hdf5_path} "
                f"(available: {', '.join(keys[:10])}{'...' if len(keys) > 10 else ''})"
            )
        d = f[f"data/demo_{demo_idx}"]
        images = {c: np.asarray(d[f"obs/{c}_image"]) for c in CAMERAS}
        obj = np.asarray(d["obs/object"])
        eef_pos = np.asarray(d["obs/robot0_eef_pos"])
        gripper = np.asarray(d["obs/robot0_gripper_qpos"])
        rewards = np.asarray(d["rewards"])
        actions = np.asarray(d["actions"])
    return images, obj, eef_pos, gripper, rewards, actions


def make_info_panel(step, T, obj, eef_pos, gripper, rewards, size=256):
    """White panel with the current demo state, drawn as text."""
    panel = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(panel)
    lift = float(obj[step, 2]) - TABLE_Z
    eef_h = float(eef_pos[step, 2]) - TABLE_Z
    success_so_far = bool(np.any(rewards[: step + 1] == 1.0))
    lines = [
        f"demo step {step} / {T - 1}",
        f"t = {step / 20.0:.1f} s  (20 Hz)",
        "",
        f"cube lift  = {lift:+.3f} m",
        f"EE height  = {eef_h:+.3f} m",
        f"success at = {SUCCESS_LIFT:+.3f} m",
        "",
        f"gripper    = {float(gripper[step, 0]):.3f} m",
        f"success    = {success_so_far}",
    ]
    y = 14
    for line in lines:
        draw.text((16, y), line, fill="black")
        y += 22
    return panel


def compose_grid(images, step, T, obj, eef_pos, gripper, rewards, scale=2):
    """3x2 grid: left/right/top | bottom/front/info. Images scaled by `scale`."""
    size = images[CAMERAS[0]].shape[1] * scale
    canvas = Image.new("RGB", (size * 3, size * 2), "black")
    panels = [images[c][step] for c in ("left", "right", "top", "bottom", "front")]
    panels.append(np.array(make_info_panel(step, T, obj, eef_pos, gripper, rewards, size)))
    for i, img in enumerate(panels):
        pil_img = Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS)
        canvas.paste(pil_img, ((i % 3) * size, (i // 3) * size))
    return canvas


def make_plot(obj, eef_pos, gripper, rewards, actions, out_path, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(T) / 20.0
    cube_lift = obj[:, 2] - TABLE_Z
    eef_h = eef_pos[:, 2] - TABLE_Z
    first_success = int(np.argmax(rewards == 1.0)) if np.any(rewards == 1.0) else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    ax = axes[0]
    ax.plot(t, cube_lift, label="cube lift height")
    ax.plot(t, eef_h, label="EE height")
    ax.axhline(SUCCESS_LIFT, color="r", ls="--", lw=1, label=f"env success threshold (+{SUCCESS_LIFT:.2f} m)")
    ax.axhline(PLANNER_LIFT, color="orange", ls=":", lw=1, label=f"planner detection (+{PLANNER_LIFT:.2f} m)")
    if first_success is not None:
        ax.axvline(t[first_success], color="g", ls=":", lw=1, label=f"first success @ t={t[first_success]:.1f}s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("height above table (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Plausibility check: does the robot grasp and lift the cube?")

    ax = axes[1]
    ax.plot(t, gripper[:, 0], label="gripper width")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("gripper opening (m)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(obj[:, 0], obj[:, 1], "o-", ms=2, label="cube XY path")
    ax.plot(eef_pos[:, 0], eef_pos[:, 1], "x-", ms=2, label="EE XY path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle(f"demo (T={T} steps, actions |a|_max={np.abs(actions).max():.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos", "eef_delta.hdf5"))
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_videos"))
    parser.add_argument("--fps", type=int, default=20, help="video playback rate (recorded at 20 Hz)")
    parser.add_argument("--scale", type=int, default=2, help="upscale factor for stored 128x128 images")
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame (1 = full speed)")
    args = parser.parse_args()

    images, obj, eef_pos, gripper, rewards, actions = load_demo(args.hdf5, args.demo_idx)
    T = images[CAMERAS[0]].shape[0]
    os.makedirs(args.output_dir, exist_ok=True)

    # --- summary ---------------------------------------------------------
    cube_lift = obj[:, 2] - TABLE_Z
    first_success = int(np.argmax(rewards == 1.0)) if np.any(rewards == 1.0) else None
    print(f"demo_{args.demo_idx}: T={T} steps ({T / 20.0:.1f} s)")
    print(f"  success: {first_success is not None}" +
          (f" (first at step {first_success}, t={first_success / 20.0:.1f} s)" if first_success is not None else ""))
    print(f"  max cube lift: {cube_lift.max():.3f} m (threshold {SUCCESS_LIFT:.2f} m)")
    print(f"  max |action|: {np.abs(actions).max():.3f}")

    # --- video -----------------------------------------------------------
    grid_path = os.path.join(args.output_dir, f"demo_{args.demo_idx}_grid.mp4")
    writer = imageio.get_writer(grid_path, fps=args.fps)
    for step in range(0, T, args.every):
        frame = compose_grid(images, step, T, obj, eef_pos, gripper, rewards, scale=args.scale)
        writer.append_data(np.asarray(frame))
    writer.close()
    print(f"  video -> {grid_path}")

    # --- plausibility plot ------------------------------------------------
    plot_path = os.path.join(args.output_dir, f"demo_{args.demo_idx}_plot.png")
    make_plot(obj, eef_pos, gripper, rewards, actions, plot_path, T)
    print(f"  plot  -> {plot_path}")


if __name__ == "__main__":
    main()
