#!/usr/bin/env python3
"""Generate the fixed 5-camera rig: left / right / top / bottom / front.

All cameras lie on a sphere centered at the task workspace (in front of the table),
are fixed, and look at the robot. This writes the camera poses in the exact JSON
format consumed by policy_robosuite (a list of 4x4 cam-to-world matrices), plus a
compact file with the named 5 poses used by gen_demos_with_cameras.py.

Outputs (default: ./camera_poses and ./preview next to this script):
    train_cameras.json  -> poses for the training demos: 5 fixed poses repeated
                           once per demo, so a training window W(i)=[5i, 5i+5)
                           always contains the same 5 cameras (train with --m 5 --n 5)
    test_cameras.json   -> same 5 fixed poses, repeated per validation demo
    custom_cameras.json -> the 5 named poses (used by gen_demos_with_cameras.py)
    rig_train.json      -> RIG format: one [4 x 4x4] rig per demo for the 4-cam
                           baseline (left/right/top/bottom, NO front), indexed
                           directly by demo (used by custom_dp_baseline)
    rig_test.json       -> same 4-cam rigs, one per validation demo
    front_eval.json     -> RIG format with only the front camera, one per eval
                           episode (front-only eval of the 4-cam checkpoint)

Run --preview-only to just render the five views to PNGs and tune the angles
before generating anything.

Examples:
    python gen_camera_poses.py --task liftrand --train_demos 50 --test_demos 10
    python gen_camera_poses.py --preview-only --radius 1.2 --look-at 0 0 0.95
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# --- Default 5-camera rig (azimuth/elevation in degrees) -------------------
# Azimuth is measured from +X (the "front" of the table, where the camera rig
# stands; the table sits at the origin, its front face at x = +0.4).
# Elevation is measured from the horizontal plane.
DEFAULT_CAMERAS = [
    ("front", 0.0, 10.0),
    ("left", -55.0, 10.0),
    ("right", 55.0, 10.0),
    ("top", 0.0, 60.0),
    ("bottom", 0.0, -18.0),
]

# Task -> robosuite env name (same mapping as gen_robosuite_format_demo.py)
TASK_TO_ENV = {
    "lift": "Lift",  # classic robosuite task: static setup, random object placement
    "liftrand": "LiftRand",
    "canrand": "CanRand",
    "squarerand": "SquareRand",
}


def build_cam_to_world(camera_pos, look_at):
    """Build a 4x4 cam-to-world matrix identical in convention to
    policy_robosuite/cam_utils.py: columns are [right, up, -forward, pos].
    """
    forward = look_at - camera_pos
    forward = forward / np.linalg.norm(forward)

    up = np.array([0.0, 0.0, 1.0])
    if abs(forward[2]) > 0.99:  # degenerate: forward ~ world up
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = -forward
    cam_to_world[:3, 3] = camera_pos
    return cam_to_world


def camera_pose(name, az_deg, el_deg, center, radius, look_at):
    """Position of one camera on the sphere (center=look-at target) and its pose."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    pos = np.array([
        center[0] + radius * np.cos(el) * np.cos(az),
        center[1] + radius * np.cos(el) * np.sin(az),
        center[2] + radius * np.sin(el),
    ])
    return build_cam_to_world(pos, look_at)


def make_env(task):
    """Create the same env used by the demo generator (offscreen renderer on)."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_configs = load_composite_controller_config(robot="Panda")
    controller_configs["body_parts"]["right"]["input_type"] = "absolute"
    controller_configs["body_parts"]["right"]["input_ref_frame"] = "world"

    return suite.make(
        env_name=TASK_TO_ENV[task],
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=20,
        reward_shaping=False,
        camera_depths=False,
        camera_heights=256,
        camera_widths=256,
        controller_configs=controller_configs,
    )


def render_previews(env, cameras, poses, look_at, out_dir, image_size=256):
    """Render one PNG per camera + a labeled grid so the user can verify the views."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    sim = env.sim
    cam_id = sim.model.camera_name2id("agentview")
    imgs = {}
    for (name, _, _), pose in zip(cameras, poses):
        sim.model.cam_pos[cam_id] = pose[:3, 3]
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()  # xyzw
        sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]
        sim.forward()
        img = sim.render(camera_name="agentview", height=image_size, width=image_size, depth=False)
        imgs[name] = np.flipud(img).copy()
        Image.fromarray(imgs[name]).save(os.path.join(out_dir, f"{name}.png"))
        print(f"  preview saved: {os.path.join(out_dir, name + '.png')}")

    # Labeled grid (3 cols x 2 rows)
    h, w = imgs["front"].shape[:2]
    grid = Image.new("RGB", (w * 3 + 8, h * 2 + 8), "black")
    draw = ImageDraw.Draw(grid)
    for i, name in enumerate([c[0] for c in cameras]):
        r, c = divmod(i, 3)
        grid.paste(Image.fromarray(imgs[name]), (c * w + 4, r * h + 4))
        draw.text((c * w + 10, r * h + 10), name, fill=(255, 0, 0))
    grid_path = os.path.join(out_dir, "grid.png")
    grid.save(grid_path)
    print(f"  grid saved: {grid_path}")
    return imgs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default="liftrand", choices=list(TASK_TO_ENV))
    parser.add_argument("--train_demos", type=int, default=50,
                        help="training demos -> poses per demo (m=5, n=5 in train.py)")
    parser.add_argument("--test_demos", type=int, default=10,
                        help="validation demos -> poses per demo (used as test_cameras.json)")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="eval episodes -> one rig per episode in front_eval.json")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.8],
                        help="sphere center / task workspace (table top is at z=0.8)")
    parser.add_argument("--look-at", type=float, nargs=3, default=[0.0, 0.0, 0.95],
                        help="point every camera looks at (default: robot arm center, "
                             "above tabletop so the bottom view clears the table edge)")
    parser.add_argument("--radius", type=float, default=1.2, help="sphere radius (m)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_poses"))
    parser.add_argument("--preview_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview"))
    parser.add_argument("--preview-only", action="store_true", help="only render preview PNGs, don't write JSONs")
    parser.add_argument("--image_size", type=int, default=256, help="preview render size (px)")
    args = parser.parse_args()

    center = np.array(args.center)
    look_at = np.array(args.look_at)

    # Camera table with per-name overrides via environment variables (e.g.
    # CAM_TOP_ELEV=65 CAM_BOTTOM_AZ=-30 python gen_camera_poses.py ...)
    cameras = []
    for name, az, el in DEFAULT_CAMERAS:
        az = float(os.environ.get(f"CAM_{name.upper()}_AZ", az))
        el = float(os.environ.get(f"CAM_{name.upper()}_ELEV", el))
        cameras.append((name, az, el))

    poses = [camera_pose(name, az, el, center, args.radius, look_at) for name, az, el in cameras]

    # --- Preview ----------------------------------------------------------
    if args.preview_only:
        env = make_env(args.task)
        env.reset()
        try:
            render_previews(env, cameras, poses, look_at, args.preview_dir, args.image_size)
        finally:
            env.close()
        print("Preview done. Tune angles with CAM_*_AZ / CAM_*_ELEV env vars or edit DEFAULT_CAMERAS.")
        return

    # --- Write JSONs -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    def config_json():
        return {
            "num_cameras": len(cameras),
            "camera_names": [c[0] for c in cameras],
            "workspace_center": center.tolist(),
            "look_at": look_at.tolist(),
            "radius": args.radius,
            "azimuths_deg": [c[1] for c in cameras],
            "elevations_deg": [c[2] for c in cameras],
        }

    def write_poses(filename, num_demos):
        # The 5 fixed poses repeated once per demo. Training with --m 5 --n 5
        # makes window W(i) = [5i, 5i+5) = the same 5 cameras for every demo i.
        repeated = [p.tolist() for _ in range(num_demos) for p in poses]
        data = {"config": config_json(), "poses": repeated}
        path = os.path.join(args.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(repeated)} poses ({num_demos} demos x {len(cameras)} cameras) -> {path}")

    def write_rig(filename, names, num_demos):
        # RIG format: one entry per demo, each the full [N x 4x4] camera rig.
        # Indexed directly by demo/episode (no window sampling), so the file
        # must have >= num_demos entries. Used by custom_dp_baseline.
        by_name = {c[0]: p.tolist() for c, p in zip(cameras, poses)}
        rigs = [[by_name[n] for n in names] for _ in range(num_demos)]
        data = {"format": "rig", "camera_names": list(names), "config": config_json(), "poses": rigs}
        path = os.path.join(args.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(rigs)} per-demo rigs x {len(names)} cameras ({names}) -> {path}")

    write_poses("train_cameras.json", args.train_demos)
    write_poses("test_cameras.json", args.test_demos)

    # Rig files for the 4-cam baseline (left/right/top/bottom, no front)
    # and the front-only eval rig.
    rig4_names = [c[0] for c in cameras if c[0] != "front"]
    write_rig("rig_train.json", rig4_names, args.train_demos)
    write_rig("rig_test.json", rig4_names, args.test_demos)
    write_rig("front_eval.json", ["front"], args.eval_episodes)

    # Named 5 poses for the demo recorder
    named = {"camera_names": [c[0] for c in cameras], "poses": [p.tolist() for p in poses]}
    path = os.path.join(args.output_dir, "custom_cameras.json")
    with open(path, "w") as f:
        json.dump(named, f, indent=2)
    print(f"Saved 5 named poses -> {path}")

    print("\nCameras (sphere center", center.tolist(), ", look-at", look_at.tolist(), ", radius", args.radius, "):")
    for (name, az, el), pose in zip(cameras, poses):
        print(f"  {name:6s} az={az:6.1f} el={el:6.1f}  pos={pose[:3, 3]}")
    print("\nFor training with these fixed cameras:")
    print(f"  --train_poses_file train_cameras.json --test_poses_file test_cameras.json --m 5 --n 5 --num_side_cam 1|2")
    print(f"  (camera_poses_dir = {args.output_dir} or copy the JSONs into policy_robosuite/camera_poses/)")
    print("\nFor the 4-cam diffusion-policy baseline (custom_dp_baseline):")
    print(f"  --train_poses_file rig_train.json --test_poses_file rig_test.json --pose_files rig_test.json front_eval.json --num_side_cam 4")
    print("\nTip: python gen_camera_poses.py --preview-only to check the views first.")


if __name__ == "__main__":
    main()
