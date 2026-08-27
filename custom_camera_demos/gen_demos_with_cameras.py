#!/usr/bin/env python3
"""Generate demos with the fixed 5-camera rig (left / right / top / bottom / front).

This is a drop-in alternative to script_robosuite_demos/gen_robosuite_format_demo.py
that additionally records one image per camera at every timestep into the HDF5
dataset (obs/<name>_image and next_obs/<name>_image), using the camera poses
produced by gen_camera_poses.py.

The original repo is NOT modified: the motion-planning controllers are imported
from script_robosuite_demos and the trajectories / HDF5 layout are identical.

Tasks:
    lift    -> classic robosuite Lift (static setup; recommended for the baseline)
    liftrand/canrand/squarerand -> the fork's randomized variants (setup is pinned
              to the nominal layout by fix_scene_setup(); tasks stay random)

Example:
    python gen_camera_poses.py --task lift --train_demos 50 --test_demos 10
    python gen_demos_with_cameras.py --task lift --num_demos 10 \
        --poses_json camera_poses/custom_cameras.json --output_dir demos
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Import the motion planning controllers from the existing file (no repo changes)
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "script_robosuite_demos")
sys.path.insert(0, SCRIPT_DIR)

from mp_lift_abs import LiftAbsMotionPlanningController  # noqa: E402
from mp_can_abs import CanAbsMotionPlanningController    # noqa: E402
from mp_square_abs import SquareAbsMotionPlanningController  # noqa: E402

VERBOSE = False

EEF_SITE_NAME = "gripper0_right_grip_site"

TASK_TO_ENV = {
    "lift": "Lift",  # classic robosuite task: static setup, random object placement
    "liftrand": "LiftRand",
    "canrand": "CanRand",
    "squarerand": "SquareRand",
}

# The mujoco camera that gets repositioned per fixed rig camera. "agentview"
# exists in every task's arena; we reuse it (the same trick policy_robosuite
# uses at train time) so no asset XMLs are touched.
RENDER_CAMERA = "agentview"


def get_eef_site_pose(env):
    """Returns (pos, axis-angle) for the EEF site in world frame."""
    site_id = env.sim.model.site_name2id(EEF_SITE_NAME)
    pos = env.sim.data.site_xpos[site_id].copy()
    rot_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
    aa = T.quat2axisangle(T.mat2quat(rot_mat))
    return pos, aa


def fix_scene_setup(env):
    """Pin the visual SETUP of the *_rand envs, keep the TASK random.

    The fork's envs re-sample two visual planes at every env.reset(): the
    table-top plane (plane_sampler / table_plane_sampler) and the floor/tile
    plane (floor_plane_sampler). The robot base and the table body are already
    static. Pinning these samplers in place fixes the background while object
    placement (ObjectSampler) and the robot's end-effector pose
    (initialization_noise="default") stay random.

    Two mechanisms, both applied here:
    1. UniformRandomSampler stores x_range / y_range / rotation as plain
       attributes, so setting them after env creation is enough; rotation=[0,0]
       means a fixed yaw of 0 (the nominal layout, center of the random range).
    2. Disable hard reset: with the fork's default hard_reset=True, every
       env.reset() destroys the env and re-runs _load_model, which RE-CREATES
       the samplers with their original random ranges (wiping the pinning).
       With hard_reset=False, reset() only does sim.reset() + _reset_internal(),
       so the pinned samplers persist and the objects/planes are re-sampled
       (task stays random) from the pinned ranges (setup stays fixed).
    """
    pinned = []
    for attr in ("plane_sampler", "table_plane_sampler", "floor_plane_sampler"):
        sampler = getattr(env, attr, None)
        if sampler is None:
            continue
        sampler.x_range = [0.0, 0.0]
        sampler.y_range = [0.0, 0.0]
        sampler.rotation = [0.0, 0.0]  # None = uniform random yaw
        pinned.append(attr)
    if pinned:
        # Only the *_rand envs have these samplers. Classic tasks (e.g. Lift)
        # have none, so fix_scene_setup is a no-op there and the env behaves
        # exactly as origin robosuite.
        env.hard_reset = False
        print(f"[fix_scene_setup] pinned setup samplers: {pinned}; hard_reset=False")
    else:
        print("[fix_scene_setup] no setup samplers found (classic task); leaving env untouched")


def create_demo_env(task: str, image_size: int):
    """Same env as gen_robosuite_format_demo.py but with an offscreen renderer.

    For the *_rand tasks the visual setup (table-top plane + floor plane) is
    pinned to the nominal layout via fix_scene_setup(); object placement and
    the end-effector pose remain random. For classic tasks (e.g. Lift) the
    setup is already static, so fix_scene_setup() is a no-op.
    """
    controller_configs = load_composite_controller_config(robot="Panda")
    controller_configs["body_parts"]["right"]["input_type"] = "absolute"
    controller_configs["body_parts"]["right"]["input_ref_frame"] = "world"

    env = suite.make(
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
        camera_heights=image_size,
        camera_widths=image_size,
        controller_configs=controller_configs,
    )
    fix_scene_setup(env)
    return env


class MultiCamRenderer:
    """Renders the fixed camera rig by repositioning RENDER_CAMERA per view.

    Args:
        env: robosuite env (must have has_offscreen_renderer=True)
        camera_names: list of names, e.g. ["left", "right", "top", "bottom", "front"]
        poses: list of 4x4 cam-to-world matrices, aligned with camera_names
        image_size: square render resolution
    """

    def __init__(self, env, camera_names, poses, image_size):
        self.env = env
        self.names = camera_names
        self.poses = [np.asarray(p, dtype=np.float64) for p in poses]
        self.image_size = image_size
        self.cam_id = env.sim.model.camera_name2id(RENDER_CAMERA)
        assert len(self.names) == len(self.poses), "names and poses must match"

    def render_all(self):
        """Returns dict {camera_name: uint8 [H, W, 3] image} for the current sim state."""
        imgs = {}
        for name, pose in zip(self.names, self.poses):
            self.env.sim.model.cam_pos[self.cam_id] = pose[:3, 3]
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()  # [x, y, z, w]
            self.env.sim.model.cam_quat[self.cam_id] = [quat[3], quat[0], quat[1], quat[2]]
            self.env.sim.forward()
            img = self.env.sim.render(
                camera_name=RENDER_CAMERA, height=self.image_size, width=self.image_size, depth=False
            )
            imgs[name] = np.flipud(img).copy()  # flip to match image/row conventions
        return imgs

    def intrinsics(self):
        """3x3 intrinsics, computed from the render camera's fovy (same as policy)."""
        fovy = self.env.sim.model.cam_fovy[self.cam_id] * np.pi / 180.0
        f = self.image_size / (2 * np.tan(fovy / 2))
        return np.array([[f, 0, self.image_size / 2], [0, f, self.image_size / 2], [0, 0, 1]])


def extract_observation_data(obs, env):
    """Extract observation data in the same format as reference dataset."""
    obs_data = {}
    obs_data["object"] = obs["object-state"]
    obs_data["robot0_eef_pos"] = obs["robot0_eef_pos"]
    obs_data["robot0_eef_quat"] = obs["robot0_eef_quat"]
    obs_data["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"]
    obs_data["robot0_gripper_qvel"] = obs["robot0_gripper_qvel"]
    obs_data["robot0_joint_pos"] = obs["robot0_joint_pos"]
    obs_data["robot0_joint_pos_cos"] = obs["robot0_joint_pos_cos"]
    obs_data["robot0_joint_pos_sin"] = obs["robot0_joint_pos_sin"]
    obs_data["robot0_joint_vel"] = obs["robot0_joint_vel"]

    eef_site_id = env.sim.model.site_name2id(EEF_SITE_NAME)
    env.sim.forward()  # Ensure state is updated
    body_id = env.sim.model.site_bodyid[eef_site_id]
    obs_data["robot0_eef_vel_lin"] = env.sim.data.cvel[body_id][:3].copy()
    obs_data["robot0_eef_vel_ang"] = env.sim.data.cvel[body_id][3:].copy()
    return obs_data


def generate_single_demo(demo_id, action_spaces, renderer, seed=None, task: str = "liftrand"):
    """Generate a single demo; identical trajectory to gen_robosuite_format_demo.py,
    plus one image per fixed camera at every obs timestep."""
    if VERBOSE:
        print(f"Generating demo {demo_id}...")

    if seed is not None:
        np.random.seed(seed + demo_id)

    env = renderer.env
    env.reset()

    if task == "lift":
        mp_controller = LiftAbsMotionPlanningController(env)  # same planner, classic env
    elif task == "liftrand":
        mp_controller = LiftAbsMotionPlanningController(env)
    elif task == "canrand":
        mp_controller = CanAbsMotionPlanningController(env)
    elif task == "squarerand":
        mp_controller = SquareAbsMotionPlanningController(env)
    else:
        raise ValueError(f"Unknown task: {task}")

    episode_common = {
        "obs": defaultdict(list),
        "next_obs": defaultdict(list),
        "rewards": [],
        "dones": [],
        "states": [],
    }
    # Per-camera image stacks: obs has T+1 frames, next_obs T (see below)
    obs_images = {name: [] for name in renderer.names}

    actions_by_space = {space: [] for space in action_spaces}

    prev_abs_pose = None
    starting_abs_eef_action = None

    obs = env._get_observations()
    initial_obs_data = extract_observation_data(obs, env)

    prev_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()

    success = False
    max_steps = 400
    reward = 0

    for step in range(max_steps):
        osc_action = mp_controller.get_real_time_action(step)
        pre_site_pos, pre_site_aa = get_eef_site_pose(env)
        if starting_abs_eef_action is None:
            starting_abs_eef_action = osc_action.copy()

        # Record current observation: proprio + one image per fixed camera
        for key, value in initial_obs_data.items():
            episode_common["obs"][key].append(value.copy())
        for name, img in renderer.render_all().items():
            obs_images[name].append(img)

        current_state = env.sim.get_state().flatten()
        episode_common["states"].append(current_state.copy())

        next_obs, reward, done, info = env.step(osc_action)
        next_obs_data = extract_observation_data(next_obs, env)

        for key, value in next_obs_data.items():
            episode_common["next_obs"][key].append(value.copy())

        for space in action_spaces:
            if space == "joint_abs":
                joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
                actions_by_space[space].append(np.concatenate([joint_pos, osc_action[-1:]]))
            elif space == "joint_delta":
                current_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
                joint_delta = current_joint_pos - prev_joint_pos
                actions_by_space[space].append(np.concatenate([joint_delta, osc_action[-1:]]))
            elif space == "eef_abs":
                actions_by_space[space].append(osc_action.copy())
            elif space == "eef_delta":
                abs_pose = osc_action[0:6]
                if prev_abs_pose is None:
                    prev_abs_pose = np.concatenate([pre_site_pos, pre_site_aa])
                delta_pose = abs_pose - prev_abs_pose
                actions_by_space[space].append(np.concatenate([delta_pose, osc_action[-1:]]))
            else:
                raise ValueError(f"Invalid action space: {space}")

        prev_joint_pos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
        prev_abs_pose = osc_action[0:6].copy()

        episode_common["rewards"].append(reward)
        episode_common["dones"].append(0)

        initial_obs_data = next_obs_data

        if env._check_success():
            if not success:
                success = True
                success_step = step
            elif step >= success_step + 20:
                break

        if mp_controller.done:
            break

    env.close()

    # Pad the last action to match states length (per action space)
    for space in action_spaces:
        if len(actions_by_space[space]) < len(episode_common["states"]):
            actions_by_space[space].append(actions_by_space[space][-1].copy())

    # Images: obs[t] is the pre-action state (T+1 frames); next_obs[t] is the
    # post-action state == pre-action state of t+1, so next_obs images are the
    # obs images shifted by one (T frames, aligned with the other next_obs keys).
    obs_images_np = {name: np.stack(frames) for name, frames in obs_images.items()}
    next_images_np = {name: frames[1:] for name, frames in obs_images_np.items()}

    common_np = {}
    common_np["obs"] = {k: np.array(v) for k, v in episode_common["obs"].items()}
    common_np["next_obs"] = {k: np.array(v) for k, v in episode_common["next_obs"].items()}
    common_np["rewards"] = np.array(episode_common["rewards"])
    common_np["dones"] = np.array(episode_common["dones"], dtype=np.int64)
    common_np["states"] = np.array(episode_common["states"])

    final_episode_data_by_space = {}
    for space in action_spaces:
        obs_with_images = dict(common_np["obs"])
        next_obs_with_images = dict(common_np["next_obs"])
        for name in renderer.names:
            obs_with_images[f"{name}_image"] = obs_images_np[name]
            next_obs_with_images[f"{name}_image"] = next_images_np[name]
        final_episode_data_by_space[space] = {
            "obs": obs_with_images,
            "next_obs": next_obs_with_images,
            "actions": np.array(actions_by_space[space]),
            "rewards": common_np["rewards"],
            "dones": common_np["dones"],
            "states": common_np["states"],
        }
        if space == "eef_delta":
            final_episode_data_by_space[space]["starting_abs_action"] = starting_abs_eef_action

    if VERBOSE:
        print(f"Demo {demo_id}: {'SUCCESS' if success else 'FAILED'}")
    return final_episode_data_by_space, success


def generate_demos(num_demos=10, output_files=None, action_spaces=None, seed=None,
                   task: str = "liftrand", poses_json=None, image_size=128, output_dir=None):
    """Generate demos with the fixed multi-camera rig and save per-space HDF5 files."""
    assert len(output_files) == len(action_spaces), "output_files must match action_spaces"

    # Load the 5 fixed camera poses
    with open(poses_json) as f:
        rig = json.load(f)
    camera_names = rig["camera_names"]
    poses = rig["poses"]

    output_dir = os.path.abspath(output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos"))
    os.makedirs(output_dir, exist_ok=True)
    output_paths = [os.path.join(output_dir, fname) for fname in output_files]

    all_demos_by_space = {space: {} for space in action_spaces}
    successful_demos = 0

    for i in tqdm(range(num_demos), desc=f"Generating {task} demos ({camera_names})", unit="demo"):
        # A fresh env per demo (as the original script does) so the rig renderer
        # is bound to the correct sim.
        env = create_demo_env(task, image_size)
        renderer = MultiCamRenderer(env, camera_names, poses, image_size)
        demo_data_by_space, success = generate_single_demo(i, action_spaces, renderer, seed=seed, task=task)
        for space in action_spaces:
            all_demos_by_space[space][f"demo_{i}"] = demo_data_by_space[space]
        if success:
            successful_demos += 1

    if VERBOSE:
        print(f"\nGenerated {num_demos} demos, {successful_demos} successful")

    env_name_meta = TASK_TO_ENV[task]
    for space, out_path in zip(action_spaces, output_paths):
        if space in ("joint_abs", "joint_delta"):
            controller_config = {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "JOINT_POSITION",
                        "input_type": "absolute",
                        "interpolation": None,
                        "gripper": {"type": "GRIP"},
                    }
                },
            }
        else:  # eef_delta / eef_abs
            controller_config = {
                "type": "BASIC",
                "body_parts": {
                    "right": {
                        "type": "OSC_POSE",
                        "input_type": "absolute",
                        "input_ref_frame": "world",
                        "interpolation": None,
                        "gripper": {"type": "GRIP"},
                    }
                },
            }

        env_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "controller_configs": controller_config,
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": image_size,
            "camera_widths": image_size,
            "reward_shaping": False,
        }
        env_args = json.dumps({"env_name": env_name_meta, "env_version": "1.4.1", "type": 1, "env_kwargs": env_kwargs})

        demos_for_space = all_demos_by_space[space]
        total_timesteps = sum(len(d["actions"]) for d in demos_for_space.values())

        # Compute intrinsics from a throwaway renderer (same for all cameras)
        probe_env = create_demo_env(task, image_size)
        try:
            probe_renderer = MultiCamRenderer(probe_env, camera_names, poses, image_size)
            intrinsics = probe_renderer.intrinsics().tolist()
        finally:
            probe_env.close()

        with h5py.File(out_path, "w") as f:
            data_group = f.create_group("data")
            data_group.attrs["env_args"] = env_args
            data_group.attrs["total"] = np.int64(total_timesteps)
            data_group.attrs["action_space"] = space
            data_group.attrs["camera_names"] = json.dumps(camera_names)
            data_group.attrs["camera_poses"] = json.dumps(poses)
            data_group.attrs["camera_intrinsics"] = json.dumps(intrinsics)
            data_group.attrs["camera_image_size"] = json.dumps([image_size, image_size])

            for demo_name, demo_data in demos_for_space.items():
                demo_group = data_group.create_group(demo_name)
                demo_group.create_dataset("actions", data=demo_data["actions"])
                demo_group.create_dataset("rewards", data=demo_data["rewards"])
                demo_group.create_dataset("dones", data=demo_data["dones"])
                demo_group.create_dataset("states", data=demo_data["states"])
                if space == "eef_delta":
                    demo_group.create_dataset("starting_abs_action", data=demo_data["starting_abs_action"])

                obs_group = demo_group.create_group("obs")
                next_obs_group = demo_group.create_group("next_obs")
                for key, value in demo_data["obs"].items():
                    if key.endswith("_image"):
                        # Compressed chunks for random per-frame access
                        obs_group.create_dataset(key, data=value, chunks=(1, image_size, image_size, 3),
                                                 compression="gzip", compression_opts=4)
                    else:
                        obs_group.create_dataset(key, data=value)
                for key, value in demo_data["next_obs"].items():
                    if key.endswith("_image"):
                        next_obs_group.create_dataset(key, data=value, chunks=(1, image_size, image_size, 3),
                                                      compression="gzip", compression_opts=4)
                    else:
                        next_obs_group.create_dataset(key, data=value)

    if VERBOSE:
        for out_path in output_paths:
            print(f"Saved {num_demos} demos to {out_path}")
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="liftrand", choices=list(TASK_TO_ENV))
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--poses_json", type=str, required=True,
                        help="path to custom_cameras.json from gen_camera_poses.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="where to save the hdf5 files (default: ./demos next to this script)")
    parser.add_argument("--image_size", type=int, default=128,
                        help="render resolution for the saved images (px, square)")
    parser.add_argument(
        "--output_files", type=str, nargs="+", default=["eef_abs.hdf5", "eef_delta.hdf5", "joint_abs.hdf5", "joint_delta.hdf5"],
    )
    parser.add_argument(
        "--action_spaces", type=str, nargs="+", default=["eef_abs", "eef_delta", "joint_abs", "joint_delta"],
        choices=["eef_delta", "eef_abs", "joint_abs", "joint_delta"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_demos(
        num_demos=args.num_demos,
        output_files=args.output_files,
        action_spaces=args.action_spaces,
        seed=args.seed,
        task=args.task,
        poses_json=args.poses_json,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )
