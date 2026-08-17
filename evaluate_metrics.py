import os
import json
import math
import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gymnasium.envs.registration import register
except ImportError:
    from gym.envs.registration import register

from reconsimulator.envs.nus import ReconSimulator
from GSDrive.env import ReconNusPPOEnv
from train import (
    TransfuserConfig,
    V2TransfuserModelWrapper,
    TRAJ_ANCHOR_PATH,
    TrajectoryProbeReward,
    arrange_agents,
)



def make_env(cuda: int = 0, scene: int = 0, debug: bool = False,
             resize_shape: Tuple[int, int] = (112, 200)) -> ReconNusPPOEnv:
    """Construct environment."""
    return ReconNusPPOEnv(
        cuda=cuda,
        scene=scene,
        debug=debug,
        resize_shape=resize_shape,
    )

def _ego_yaw(env: ReconNusPPOEnv) -> float:
    """Read current ego yaw (radians) from environment pose."""
    rot = env.base_env.start_ego[:3, :3]
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def _ego_pos_xz(env: ReconNusPPOEnv) -> np.ndarray:
    """Read ego position in (x, z) world frame."""
    return env.base_env.start_ego[:3, 3][[0, 2]].copy()

class EpisodeBuffer:
    """
    Stores all the per-step signals required to compute the target metrics.
    One instance per episode.
    """

    def __init__(self, dt: float = 0.5):
        self.dt = dt
        self.rewards: List[float] = []
        self.speeds: List[float] = []           # linear_v [m/s]
        self.accs: List[float] = []             # acceleration [m/s²]
        self.jerks: List[float] = []            # longitudinal jerk [m/s³]
        self.yaw_rates: List[float] = []        # yaw velocity [rad/s]
        self.yaw_accels: List[float] = []       # yaw acceleration
        self.yaw_jerks: List[float] = []        # yaw jerk [rad/s³]
        self.collisions: List[bool] = []        # dynamic collision flag
        self.lateral_pos: List[float] = []      # lateral world position
        # internal kinematics state
        self._prev_speed = 0.0
        self._prev_acc = 0.0
        self._prev_yaw = 0.0
        self._prev_yaw_rate = 0.0
        self._prev_yaw_acc = 0.0

    def record_step(self, reward: float, info: dict, env: ReconNusPPOEnv):
        """
        Append one transition to the buffer.
        Must be called AFTER env.step() so that env.prev_speed / info reflects
        the new state.
        """
        # ---------- reward ----------
        self.rewards.append(float(reward))

        # ---------- speed / acceleration / jerk ----------
        speed = float(env.prev_speed)            # already Δ-position / dt
        acc = speed - self._prev_speed           # Δv
        jerk = acc - self._prev_acc              # Δa

        self.speeds.append(speed)
        self.accs.append(acc)
        self.jerks.append(jerk)

        # ---------- steering yaw / yaw-rate / yaw-jerk ----------
        yaw_now = _ego_yaw(env)
        yaw_rate = float(info.get("yaw_v", 0.0))
        yaw_acc = yaw_rate - self._prev_yaw_rate
        yaw_jerk = yaw_acc - self._prev_yaw_acc

        self.yaw_rates.append(yaw_rate)
        self.yaw_accels.append(yaw_acc)
        self.yaw_jerks.append(yaw_jerk)

        # ---------- lateral world position (for lane-change detection) ----------
        xz = _ego_pos_xz(env)
        self.lateral_pos.append(float(xz[1]))    # z-axis is lateral in nuScenes

        # ---------- collision flag ----------
        self.collisions.append(bool(info.get("is_dynamic_collision_box", False)))

        # ---------- update internal state ----------
        self._prev_speed = speed
        self._prev_acc = acc
        self._prev_yaw = yaw_now
        self._prev_yaw_rate = yaw_rate
        self._prev_yaw_acc = yaw_acc

    # ------------------------------------------------------------------
    # Aggregate metric computations
    # ------------------------------------------------------------------
    def episode_reward(self) -> float:
        return float(np.sum(self.rewards))

    def mean_driving_speed(self) -> float:
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    def max_abs_acceleration(self) -> float:
        return float(np.max(np.abs(self.accs))) if self.accs else 0.0

    def max_abs_action_jerks(self) -> float:
        """
        Worst-case jerk in m/s³ = max( |longitudinal_jerk|, |yaw_jerk|_scaled )
        We keep them separate for full transparency but report the global max
        to satisfy the spec.
        """
        cand = []
        if self.jerks:
            cand.append(float(np.max(np.abs(self.jerks))))
        if self.yaw_jerks:
            cand.append(float(np.max(np.abs(self.yaw_jerks))))
        return float(max(cand)) if cand else 0.0

    def max_abs_steering_angle(self) -> float:
        """
        MSA ← max |yaw-rate| within the episode, in radians.
        Steering angle is approximated by the per-step yaw change because the
        action space uses a discrete yaw-rate grid.
        """
        return float(np.max(np.abs(self.yaw_rates))) if self.yaw_rates else 0.0

    def lane_changes(self,
                     min_step_separation: int = 4,
                     threshold_m: float = 0.6) -> int:
        """
        Lane change ≈ a sustained lateral displacement of at least `threshold_m`
        followed by a reversal of direction.

        A simple yet robust detector:
          1. Smooth the lateral position with a small moving average.
          2. Compute Δlat per step.
          3. When |Δlat| stays above a fraction of `threshold_m` for
             `min_step_separation` steps, a lane change is counted.
        """
        if len(self.lateral_pos) < min_step_separation + 2:
            return 0

        lat = np.asarray(self.lateral_pos, dtype=np.float32)
        # smooth (window=3)
        kernel = np.ones(3) / 3.0
        smooth = np.convolve(lat, kernel, mode="same")

        # detect sustained lateral motion
        delta = np.diff(smooth)
        sign = np.sign(delta)
        # find runs of consistent sign
        change_count = 0
        run_sign = 0
        run_len = 0
        last_lc_step = -10
        for i, s in enumerate(sign):
            if s == 0:
                continue
            if s == run_sign:
                run_len += 1
            else:
                run_sign = s
                run_len = 1
            if (run_len >= min_step_separation and
                    abs(smooth[i] - smooth[max(0, i - run_len)]) > threshold_m and
                    (i - last_lc_step) > min_step_separation):
                change_count += 1
                last_lc_step = i
                run_sign = 0     # reset to avoid double counting
                run_len = 0
        return int(change_count)

    def collision_occurred(self) -> bool:
        return bool(any(self.collisions))

    def to_dict(self) -> Dict[str, float]:
        return {
            "episode_reward":       self.episode_reward(),
            "mean_driving_speed":   self.mean_driving_speed(),
            "max_abs_acceleration": self.max_abs_acceleration(),
            "lane_changes":         self.lane_changes(),
            "max_abs_action_jerks": self.max_abs_action_jerks(),
            "max_abs_steering_angle": self.max_abs_steering_angle(),
            "collision_occurred":    self.collision_occurred(),
        }

def load_model(model_path: str, env: ReconNusPPOEnv, device: torch.device):
    config = TransfuserConfig()
    model = V2TransfuserModelWrapper(
        config,
        env.observation_space.shape,
        env.action_space,
        plan_anchor_path=TRAJ_ANCHOR_PATH,
        training_stage=False,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def evaluate_episode(env: ReconNusPPOEnv,
                     model,
                     device: torch.device,
                     info_init: dict,
                     dt: float = 0.5) -> EpisodeBuffer:
    """Run one full episode in `env` using `model`, return an EpisodeBuffer."""
    buf = EpisodeBuffer(dt=dt)
    obs, _info = env.reset()
    obs = obs.astype(np.float32)

    camera_intrinsics, camera_extrinsics = _info["intrinsics"], _info["extrinsics"]
    agents = arrange_agents([_info["agents"]])

    done = False
    while not done:
        obs_tensor       = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        agents_tensor    = agents.to(dtype=torch.float32, device=device).unsqueeze(0)
        intr_tensor      = camera_intrinsics.to(dtype=torch.float32, device=device).unsqueeze(0)
        extr_tensor      = camera_extrinsics.to(dtype=torch.float32, device=device).unsqueeze(0)
        target_tensor    = _info["target_trajs"].to(dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                obs_tensor, agents_tensor, target_tensor, intr_tensor, extr_tensor,
            )
            action_tensor = outputs[3]

        action = action_tensor.squeeze(0).cpu().numpy()
        # Env expects 3-element flag layout [ax, ay, brake];]
        next_obs, reward, terminated, truncated, info = env.step(
            [int(action[0]), int(action[1]), 0]
        )

        buf.record_step(reward, info, env)

        obs = next_obs.astype(np.float32)
        camera_intrinsics, camera_extrinsics = info["intrinsics"], info["extrinsics"]
        agents = arrange_agents([info["agents"]])
        _info = info
        done = terminated or truncated
    return buf


# ============================================================================
# Reporting
# =========================================================================
TARGET_METRICS = [
    ("episode_reward",         "Episode Reward (ER ↑)",       "mean"),
    ("mean_driving_speed",     "Driving Speed [m/s] (DS ↑)",  "mean"),
    ("max_abs_acceleration",   "Max Acceleration [m/s²] (MA ↓)", "mean"),
    ("lane_changes",           "Lane Changes (LC ↑)",         "mean"),
    ("max_abs_action_jerks",   "Max Action Jerks [m/s³] (MAJ ↓)", "mean"),
    ("max_abs_steering_angle", "Max Steering Angle [rad] (MSA ↓)", "mean"),
    ("collision_occurred",     "Collision Rate (CR ↓)",       "rate"),
]


def aggregate(per_episode: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std/rate over all episodes."""
    agg = {}
    for key, _, kind in TARGET_METRICS:
        vals = np.array([ep[key] for ep in per_episode], dtype=np.float64)
        if kind == "rate":
            agg[key] = {
                "rate":  float(np.mean(vals)),
                "count": int(np.sum(vals)),
                "n":     int(len(vals)),
            }
        else:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
                "median": float(np.median(vals)),
                "n":    int(len(vals)),
            }
    return agg


def print_report(agg: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    for key, label, kind in TARGET_METRICS:
        if kind == "rate":
            v = agg[key]
            print(f"  {label:<42s}  {v['rate']*100:7.2f}% "
                  f"({v['count']}/{v['n']})")
        else:
            v = agg[key]
            print(f"  {label:<42s}  mean={v['mean']:.4f} ± {v['std']:.4f} "
                  f"(min={v['min']:.4f}, max={v['max']:.4f})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO/BC driving model")
    parser.add_argument("--model_path",  type=str,   required=True,
                        help="Path to .pt checkpoint (PPO or BC)")
    parser.add_argument("--scenes",      type=int,   nargs="+", default=None,
                        help="Scene ids to evaluate on. Default = scene 0 only.")
    parser.add_argument("--episodes",    type=int,   default=10,
                        help="Number of episodes per scene")
    parser.add_argument("--cuda",        type=int,   default=0)
    parser.add_argument("--resize_h",    type=int,   default=112)
    parser.add_argument("--resize_w",    type=int,   default=200)
    parser.add_argument("--device",      type=str,   default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--dt",          type=float, default=0.5,
                        help="Environment step duration [s]")
    parser.add_argument("--out_json",    type=str,   default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = args.scenes if args.scenes else [0]
    resize_shape = (args.resize_h, args.resize_w)

    register(id="ReconSimulator-v0",
             entry_point="reconsimulator.envs.nus:ReconSimulator")

    env = make_env(cuda=args.cuda, scene=int(scenes[0]),
                   debug=False, resize_shape=resize_shape)
    model = load_model(args.model_path, env, device)

    all_episodes: List[Dict[str, float]] = []
    for sc in scenes:
        env.set_scene(int(sc))
        for ep in range(args.episodes):
            buf = evaluate_episode(env, model, device, info_init=None, dt=args.dt)
            m = buf.to_dict()
            m["scene"] = int(sc)
            m["episode"] = ep
            all_episodes.append(m)
            print(f"[scene {sc} ep {ep+1}]  "
                  f"R={m['episode_reward']:+.3f}  "
                  f"v={m['mean_driving_speed']:.2f}m/s  "
                  f"|a|max={m['max_abs_acceleration']:.2f}m/s²  "
                  f"|jerk|max={m['max_abs_action_jerks']:.2f}m/s³  "
                  f"|steer|max={m['max_abs_steering_angle']:.2f}rad  "
                  f"LC={m['lane_changes']}  "
                  f"coll={int(m['collision_occurred'])}")

    agg = aggregate(all_episodes)
    print_report(agg)

    if args.out_json:
        out = {
            "model_path": args.model_path,
            "scenes": scenes,
            "episodes_per_scene": args.episodes,
            "per_episode": all_episodes,
            "aggregate": agg,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".",
                    exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.out_json}")


if __name__ == "__main__":
    main()