import numpy as np
import math
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False
import gymnasium as gym
from reconsimulator.envs.nus import ReconSimulator


class ReconNusPPOEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, cuda=0, scene=0, debug=False, resize_shape=(112, 200)):
        super().__init__()
        self.base_env = ReconSimulator(cuda=cuda, scene=scene, debug=debug)
        self.resize_shape = resize_shape
        self.keys_order = [
            "front", "front_left", "front_right",
            "back_left", "back_right", "back",
        ]
        h = self.base_env.h
        w = self.base_env.w
        c = 3 * len(self.keys_order)

        if self.resize_shape is not None:
            rh, rw = self.resize_shape
            obs_shape = (c, rh, rw)
        else:
            obs_shape = (c, h, w)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.base_env.action_space

        self.prev_distance = 0.0
        self.last_ego_pos = None
        self.prev_speed = 0.0
        self.prev_acc = 0.0
        self.prev_yaw = 0.0
        self.distance_coeff = 2.0
        self.comfort_coeff = 0.05

    def _compute_yaw(self):
        rot = self.base_env.start_ego[:3, :3]
        return float(math.atan2(rot[1, 0], rot[0, 0]))

    def _compute_ego2match_yaw_degrees(self):
        pos = self.base_env.start_ego[:3, 3][[0, 2]]
        expert_pairs = np.stack(self.base_env.expert_pair, axis=0)
        dists = np.linalg.norm(expert_pairs - pos[None, :], axis=1) + 1e-8
        nearest_idx = int(np.argmin(dists))
        current_yaw = self._compute_yaw()
        expert_rot = self.base_env.expert_world_all[nearest_idx][:3, :3]
        expert_yaw = float(math.atan2(expert_rot[1, 0], expert_rot[0, 0]))
        diff = current_yaw - expert_yaw
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return float(diff * 180.0 / math.pi)

    def _obs_dict_to_tensor(self, obs_dict):
        imgs = [obs_dict[k] for k in self.keys_order]
        imgs = [img.astype(np.float32) / 255.0 for img in imgs]

        if self.resize_shape is not None:
            rh, rw = self.resize_shape
            resized = []
            if _CV2_AVAILABLE:
                for img in imgs:
                    img_resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
                    resized.append(img_resized)
            else:
                import torch.nn.functional as F
                for img in imgs:
                    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                    t_resized = F.interpolate(t, size=(rh, rw), mode="area")
                    img_resized = t_resized.squeeze(0).permute(1, 2, 0).numpy()
                    resized.append(img_resized)
            imgs = resized

        stacked = np.concatenate(imgs, axis=2)
        stacked = np.transpose(stacked, (2, 0, 1))
        return stacked

    def _compute_distance_to_expert(self):
        ego_pos = self.base_env.start_ego[:3, 3][[0, 2]]
        expert_pairs = np.stack(self.base_env.expert_pair, axis=0)
        dists = np.linalg.norm(expert_pairs - ego_pos[None, :], axis=1) + 1e-8
        return float(dists.min())

    def _compute_comfort_penalty(self):
        """Compute comfort penalty - only penalize large acceleration/jerk"""
        pos = self.base_env.start_ego[:3, 3]
        if self.last_ego_pos is None:
            self.last_ego_pos = pos.copy()
        delta = pos[[0, 2]] - self.last_ego_pos[[0, 2]]
        speed = float(np.linalg.norm(delta)) + 1e-8
        acc = speed - self.prev_speed
        jerk = acc - self.prev_acc
        
        acc_threshold = 2.0
        jerk_threshold = 5.0
        acc_penalty = max(0, abs(acc) - acc_threshold)
        jerk_penalty = max(0, abs(jerk) - jerk_threshold)
        comfort_penalty = self.comfort_coeff * (acc_penalty + jerk_penalty)
        
        self.last_ego_pos = pos.copy()
        self.prev_speed = speed
        self.prev_acc = acc
        return comfort_penalty

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        obs_raw, info = self.base_env.reset(seed=seed, options=options)
        obs = self._obs_dict_to_tensor(obs_raw)
   

        self.prev_distance = self._compute_distance_to_expert()
        self.last_ego_pos = self.base_env.start_ego[:3, 3].copy()
        self.prev_speed = 0.0
        self.prev_acc = 0.0
        self.prev_yaw = self._compute_yaw()
        return obs, info # def reset and step both self._get_info()

    def step(self, action):
        ax = int(action[0])
        ay = int(action[1])
        raw_action = [ax, ay, 0]
        obs_raw, terminated, truncated, info = self.base_env.step(raw_action)
        obs = self._obs_dict_to_tensor(obs_raw)

        distance = self._compute_distance_to_expert()
        distance_reward = (self.prev_distance - distance) * self.distance_coeff
        comfort_penalty = self._compute_comfort_penalty()

        collision_penalty = 0.0
        d_max = 5.0

        if distance > d_max:
            collision_penalty += 1.0
            truncated = True
            offroad_info = {
                "frame_idx": self.base_env.now_frame,
                "type": "off_road",
            }
        else:
            offroad_info = None

        if hasattr(self.base_env, "check_coliision"):
            if self.base_env.check_coliision():
                collision_penalty += 5.0
                truncated = True
           

        if info is None:
            info = {}
        info = dict(info)
        info["distance"] = distance
        info["distance_reward"] = distance_reward
        info["comfort_penalty"] = comfort_penalty
        info["collision_penalty"] = collision_penalty
        info["raw_obs"] = obs_raw

        if "is_dynamic_collision_box" not in info:
            info["is_dynamic_collision_box"] = False
        if "static_collision_score" not in info:
            info["static_collision_score"] = 0.0

        collision_info = None
        if hasattr(self.base_env, "last_collision_frame") and self.base_env.last_collision_type is not None:
            collision_info = {
                "frame_idx": self.base_env.last_collision_frame,
                "type": self.base_env.last_collision_type,
            }
        elif offroad_info is not None:
            collision_info = offroad_info

        info["collision"] = collision_info

        current_yaw = self._compute_yaw()
        yaw_v = current_yaw - self.prev_yaw
        self.prev_yaw = current_yaw
        info["yaw_v"] = float(yaw_v)
        info["ego2match_yaw_degrees"] = float(self._compute_ego2match_yaw_degrees())

        survival_bonus = 1.0  
        
        if distance_reward > 0:
            distance_reward = distance_reward * 1.5
        else:
            distance_reward = max(distance_reward, -0.2)  
        
        if distance < 0.5:
            proximity_bonus = 0.5
        elif distance < 1.0:
            proximity_bonus = 0.3
        elif distance < 2.0:
            proximity_bonus = 0.1
        else:
            proximity_bonus = 0.0
        
        progress_bonus = 0.0
        if hasattr(self.base_env, 'expert_pair') and len(self.base_env.expert_pair) > 0:
            # Find direction to next expert point
            ego_pos = self.base_env.start_ego[:3, 3][[0, 2]]
            expert_pairs = np.stack(self.base_env.expert_pair, axis=0)
            dists = np.linalg.norm(expert_pairs - ego_pos[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))
            
            # Check if we've advanced along the trajectory
            if nearest_idx > 0 and self.last_ego_pos is not None:
                prev_dists = np.linalg.norm(expert_pairs - self.last_ego_pos[[0, 2]][None, :], axis=1)
                prev_nearest_idx = int(np.argmin(prev_dists))
                if nearest_idx > prev_nearest_idx:
                    # We've moved forward along the expert trajectory
                    progress_bonus = 0.3 * (nearest_idx - prev_nearest_idx)
        
        alignment_bonus = 0.0
        yaw_diff_deg = abs(info["ego2match_yaw_degrees"])
        if yaw_diff_deg < 5:  
            alignment_bonus = 0.2
        elif yaw_diff_deg < 15:  
            alignment_bonus = 0.1
        
        positive_rewards = survival_bonus + proximity_bonus + progress_bonus + alignment_bonus
        shaped_rewards = distance_reward  
        
        if collision_penalty > 0 and not truncated:
            collision_penalty = collision_penalty * 0.3
        
        reward = positive_rewards + shaped_rewards - collision_penalty - comfort_penalty
        
        # Store metrics for logging
        info["reward_breakdown"] = {
            "survival_bonus": survival_bonus,
            "distance_reward": distance_reward,
            "proximity_bonus": proximity_bonus,
            "progress_bonus": progress_bonus,
            "alignment_bonus": alignment_bonus,
            "collision_penalty": collision_penalty,
            "comfort_penalty": comfort_penalty,
            "total_reward": reward,
        }
        
        self.prev_distance = distance

        return obs, reward, terminated, truncated, info

    def set_scene(self, scene: int):
        self.base_env.update(scene=int(scene))

    def _angle_diff(self, a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _extract_anchor_xy(self, anchor):

        if anchor.ndim == 2 and anchor.shape[-1] >= 2:
            return float(anchor[-1, 0].item()), float(anchor[-1, 1].item())
        elif anchor.ndim == 1:
            if anchor.numel() == 2:
                return float(anchor[0].item()), float(anchor[1].item())
            elif anchor.numel() % 2 == 0:
                a2 = anchor.view(-1, 2)
                return float(a2[-1, 0].item()), float(a2[-1, 1].item())
            else:
                return float(anchor[0].item()), float(anchor[1].item()) if anchor.numel() > 1 else 0.0
        else:
            return 0.0, 0.0

    

    def verify_action_traj_alignment(self, ax_idx, ay_idx, target_trajs, prev_ego, next_ego, verbose=True):

        import math

        # 1. Get the actual ego motion from prev_ego to next_ego
        delta = np.linalg.inv(prev_ego) @ next_ego
        actual_ty = float(delta[0, 3])
        actual_tx = float(delta[1, 3])
        actual_yaw = float(math.atan2(delta[1, 0], delta[0, 0]))

        # 2. Get the selected anchor's pose
        anchor_idx = int(ax_idx * self.base_env.y_anchor + ay_idx)
        anchor = self.base_env.plan_anchors[anchor_idx]
        anchor_ty, anchor_tx = self._extract_anchor_xy(anchor)
        anchor_yaw = float(self.base_env.plan_anchors_yaw[anchor_idx].item())
        
        # 3. Compute position and yaw errors
        pos_error = math.sqrt((actual_tx - anchor_tx) ** 2 + (actual_ty - anchor_ty) ** 2)
        yaw_error = abs(self._angle_diff(actual_yaw, anchor_yaw))    

        # 4. Get target trajectory info
        # target_trajs is delta trajectory (relative displacements), so compute cumulative
        if isinstance(target_trajs, torch.Tensor):
            target_trajs_np = target_trajs.cpu().numpy()
        else:
            target_trajs_np = target_trajs

        # Cumulative trajectory (starting from origin)
        cumsum_traj = np.cumsum(target_trajs_np, axis=0)

        # First waypoint displacement
        first_wp_x = float(target_trajs_np[0, 0]) if len(target_trajs_np) > 0 else 0.0
        first_wp_y = float(target_trajs_np[0, 1]) if len(target_trajs_np) > 0 else 0.0
        
        # Final waypoint position
        final_wp_x = float(cumsum_traj[-1, 0]) if len(cumsum_traj) > 0 else 0.0
        final_wp_y = float(cumsum_traj[-1, 1]) if len(cumsum_traj) > 0 else 0.0

        if len(target_trajs_np) >= 2:
            
            dx_first = float(target_trajs_np[1, 0]) if len(target_trajs_np) > 1 else float(target_trajs_np[0, 0])
            dy_first = float(target_trajs_np[1, 1]) if len(target_trajs_np) > 1 else float(target_trajs_np[0, 1])
            target_yaw_from_first = math.atan2(dy_first, dx_first)
        else:
            target_yaw_from_first = 0.0
        
        ego_to_first_wp_error = math.sqrt(
            (actual_tx - first_wp_x) ** 2 + (actual_ty - first_wp_y) ** 2
        )
        
        # Yaw error between actual ego yaw and target traj yaw
        ego_to_target_yaw_error = abs(self._angle_diff(actual_yaw, target_yaw_from_first))
        
        num_points_to_check = min(3, len(target_trajs_np))
        if num_points_to_check > 0:
            cumsum_first_n = np.sum(target_trajs_np[:num_points_to_check], axis=0)
            cumsum_first_n_x = float(cumsum_first_n[0])
            cumsum_first_n_y = float(cumsum_first_n[1])
        else:
            cumsum_first_n_x, cumsum_first_n_y = 0.0, 0.0
        
        # Thresholds for ego-to-target-traj alignment
        EGO_TO_FIRST_WP_THRESHOLD = 0.3
        EGO_TO_TARGET_YAW_THRESHOLD = 0.3
        
        ego_motion_aligned = ego_to_first_wp_error < EGO_TO_FIRST_WP_THRESHOLD
        ego_yaw_aligned = ego_to_target_yaw_error < EGO_TO_TARGET_YAW_THRESHOLD

        if anchor.ndim == 2 and anchor.shape[0] > 1:
            anchor_traj = anchor.cpu().numpy() if hasattr(anchor, 'cpu') else anchor.numpy() if hasattr(anchor, 'numpy') else anchor

            # Compute trajectory error (L2 distance between trajectories)
            if len(anchor_traj) == len(target_trajs_np):
                traj_error = np.sqrt(np.mean((anchor_traj - target_trajs_np) ** 2))
            else:
                # If different lengths, compare up to min length
                min_len = min(len(anchor_traj), len(target_trajs_np))
                traj_error = np.sqrt(np.mean((anchor_traj[:min_len] - target_trajs_np[:min_len]) ** 2))
        else:
            traj_error = float('nan')

        POS_THRESHOLD = 0.2  # meters
        YAW_THRESHOLD = 0.2  # radians (~11 degrees)
        TRAJ_THRESHOLD = 0.5  # meters average error

        pos_aligned = pos_error < POS_THRESHOLD
        yaw_aligned = yaw_error < YAW_THRESHOLD
        traj_aligned = traj_error < TRAJ_THRESHOLD if not math.isnan(traj_error) else True
        is_aligned = pos_aligned and yaw_aligned
        
        # Overall alignment including ego-to-target-traj check
        overall_aligned = is_aligned and ego_motion_aligned and ego_yaw_aligned
        
        results = {
            'is_aligned': is_aligned,
            'overall_aligned': overall_aligned,
            'pos_error': pos_error,
            'yaw_error': yaw_error,
            'traj_error': traj_error,
            'actual_motion': (actual_tx, actual_ty, actual_yaw),
            'anchor_pose': (anchor_tx, anchor_ty, anchor_yaw),
            'target_first_wp': (first_wp_x, first_wp_y),
            'target_final_wp': (final_wp_x, final_wp_y),
            'anchor_idx': anchor_idx,
            'pos_aligned': pos_aligned,
            'yaw_aligned': yaw_aligned,
            'traj_aligned': traj_aligned,
            # Ego motion vs target_trajs alignment
            'ego_to_first_wp_error': ego_to_first_wp_error,
            'ego_to_target_yaw_error': ego_to_target_yaw_error,
            'target_yaw_from_first': target_yaw_from_first,
            'ego_motion_aligned': ego_motion_aligned,
            'ego_yaw_aligned': ego_yaw_aligned,
            'cumsum_first_n': (cumsum_first_n_x, cumsum_first_n_y),
        }

        if verbose:
            # print(f"\n=== Action-Trajectory Alignment Verification ===")
            # print(f"Anchor index: ({ax_idx}, {ay_idx}) -> idx {anchor_idx}")
            # print(f"Actual ego motion: tx={actual_tx:.4f}, ty={actual_ty:.4f}, yaw={actual_yaw:.4f}")
            # print(f"Anchor pose:       tx={anchor_tx:.4f}, ty={anchor_ty:.4f}, yaw={anchor_yaw:.4f}")
            # print(f"Target traj first wp: ({first_wp_x:.4f}, {first_wp_y:.4f})")
            # print(f"Target traj final wp: ({final_wp_x:.4f}, {final_wp_y:.4f})")
            # print(f"Errors: pos={pos_error:.4f}m (threshold={POS_THRESHOLD}), yaw={yaw_error:.4f}rad (threshold={YAW_THRESHOLD})")

            if not math.isnan(traj_error):
                print(f"Trajectory error: {traj_error:.4f}m (threshold={TRAJ_THRESHOLD})")
            print(f"Alignment: {'PASS' if is_aligned else 'FAIL'} (pos={'OK' if pos_aligned else 'FAIL'}, yaw={'OK' if yaw_aligned else 'FAIL'})")
            
            print(f"\n--- Ego Motion vs Target Traj Verification ---")
            print(f"Actual ego motion:    tx={actual_tx:.4f}, ty={actual_ty:.4f}, yaw={actual_yaw:.4f} rad ({math.degrees(actual_yaw):.1f} deg)")
            print(f"Target traj first wp: x={first_wp_x:.4f}, y={first_wp_y:.4f}")
            print(f"Target traj yaw (from first delta): {target_yaw_from_first:.4f} rad ({math.degrees(target_yaw_from_first):.1f} deg)")
            print(f"Ego-to-first-wp error: {ego_to_first_wp_error:.4f}m (threshold={EGO_TO_FIRST_WP_THRESHOLD})")
            print(f"Ego-to-target-yaw error: {ego_to_target_yaw_error:.4f}rad (threshold={EGO_TO_TARGET_YAW_THRESHOLD})")
            print(f"Cumsum first {num_points_to_check} pts: ({cumsum_first_n_x:.4f}, {cumsum_first_n_y:.4f})")
            print(f"Ego motion aligned: {'PASS' if ego_motion_aligned else 'FAIL'}, Yaw aligned: {'PASS' if ego_yaw_aligned else 'FAIL'}")
            print(f"Overall alignment: {'PASS' if overall_aligned else 'FAIL'}")
            print("=" * 50)

        return results

    def compute_expert_action(self, prev_ego, next_ego):
        """
        Compute expert action based on ego vehicle motion.
        
        Args:
            prev_ego: Previous ego pose (4x4 transformation matrix)
            next_ego: Next ego pose (4x4 transformation matrix)
            
        Returns:
            ax_idx: x anchor index for action
            ay_idx: y anchor index for action
        """
        delta = np.linalg.inv(prev_ego) @ next_ego
        yaw = float(math.atan2(delta[1, 0], delta[0, 0]))
        tx = float(delta[0, 3])
        ty = float(delta[1, 3])
        
        x_min, x_max = -4.16, 3.51
        y_min, y_max = -0.65, 10.09
        
        # Normalize tx and ty to [0, 1] range
        tx_norm = (tx - x_min) / (x_max - x_min)
        ty_norm = (ty - y_min) / (y_max - y_min)
        
        num_anchors_x = int(self.action_space.nvec[0])
        num_anchors_y = int(self.action_space.nvec[1])

        ax_idx = int(np.clip(tx_norm * (num_anchors_x - 1), 0, num_anchors_x - 1))
        ay_idx = int(np.clip(ty_norm * (num_anchors_y - 1), 0, num_anchors_y - 1))
        
        return ax_idx, ay_idx

    def save_state(self):
        """
        Save the current environment state for trajectory probing.
        
        Returns:
            state: dict containing all necessary state information to restore
        """
        state = {
            'start_ego': self.base_env.start_ego.copy() if hasattr(self.base_env.start_ego, 'copy') else np.array(self.base_env.start_ego),
            'now_frame': self.base_env.now_frame,
            'last_collision_frame': self.base_env.last_collision_frame,
            'last_collision_type': self.base_env.last_collision_type,
            # Save PPO environment internal state
            'prev_distance': self.prev_distance,
            'last_ego_pos': self.last_ego_pos.copy() if self.last_ego_pos is not None else None,
            'prev_speed': self.prev_speed,
            'prev_acc': self.prev_acc,
            'prev_yaw': self.prev_yaw,
        }
        return state
    
    def restore_state(self, state):
        """
        Restore the environment state from a saved state.
        
        Args:
            state: dict containing previously saved state information
        """
        self.base_env.start_ego = state['start_ego'].copy() if hasattr(state['start_ego'], 'copy') else np.array(state['start_ego'])
        self.base_env.now_frame = state['now_frame']
        self.base_env.last_collision_frame = state['last_collision_frame']
        self.base_env.last_collision_type = state['last_collision_type']
        # Restore PPO environment internal state
        self.prev_distance = state['prev_distance']
        self.last_ego_pos = state['last_ego_pos'].copy() if state['last_ego_pos'] is not None else None
        self.prev_speed = state['prev_speed']
        self.prev_acc = state['prev_acc']
        self.prev_yaw = state['prev_yaw']

    def build_vad_img_metas(self):
        metas = []
        H = int(self.base_env.h)
        W = int(self.base_env.w)
        for i in range(6):
            K = np.array(self.base_env.all_cams[i]['intrinsics'].cpu().numpy() if hasattr(self.base_env.all_cams[i]['intrinsics'], 'cpu') else self.base_env.all_cams[i]['intrinsics'])
            K_pad = np.eye(4, dtype=np.float32)
            K_pad[0, 0] = float(K[0, 0]); K_pad[0, 2] = float(K[0, 2])
            K_pad[1, 1] = float(K[1, 1]); K_pad[1, 2] = float(K[1, 2])
            K_pad[2, 2] = 1.0
            lidar2cam = np.linalg.inv(self.base_env.cam2ego[i]).astype(np.float32)
            lidar2img = (K_pad @ lidar2cam).astype(np.float32)
            metas.append({
                "lidar2img": lidar2img,
                "img_shape": [(H, W)]
            })
        return metas