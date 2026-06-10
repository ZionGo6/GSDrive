import torch
import math
import numpy as np



class TrajectoryProbeReward:
    """
    Trajectory Probe Reward: Uses imitation model's predicted trajectory to probe
    the environment in advance and compute future rewards for current action guidance.
    """
    
    def __init__(
        self,
        num_probe_steps=3,
        discount_factor=0.9,
        collision_penalty_weight=2.0,
        deviation_penalty_weight=1.0,
        comfort_weight=0.5,
        success_bonus=1.0,
        survival_bonus=0.3,          
        progress_weight=0.2,         
        proximity_weight=0.15,       
        alignment_weight=0.1,        
        smoothness_weight=0.05,      
        agent_safety_weight=0.1,     
        goal_progress_weight=0.1,    
        use_dense_rewards=True,      
    ):
        self.num_probe_steps = num_probe_steps
        self.discount_factor = discount_factor
        self.collision_penalty_weight = collision_penalty_weight
        self.deviation_penalty_weight = deviation_penalty_weight
        self.comfort_weight = comfort_weight
        self.success_bonus = success_bonus
        self.survival_bonus = survival_bonus
        self.progress_weight = progress_weight
        self.proximity_weight = proximity_weight
        self.alignment_weight = alignment_weight
        self.smoothness_weight = smoothness_weight
        self.agent_safety_weight = agent_safety_weight
        self.goal_progress_weight = goal_progress_weight
        self.use_dense_rewards = use_dense_rewards
        
        self.prev_ego_pos = None
        self.prev_trajectory_point_idx = 0
        
    def trajectory_to_actions(self, trajectory, plan_anchors, plan_anchors_yaw, x_anchor=9, y_anchor=9):
        """
        Convert trajectory waypoints to action anchor indices.
        """
        actions = []
        T = trajectory.shape[0]
        
        incremental_displacements = []
        for t in range(min(T, self.num_probe_steps)):
            if t == 0:
                disp_x = trajectory[t, 0].item()
                disp_y = trajectory[t, 1].item()
                disp_yaw = trajectory[t, 2].item()
            else:
                prev_x = trajectory[t-1, 0].item()
                prev_y = trajectory[t-1, 1].item()
                prev_yaw = trajectory[t-1, 2].item()
                curr_x = trajectory[t, 0].item()
                curr_y = trajectory[t, 1].item()
                curr_yaw = trajectory[t, 2].item()
                cos_yaw = math.cos(-prev_yaw)
                sin_yaw = math.sin(-prev_yaw)
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                disp_x = dx * cos_yaw - dy * sin_yaw
                disp_y = dx * sin_yaw + dy * cos_yaw
                disp_yaw = self._angle_diff(curr_yaw, prev_yaw)
            
            incremental_displacements.append((disp_x, disp_y, disp_yaw))
        
        for t, (target_x, target_y, target_yaw) in enumerate(incremental_displacements):
            # Find best matching anchor for this incremental displacement
            best_idx = 0
            best_cost = float('inf')
            
            total_anchors = x_anchor * y_anchor
            
            for idx in range(total_anchors):
                anchor = plan_anchors[idx]
                
                # Extract anchor endpoint
                if anchor.ndim == 2 and anchor.shape[-1] >= 2:
                    ax = anchor[-1, 0].item()
                    ay = anchor[-1, 1].item()
                elif anchor.ndim == 1:
                    if anchor.numel() == 2:
                        ax = anchor[0].item()
                        ay = anchor[1].item()
                    elif anchor.numel() % 2 == 0:
                        a2 = anchor.view(-1, 2)
                        ax = a2[-1, 0].item()
                        ay = a2[-1, 1].item()
                    else:
                        ax = anchor[0].item()
                        ay = anchor[1].item() if anchor.numel() > 1 else 0.0
                else:
                    ax, ay = 0.0, 0.0
                    
                ayaw = plan_anchors_yaw[idx].item()
                
                # Compute cost (position + yaw)
                pos_err = math.sqrt((target_x - ax) ** 2 + (target_y - ay) ** 2)
                yaw_err = self._angle_diff(target_yaw, ayaw)
                cost = pos_err + 0.5 * abs(yaw_err)
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            
            # Convert to action indices
            ax_idx = best_idx // y_anchor
            ay_idx = best_idx % y_anchor
            actions.append((ax_idx, ay_idx))
            
        return actions
    
    def _angle_diff(self, a, b):
        """Compute angular difference normalized to [-pi, pi]"""
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d
    
    def compute_progress_reward(self, env, prev_ego_pos, ego_yaw, expert_trajectory, step_idx):
        """
        Compute forward progress reward based on movement along expert trajectory.    
        Rewards moving forward along the intended path, penalizes moving backward.
        """
        if prev_ego_pos is None:
            return 0.0, {}
        
        # Get current ego position in world frame
        curr_ego_pos = env.base_env.start_ego[:3, 3][[0, 2]].copy()
        
        # Compute displacement
        displacement = curr_ego_pos - prev_ego_pos
        displacement_mag = np.linalg.norm(displacement)
        
        if displacement_mag < 1e-6:
            return 0.0, {'displacement': 0.0}
        
        # Get expert direction at current step
        # expert_trajectory is in ego frame, convert to world frame direction
        if expert_trajectory is not None and step_idx < len(expert_trajectory) - 1:
            # Direction from current to next expert point
            expert_dir_ego = expert_trajectory[step_idx + 1] - expert_trajectory[step_idx]
            # Rotate to world frame using ego yaw
            cos_yaw = math.cos(ego_yaw)
            sin_yaw = math.sin(ego_yaw)
            expert_dir_world = np.array([
                expert_dir_ego[0] * cos_yaw - expert_dir_ego[1] * sin_yaw,
                expert_dir_ego[0] * sin_yaw + expert_dir_ego[1] * cos_yaw
            ])
            expert_dir_norm = np.linalg.norm(expert_dir_world)
            if expert_dir_norm > 1e-6:
                expert_dir_world = expert_dir_world / expert_dir_norm
            else:
                expert_dir_world = np.array([1.0, 0.0])
        else:
            expert_dir_world = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
        
        # Project displacement onto expert direction
        forward_progress = np.dot(displacement, expert_dir_world)
        
        progress_reward = self.progress_weight * forward_progress
        
        # Cap negative progress to avoid overwhelming penalties
        progress_reward = max(progress_reward, -0.1)
        
        progress_info = {
            'displacement': displacement_mag,
            'forward_progress': forward_progress,
            'expert_dir': expert_dir_world.tolist(),
        }
        
        return progress_reward, progress_info
    
    def compute_trajectory_smoothness_reward(self, trajectory, step_idx):
        """
        Compute smoothness reward based on trajectory curvature.
        Penalizes sharp turns and rewards smooth, realistic trajectories.
        """
        if step_idx < 1 or step_idx >= len(trajectory):
            return 0.0, {}
        
        # Compute yaw change
        prev_yaw = trajectory[step_idx - 1, 2].item()
        curr_yaw = trajectory[step_idx, 2].item()
        yaw_change = abs(self._angle_diff(curr_yaw, prev_yaw))
        
        # Compute position change direction change
        if step_idx >= 2:
            prev_prev_pos = trajectory[step_idx - 2, :2]
            prev_pos = trajectory[step_idx - 1, :2]
            curr_pos = trajectory[step_idx, :2]
            
            v1 = prev_pos - prev_prev_pos
            v2 = curr_pos - prev_pos
            
            # Handle CUDA tensors - need to move to CPU before numpy conversion
            if hasattr(v1, 'cpu'):
                v1_np = v1.cpu().numpy()
                v2_np = v2.cpu().numpy()
            else:
                v1_np = v1.numpy() if hasattr(v1, 'numpy') else v1
                v2_np = v2.numpy() if hasattr(v2, 'numpy') else v2
            
            v1_norm = np.linalg.norm(v1_np)
            v2_norm = np.linalg.norm(v2_np)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1_unit = v1_np / v1_norm
                v2_unit = v2_np / v2_norm
                # Angle between consecutive displacement vectors
                cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                direction_change = math.acos(cos_angle)
            else:
                direction_change = 0.0
        else:
            direction_change = 0.0
        
        yaw_threshold = 0.3  # radians
        direction_threshold = 0.4  # radians
        
        yaw_penalty = max(0, yaw_change - yaw_threshold)
        direction_penalty = max(0, direction_change - direction_threshold)
        
        smoothness_reward = -self.smoothness_weight * (yaw_penalty + direction_penalty)
        
        smoothness_info = {
            'yaw_change': yaw_change,
            'direction_change': direction_change,
            'yaw_penalty': yaw_penalty,
            'direction_penalty': direction_penalty,
        }
        
        return smoothness_reward, smoothness_info
    
    def compute_agent_proximity_reward(self, env, agents_data, ego_pos_world):
        """
        Compute reward based on proximity to nearby agents.
        Rewards maintaining safe distance from other agents (defensive driving).
        """
        if agents_data is None or len(agents_data) == 0:
            return 0.0, {'num_agents': 0}
        
        try:
            agents_np = agents_data.cpu().numpy() if hasattr(agents_data, 'cpu') else agents_data
            
            if agents_np.ndim == 1:
                agents_np = agents_np.reshape(1, -1)
            
            num_agents = agents_np.shape[0]
            min_distance = float('inf')
        
            for i in range(num_agents):
        
                agent_feat = agents_np[i]
                if len(agent_feat) >= 2:
                    agent_pos = agent_feat[:2]
                    dist = np.linalg.norm(ego_pos_world[:2] - agent_pos)
                    min_distance = min(min_distance, dist)
            
            # Safe distance threshold (in meters)
            safe_distance = 5.0
            critical_distance = 2.0
            
            if min_distance < critical_distance:
                # Too close - strong penalty
                proximity_reward = -self.agent_safety_weight * 2.0 * (critical_distance - min_distance) / critical_distance
            elif min_distance < safe_distance:
                # Somewhat close - mild penalty
                proximity_reward = -self.agent_safety_weight * 0.5 * (safe_distance - min_distance) / safe_distance
            else:
                # Safe distance - small bonus for defensive driving
                proximity_reward = self.agent_safety_weight * 0.1
            
            proximity_info = {
                'num_agents': num_agents,
                'min_distance': min_distance,
                'safe_distance': safe_distance,
            }
            
            return proximity_reward, proximity_info
            
        except Exception:
            return 0.0, {'num_agents': 0, 'error': True}
    
    def compute_alignment_reward(self, ego_yaw, expert_yaw):
        """
        Compute reward for heading alignment with expert trajectory.
        """
        yaw_error = abs(self._angle_diff(ego_yaw, expert_yaw))
        
        # Graduated rewards based on alignment
        if yaw_error < 0.05:
            alignment_reward = self.alignment_weight * 1.0
        elif yaw_error < 0.15:
            alignment_reward = self.alignment_weight * 0.7
        elif yaw_error < 0.3:
            alignment_reward = self.alignment_weight * 0.4
        elif yaw_error < 0.5:
            alignment_reward = self.alignment_weight * 0.1
        else:
            # Misaligned - penalty
            alignment_reward = -self.alignment_weight * 0.2 * (yaw_error - 0.5)
        
        alignment_info = {
            'yaw_error': yaw_error,
            'yaw_error_deg': yaw_error * 180.0 / math.pi,
        }
        
        return alignment_reward, alignment_info
    
    def compute_proximity_bonus(self, distance_to_expert):
        """
        Compute bonus for staying close to expert trajectory.
        Graduated rewards based on distance to expert.
        """
        if distance_to_expert < 0.3:
            # Very close - high bonus
            proximity_bonus = self.proximity_weight * 1.0
        elif distance_to_expert < 0.5:
            proximity_bonus = self.proximity_weight * 0.8
        elif distance_to_expert < 1.0:
            proximity_bonus = self.proximity_weight * 0.5
        elif distance_to_expert < 2.0:
            proximity_bonus = self.proximity_weight * 0.2
        elif distance_to_expert < 3.0:
            proximity_bonus = self.proximity_weight * 0.05
        else:
            # Far from expert - no bonus
            proximity_bonus = 0.0
        
        proximity_info = {
            'distance': distance_to_expert,
            'bonus': proximity_bonus,
        }
        
        return proximity_bonus, proximity_info
    
    def compute_goal_progress_reward(self, env, prev_ego_pos, goal_pos=None):
        """
        Compute reward for progress toward goal position.
        """
        curr_ego_pos = env.base_env.start_ego[:3, 3][[0, 2]].copy()
        
        if goal_pos is None:
            # Use end of expert trajectory as goal
            if hasattr(env.base_env, 'expert_pair') and len(env.base_env.expert_pair) > 0:
                goal_pos = np.array(env.base_env.expert_pair[-1])
            else:
                return 0.0, {}
        
        if prev_ego_pos is None:
            return 0.0, {'goal_pos': goal_pos.tolist()}
        
        # Compute distance to goal
        prev_dist = np.linalg.norm(prev_ego_pos - goal_pos)
        curr_dist = np.linalg.norm(curr_ego_pos - goal_pos)
        
        progress = prev_dist - curr_dist
        goal_reward = self.goal_progress_weight * progress
        
        # Cap negative progress
        goal_reward = max(goal_reward, -0.05)
        
        goal_info = {
            'prev_dist': prev_dist,
            'curr_dist': curr_dist,
            'progress': progress,
        }
        
        return goal_reward, goal_info
    
    def probe_environment(self, env, trajectory, expert_trajectory=None):
        """
        Probe the environment using the predicted trajectory.
        """
       
        state_backup = env.save_state()
        
        # Save initial world position for coordinate frame conversion
        initial_ego_world_pos = env.base_env.start_ego[:3, 3][[0, 2]].copy()
        
        # Get initial ego yaw for coordinate transformations
        rot = env.base_env.start_ego[:3, :3]
        initial_ego_yaw = math.atan2(rot[1, 0], rot[0, 0])
        
        plan_anchors = env.base_env.plan_anchors
        plan_anchors_yaw = env.base_env.plan_anchors_yaw
        
        actions = self.trajectory_to_actions(
            trajectory, plan_anchors, plan_anchors_yaw,
            x_anchor=int(env.base_env.x_anchor),
            y_anchor=int(env.base_env.y_anchor)
        )
        
        # Initialize probe metrics
        probe_rewards = []
        probe_metrics = {
            'collision_occurred': False,
            'offroad_occurred': False,
            'total_deviation': 0.0,
            'min_distance_to_expert': float('inf'),
            'avg_distance_to_expert': 0.0,
            'comfort_violations': 0,
            'successful_steps': 0,
            'total_survival_bonus': 0.0,
            'total_progress_reward': 0.0,
            'total_proximity_bonus': 0.0,
            'total_alignment_bonus': 0.0,
            'total_smoothness_reward': 0.0,
            'total_agent_safety_reward': 0.0,
            'total_goal_progress_reward': 0.0,
            'endpoint_position': None,
            'endpoint_yaw': None,
            'endpoint_distance_to_expert': None,
            'endpoint_reward': 0.0,
            'last_action': None,
        }
        
        cumulative_reward = 0.0
        discount = 1.0
        
        # Track previous state for progress computation
        prev_ego_pos = initial_ego_world_pos.copy()
        
        # Execute probe steps to reach the endpoint
        for step_idx, (ax, ay) in enumerate(actions):
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step([ax, ay])

            distance = info.get('distance', 0.0)
            collision = info.get('collision', None)
            
            curr_ego_pos = env.base_env.start_ego[:3, 3][[0, 2]].copy()
            rot = env.base_env.start_ego[:3, :3]
            curr_ego_yaw = math.atan2(rot[1, 0], rot[0, 0])
            
            step_reward = 0.0
            
            if self.use_dense_rewards:
               
                if not (terminated or truncated):
                    survival_reward = self.survival_bonus
                    step_reward += survival_reward
                    probe_metrics['total_survival_bonus'] += survival_reward
                
                progress_reward, progress_info = self.compute_progress_reward(
                    env, prev_ego_pos, curr_ego_yaw, expert_trajectory, step_idx
                )
                step_reward += progress_reward
                probe_metrics['total_progress_reward'] += progress_reward

                if expert_trajectory is not None and step_idx < expert_trajectory.shape[0]:
                    expert_relative_pos = expert_trajectory[step_idx]
                    expert_world_pos = initial_ego_world_pos + np.array([expert_relative_pos[0], expert_relative_pos[1]])
                    deviation = np.linalg.norm(curr_ego_pos - expert_world_pos)
                    
                    proximity_bonus, proximity_info = self.compute_proximity_bonus(deviation)
                    step_reward += proximity_bonus
                    probe_metrics['total_proximity_bonus'] += proximity_bonus
                    
                    probe_metrics['total_deviation'] += deviation
                    probe_metrics['min_distance_to_expert'] = min(
                        probe_metrics['min_distance_to_expert'], deviation
                    )
                
                if expert_trajectory is not None and step_idx < len(expert_trajectory) - 1:
                    if step_idx + 1 < len(expert_trajectory):
                        expert_dir = expert_trajectory[step_idx + 1] - expert_trajectory[step_idx]
                        expert_yaw = math.atan2(expert_dir[1], expert_dir[0]) + initial_ego_yaw
                    else:
                        expert_yaw = initial_ego_yaw
                    
                    alignment_reward, alignment_info = self.compute_alignment_reward(curr_ego_yaw, expert_yaw)
                    step_reward += alignment_reward
                    probe_metrics['total_alignment_bonus'] += alignment_reward
                
                smoothness_reward, smoothness_info = self.compute_trajectory_smoothness_reward(trajectory, step_idx)
                step_reward += smoothness_reward
                probe_metrics['total_smoothness_reward'] += smoothness_reward
                
                agents_data = info.get('agents', None)
                if agents_data is not None:
                    agent_reward, agent_info = self.compute_agent_proximity_reward(env, agents_data, curr_ego_pos)
                    step_reward += agent_reward
                    probe_metrics['total_agent_safety_reward'] += agent_reward
                
                goal_reward, goal_info = self.compute_goal_progress_reward(env, prev_ego_pos)
                step_reward += goal_reward
                probe_metrics['total_goal_progress_reward'] += goal_reward
            
            # Collision penalties
            if collision is not None:
                probe_metrics['collision_occurred'] = True
                if collision.get('type') == 'dynamic':
                    step_reward -= 1.0
                elif collision.get('type') == 'off_road':
                    probe_metrics['offroad_occurred'] = True
                    step_reward -= 0.5
                else:
                    step_reward -= 0.75
            
            # Comfort penalty
            comfort_penalty = abs(info.get('yaw_v', 0.0))
            if comfort_penalty > 0.5:
                probe_metrics['comfort_violations'] += 1
                step_reward -= 0.05 * comfort_penalty
            
            # Track successful steps
            if not (terminated or truncated):
                probe_metrics['successful_steps'] += 1
            
            # Accumulate discounted reward
            cumulative_reward += discount * step_reward
            probe_rewards.append(step_reward)
            discount *= self.discount_factor
            
            # Update previous position
            prev_ego_pos = curr_ego_pos.copy()
            
            if step_idx == len(actions) - 1:

                probe_metrics['endpoint_position'] = curr_ego_pos.copy()
                probe_metrics['endpoint_yaw'] = curr_ego_yaw
                probe_metrics['last_action'] = (ax, ay)

                endpoint_reward = step_reward
                
                if expert_trajectory is not None and len(expert_trajectory) > 0:
                    final_expert_idx = min(len(expert_trajectory) - 1, step_idx)
                    expert_relative_pos = expert_trajectory[final_expert_idx]
                    expert_world_pos = initial_ego_world_pos + np.array([expert_relative_pos[0], expert_relative_pos[1]])
                    endpoint_deviation = np.linalg.norm(curr_ego_pos - expert_world_pos)
                    probe_metrics['endpoint_distance_to_expert'] = endpoint_deviation
                    
                    if endpoint_deviation < 0.5:
                        endpoint_reward += 1.0
                    elif endpoint_deviation < 1.0:
                        endpoint_reward += 0.5
                    elif endpoint_deviation < 2.0:
                        endpoint_reward += 0.2
                
                probe_metrics['endpoint_reward'] = endpoint_reward
            
            if terminated or truncated:
                break
        
        if probe_metrics['successful_steps'] == len(actions):
            cumulative_reward += self.success_bonus
        
        if probe_metrics['successful_steps'] > 0:
            probe_metrics['avg_distance_to_expert'] = (
                probe_metrics['total_deviation'] / probe_metrics['successful_steps']
            )
        
        env.restore_state(state_backup)
        
        return {
            'probe_reward': cumulative_reward,
            'probe_rewards_per_step': probe_rewards,
            'probe_metrics': probe_metrics,
            'num_steps_probed': len(actions),
        }
    
    def compute_future_reward(
        self,
        env,
        trajectory,
        expert_trajectory=None,
        current_action=None,
    ):
        """
        Compute the future reward signal for the current action based on
        trajectory probe.
        """

        probe_result = self.probe_environment(env, trajectory, expert_trajectory)
        
        metrics = probe_result['probe_metrics']
        
        endpoint_reward = metrics.get('endpoint_reward', 0.0)
        
        cumulative_reward = probe_result['probe_reward']
       
        endpoint_weight = 0.7
        cumulative_weight = 0.3
        
        future_reward = endpoint_weight * endpoint_reward + cumulative_weight * cumulative_reward
        
        if metrics['collision_occurred']:
            future_reward -= self.collision_penalty_weight * 2.0
        
        if self.use_dense_rewards:
            num_steps = max(metrics['successful_steps'], 1)
            
            avg_proximity = metrics['total_proximity_bonus'] / num_steps
            if avg_proximity > 0.1:
                future_reward += 0.3 * avg_proximity
            
            avg_alignment = metrics['total_alignment_bonus'] / num_steps
            if avg_alignment > 0.05:
                future_reward += 0.2 * avg_alignment
            
            avg_progress = metrics['total_progress_reward'] / num_steps
            if avg_progress > 0:
                future_reward += 0.3 * avg_progress
            
            avg_smoothness = metrics['total_smoothness_reward'] / num_steps
            if avg_smoothness < -0.01:
                future_reward += avg_smoothness * 0.5
        
        endpoint_dist = metrics.get('endpoint_distance_to_expert', None)
        if endpoint_dist is not None:
            if endpoint_dist < 0.5:
                future_reward += 0.5
            elif endpoint_dist < 1.0:
                future_reward += 0.3
            elif endpoint_dist > 5.0:
                future_reward -= 0.5
        
        success_ratio = metrics['successful_steps'] / max(probe_result['num_steps_probed'], 1)
        if success_ratio == 1.0:
            future_reward += self.success_bonus * 0.5
        elif success_ratio > 0.5:
            future_reward += self.success_bonus * 0.2 * success_ratio
        
        future_reward = np.clip(future_reward, -5.0, 5.0)
        
        return future_reward, probe_result

class ProbingModule:
    """
    This approach:
    1. Samples K distinct trajectories from the Flow policy
    2. Probes the environment with all K trajectories
    3. Assigns the highest reward found among K trajectories back to the policy
    4. Encourages maintaining diverse options rather than collapsing to single behavior
    """
    
    def __init__(
        self,
        num_modes=5,  # Number of trajectories to sample (K)
        num_probe_steps=3,  # Steps to probe for each trajectory
        discount_factor=0.9,
        temperature=1.0,  # Temperature for mode selection (higher = more exploration)
        min_reward_gap=0.1,  # Minimum gap between best and second-best to consider clear winner
    ):

        self.num_modes = num_modes
        self.num_probe_steps = num_probe_steps
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.min_reward_gap = min_reward_gap
        
    def sample_trajectory_modes(self, trajectory_head, trajectory_query, agents_query, 
                                 bev_feature, bev_spatial_shape, status_encoding,
                                 target_trajs=None, global_img=None):

        K = self.num_modes
        B = trajectory_query.shape[0]
        device = trajectory_query.device
        
        trajectories = []
        mode_scores = []
        
        with torch.no_grad():
            for k in range(K):
                traj_dict = trajectory_head(
                    trajectory_query, agents_query, bev_feature, bev_spatial_shape,
                    status_encoding, target_trajs, global_img
                )
                
                traj = traj_dict.get("best_trajectory", None)
                if traj is None:
                    traj = traj_dict.get("trajectory", torch.zeros(B, 6, 3, device=device))
                
                trajectories.append(traj)
                
                if "trajectory_loss_dict" in traj_dict:
                    loss = traj_dict["trajectory_loss"]
                    score = -loss.item() if torch.is_tensor(loss) else -loss
                else:
                    score = 0.0
                mode_scores.append(score)
        
        trajectories = torch.stack(trajectories, dim=0)
        
        # Compute mode probabilities from scores
        mode_scores = torch.tensor(mode_scores, device=device)
        mode_probs = torch.softmax(mode_scores / self.temperature, dim=0)
        
        return trajectories, mode_probs
    
    def probe_all_modes(self, env, trajectories, expert_trajectory, trajectory_probe):

        K = trajectories.shape[0]
        probe_results = []
        rewards = []
        
        for k in range(K):
            traj = trajectories[k]
            reward, result = trajectory_probe.compute_future_reward(
                env=env,
                trajectory=traj,
                expert_trajectory=expert_trajectory,
                current_action=None
            )
            probe_results.append(result)
            rewards.append(reward)
        
        rewards = np.array(rewards)
        
        # Compute statistics
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        metrics = {
            'num_modes': K,
            'best_idx': int(best_idx),
            'worst_idx': int(worst_idx),
            'best_reward': float(rewards[best_idx]),
            'worst_reward': float(rewards[worst_idx]),
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'reward_gap': float(rewards[best_idx] - rewards[worst_idx]),
            'all_rewards': rewards.tolist(),
        }
        
        return probe_results, rewards, metrics
    
    def select_best_trajectory(self, trajectories, rewards, mode_probs=None):
        
        K = len(rewards)
        
        best_idx = np.argmax(rewards)
        best_trajectory = trajectories[best_idx]
        best_reward = rewards[best_idx]
     
        sorted_rewards = np.sort(rewards)[::-1]
        reward_gap = sorted_rewards[0] - sorted_rewards[1] if K > 1 else sorted_rewards[0]
        
        clear_winner = reward_gap > self.min_reward_gap
        
        # Compute weighted average of top trajectories (for smoother gradient)
        if mode_probs is not None and not clear_winner:
            # Softmax weighting based on rewards
            reward_weights = torch.softmax(torch.tensor(rewards) / self.temperature, dim=0)
            combined_weights = 0.5 * reward_weights + 0.5 * torch.tensor(mode_probs)
            combined_weights = combined_weights / combined_weights.sum()
            
            # Weighted combination of trajectories
            weighted_trajectory = torch.zeros_like(trajectories[0])
            for k in range(K):
                weighted_trajectory += combined_weights[k].item() * trajectories[k]
            
            selection_method = "weighted_average"
        else:
            weighted_trajectory = best_trajectory
            selection_method = "best_only"
        
        selection_info = {
            'best_idx': int(best_idx),
            'best_reward': float(best_reward),
            'reward_gap': float(reward_gap),
            'clear_winner': clear_winner,
            'selection_method': selection_method,
        }
        
        return best_trajectory, best_reward, best_idx, selection_info, weighted_trajectory
    
    def compute_traj_probe_reward(
        self,
        env,
        trajectory_head,
        trajectory_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        expert_trajectory,
        trajectory_probe,
        target_trajs=None,
        global_img=None,
    ):

        trajectories, mode_probs = self.sample_trajectory_modes(
            trajectory_head, trajectory_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding,
            target_trajs, global_img
        )
        
        K, B, T, _ = trajectories.shape
        
        results = []
        for b in range(B):
            traj_modes = trajectories[:, b, :, :]
            probe_results, rewards, metrics = self.probe_all_modes(
                env[b] if isinstance(env, list) else env,
                traj_modes,
                expert_trajectory[b] if expert_trajectory is not None else None,
                trajectory_probe
            )
            
            best_traj, best_reward, best_idx, selection_info, weighted_traj = self.select_best_trajectory(
                traj_modes, rewards, mode_probs
            )
            
            results.append({
                'best_trajectory': best_traj,
                'weighted_trajectory': weighted_traj,
                'best_reward': best_reward,
                'best_idx': best_idx,
                'all_rewards': rewards,
                'probe_metrics': metrics,
                'selection_info': selection_info,
                'mode_probs': mode_probs.cpu().numpy() if torch.is_tensor(mode_probs) else mode_probs,
            })
        
        # Aggregate results across batch
        best_rewards = [r['best_reward'] for r in results]
        best_trajectories = torch.stack([r['best_trajectory'] for r in results], dim=0)  # [B, T, 3]
        weighted_trajectories = torch.stack([r['weighted_trajectory'] for r in results], dim=0)  # [B, T, 3]
        
        result = {
            'best_rewards': np.array(best_rewards),  # [B]
            'best_trajectories': best_trajectories,  # [B, T, 3]
            'weighted_trajectories': weighted_trajectories,  # [B, T, 3]
            'batch_results': results,
            'mean_best_reward': float(np.mean(best_rewards)),
            'std_best_reward': float(np.std(best_rewards)),
        }
        
        return result