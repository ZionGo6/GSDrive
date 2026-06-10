import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_infra.blocks import *
from model_infra.rewarding import *
from model_infra.backbone.backbone_config import TransfuserConfig
from model_infra.backbone.modules.multimodal_loss import LossComputer
from model_infra.backbone.modules.conditional_unet1d import SinusoidalPosEmb
from model_infra.backbone.modules.blocks import linear_relu_ln, gen_sineembed_for_position



class TrajectoryHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str, config: TransfuserConfig, training: bool):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.ego_fut_ts = num_poses
        self.ego_fut_mode = 18
        self.training = training

        self.num_groups = getattr(config, 'num_groups', 1)
        
        # Load plan anchors
        if plan_anchor_path and os.path.exists(plan_anchor_path):
            plan_anchor = np.load(plan_anchor_path)
            relative_plan_anchor = np.diff(plan_anchor.reshape(-1, 6, 2), axis=1, prepend=0)
      
        else:
            # Create default anchors
            plan_anchor = np.zeros((18, 6, 2), dtype=np.float32)
            for i in range(20):
                for j in range(8):
                    plan_anchor[i, j, 0] = j * 0.5  # x
                    plan_anchor[i, j, 1] = (i - 10) * 0.2  # y
        
        self.plan_anchor = nn.Parameter(
            torch.tensor(relative_plan_anchor, dtype=torch.float32), requires_grad=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 384),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # Diffusion decoder
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 1)
        self.loss_computer = LossComputer(config)
        
        # EMA parameters
        self.use_ema = getattr(config, 'use_ema', True)
        self.ema_decay = getattr(config, 'ema_decay', 0.99)
        self.ema_update_freq = getattr(config, 'ema_update_freq', 6)
        
        if self.use_ema:
            # Create EMA model parameters as a copy of the main model
            self.ema_params = {}
        
        # Initialize step counter for EMA updates
        self.step_counter = 0
        
        self.cfg_dropout_prob = getattr(config, 'cfg_dropout_prob', 0.15)  
        
        self.cfg_guidance_scale = getattr(config, 'cfg_guidance_scale', 2.0)  
        
        # Whether to use CFG during inference (can be disabled for faster inference)
        self.use_cfg_inference = getattr(config, 'use_cfg_inference', True)

        self.x_min = -4.16
        self.x_max = 3.51
        self.y_min = -0.65
        self.y_max = 10.09

    def norm_odo(self, odo_info_fut):

        odo_info_fut_x = (odo_info_fut[..., 0:1]- self.x_min) / (self.x_max - self.x_min)
        odo_info_fut_y = (odo_info_fut[..., 1:2]- self.y_min) / (self.y_max - self.y_min)
        odo_info_fut_head = odo_info_fut[..., 2:3]

        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def denorm_odo(self, odo_info_fut):

        odo_info_fut_x = odo_info_fut[..., 0:1] * (self.x_max - self.x_min) + self.x_min
        odo_info_fut_y = odo_info_fut[..., 1:2] * (self.y_max - self.y_min) + self.y_min
        odo_info_fut_head = odo_info_fut[..., 2:3]

        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def bezier_xyyaw(self, xy6: torch.Tensor) -> torch.Tensor:
        """Convert xy8 to xyyaw using bezier interpolation"""
        assert xy6.shape[-2:] == (6, 2), "Input must be (B, G, 6, 2)"

        B, G, _, _ = xy6.shape
        device, dtype = xy6.device, xy6.dtype

        origin = torch.zeros_like(xy6[..., :1, :])
        ctrl = torch.cat([origin, xy6], dim=-2)
        n = ctrl.shape[-2] - 1

        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]

        binom = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)],
            device=device, dtype=dtype
        )

        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n

        t_pow = t.view(-1, 1) ** torch.arange(0, n, device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n-1, -1, -1, device=device, dtype=dtype)
        basis = binom * t_pow * one_pow

        delta_exp = delta.unsqueeze(2)
       
        basis_exp = basis.view(1, 1, 6, 6, 1)

        deriv = n * (delta_exp * basis_exp).sum(dim=3)

        dx, dy = deriv[..., 0], deriv[..., 1]
        eps = 1e-7
        dx_safe = dx + eps
        dy_safe = dy + eps
        yaw = torch.atan2(dy_safe, dx_safe).unsqueeze(-1)

        return torch.cat([xy6, yaw], dim=-1)
    
    def compute_ot_coupling(self, source: torch.Tensor, target: torch.Tensor, 
                           epsilon: float = 0.1, num_iterations: int = 50) -> torch.Tensor:
        """
        Compute Optimal Transport coupling using Sinkhorn algorithm.
        
        This finds the optimal assignment between source (anchor modes) and target (ground truth trajectories)
        to minimize transport cost, resulting in straighter interpolation paths.
        
        Args:
            source: [B, M, ...] source samples (anchor modes)
            target: [B, N, ...] target samples (ground truth trajectories)
            epsilon: regularization parameter for Sinkhorn (smaller = more precise but slower)
            num_iterations: number of Sinkhorn iterations
            
        Returns:
            coupling: [B, M, N] optimal transport coupling matrix (sum to 1 per batch)
        """
        B = source.shape[0]
        M = source.shape[1]
        N = target.shape[1]
        device = source.device
        dtype = source.dtype
        
        # Flatten spatial dimensions for cost computation
        source_flat = source.view(B, M, -1)  # [B, M, D]
        target_flat = target.view(B, N, -1)  # [B, N, D]
        
        # Compute squared Euclidean distance cost matrix
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
        source_sq = (source_flat ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
        target_sq = (target_flat ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # [B, 1, N]
        cross_term = torch.bmm(source_flat, target_flat.transpose(1, 2))  # [B, M, N]
        
        cost_matrix = source_sq + target_sq - 2 * cross_term  # [B, M, N]
        cost_matrix = cost_matrix.clamp(min=0)  # Ensure non-negative
        
        # Sinkhorn algorithm for entropy-regularized OT
        # Initialize dual variables
        mu = torch.ones(B, M, 1, device=device, dtype=dtype) / M  # [B, M, 1]
        nu = torch.ones(B, 1, N, device=device, dtype=dtype) / N  # [B, 1, N]
        
        # Kernel matrix: K = exp(-C / epsilon)
        K = torch.exp(-cost_matrix / epsilon)  # [B, M, N]
        
        # Initialize scaling vectors
        u = torch.ones(B, M, 1, device=device, dtype=dtype)  # [B, M, 1]
        v = torch.ones(B, 1, N, device=device, dtype=dtype)  # [B, 1, N]
        
        # Sinkhorn iterations
        for _ in range(num_iterations):
            # Update u: u = mu / (K @ v)
            Kv = torch.bmm(K, v.transpose(1, 2)).transpose(1, 2)  # [B, M, 1]
            u = mu / (Kv + 1e-8)
            
            # Update v: v = nu / (K^T @ u)
            Ktu = torch.bmm(K.transpose(1, 2), u)  # [B, N, 1]
            v = nu / (Ktu + 1e-8)
        
        # Compute coupling: P = diag(u) @ K @ diag(v)
        coupling = u * K * v.transpose(1, 2)  # [B, M, N]
        
        # Normalize to ensure it sums to 1
        coupling = coupling / (coupling.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        return coupling
    
    def apply_ot_path(self, source: torch.Tensor, target: torch.Tensor, 
                     t: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        """
        Apply Optimal Transport path interpolation.
        
        Instead of simple linear interpolation: x_t = (1-t) * x_0 + t * x_1
        We use OT-coupled interpolation: x_t = sum_j coupling[i,j] * ((1-t) * x_0[i] + t * x_1[j])
        
        This creates straighter paths that lead to better training dynamics.
        
        Args:
            source: [B, M, ...] source samples (anchor modes)
            target: [B, N, ...] target samples
            t: [B, 1, 1, 1] interpolation time in [0, 1]
            coupling: [B, M, N] optimal transport coupling
            
        Returns:
            interpolated: [B, M, ...] interpolated samples following OT paths
        """
        B = source.shape[0]
        M = source.shape[1]
        N = target.shape[1]
        
        # For each source point, compute weighted combination of paths to all targets
        # interpolated[i] = sum_j coupling[i,j] * ((1-t) * source[i] + t * target[j])
        
        # Expand dimensions for broadcasting
        source_exp = source.unsqueeze(2)  # [B, M, 1, ...]
        target_exp = target.unsqueeze(1)  # [B, 1, N, ...]
        
        # Compute all possible linear interpolations
        all_interpolations = (1 - t) * source_exp + t * target_exp  # [B, M, N, ...]
        
        # Apply coupling weights
        coupling_exp = coupling.unsqueeze(-1)  # [B, M, N, 1] (for broadcasting over other dims)
        weighted_interpolations = coupling_exp * all_interpolations  # [B, M, N, ...]
        
        # Sum over target dimension N to get final interpolation for each source
        interpolated = weighted_interpolations.sum(dim=2)  # [B, M, ...]
        
        return interpolated
    
    def compute_minibatch_ot(self, anchors: torch.Tensor, targets: torch.Tensor,
                            t: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """
        Compute minibatch optimal transport path.
        
        This is a simplified version that:
        1. Computes OT coupling between anchor modes and targets for each sample
        2. Applies OT path interpolation
        3. Returns interpolated trajectories with better training dynamics
        
        Args:
            anchors: [B, M, T, D] anchor trajectories (normalized)
            targets: [B, T, D] target trajectories (normalized, single mode)
            t: [B, 1, 1, 1] interpolation time
            epsilon: OT regularization parameter
            
        Returns:
            interpolated: [B, M, T, D] interpolated trajectories
            coupling: [B, M, 1] coupling weights for loss computation
        """
        B, M, T, D = anchors.shape
        device = anchors.device
        dtype = anchors.dtype
        
        # Expand targets to have same shape as anchors for comparison
        targets_exp = targets.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, D]
        
        # Compute cost: L2 distance between each anchor and the target
        # Since we have single target, we compute simple distance
        cost = ((anchors - targets_exp) ** 2).sum(dim=(2, 3))  # [B, M]
        
        # Convert cost to coupling weights using softmax
        # Lower cost = higher weight
        coupling_weights = F.softmax(-cost / epsilon, dim=1)  # [B, M]
        
        # For OT path with single target, we use weighted linear interpolation
        # Each anchor interpolates towards the same target, but weighted by coupling
        interpolated = (1 - t) * anchors + t * targets_exp  # [B, M, T, D]
        
        # We can also add a small noise scaled by the coupling to create diverse paths
        # This helps prevent all modes from collapsing to the same trajectory
        if self.training:
            # Add mode-specific noise scaled by inverse coupling (less weight = more noise)
            noise_scale = 0.01 * (1.0 - coupling_weights.unsqueeze(-1).unsqueeze(-1))  # [B, M, 1, 1]
            noise = torch.randn_like(interpolated) * noise_scale
            interpolated = interpolated + noise
        
        return interpolated, coupling_weights.unsqueeze(-1)  # [B, M, 1]
    
    def compute_velocity_target_ot(self, source: torch.Tensor, target: torch.Tensor,
                                   interpolated: torch.Tensor, coupling: torch.Tensor,
                                   t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity target for OT-based flow matching.
        
        For OT paths, the velocity field is:
        v(x_t, t) = E[x_1 - x_0 | x_t]
        
        This is different from standard flow matching where v = x_1 - x_0.
        The OT conditioning leads to straighter paths.
        
        Args:
            source: [B, M, T, D] source samples (denormalized anchors)
            target: [B, T, D] target samples (denormalized ground truth)
            interpolated: [B, M, T, D] current interpolated state
            coupling: [B, M, 1] coupling weights
            t: [B, 1, 1, 1] current time
            
        Returns:
            velocity_target: [B, M, T, D] velocity field to learn
        """
        B, M, T, D = source.shape
        
        # Expand target to match source shape
        # target: [B, T, D] -> target.unsqueeze(1): [B, 1, T, D] -> expand: [B, M, T, D]
        target_exp = target.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, D]
        
        # For OT path, the optimal velocity is simply the direction to the target
        # weighted by the coupling
        velocity = target_exp - source  # [B, M, T, D]
        
        # Scale by coupling weights (higher coupling = stronger signal)
        # coupling: [B, M, 1] -> expand to [B, M, 1, 1] for broadcasting
        coupling_exp = coupling.unsqueeze(-1)  # [B, M, 1, 1]
        velocity_weighted = coupling_exp * velocity  # [B, M, T, D]
        
        # For conditional flow matching, we also need to account for the current position
        # The velocity should point from current position towards the target
        # v = (target - interpolated) / (1 - t)
        # But to avoid division by zero at t=1, we use:
        # v = target - source (which is constant for straight line OT)
        
        return velocity  # Return unscaled velocity, let the network learn the scaling

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, target_trajs=None, global_img=None, cfg_guidance_scale=2.0) -> Dict[str, torch.Tensor]:
        """
        Torch module forward pass.
        
        Args:
            ego_query: Ego vehicle query features
            agents_query: Agent query features
            bev_feature: Bird's eye view features
            bev_spatial_shape: BEV spatial dimensions
            status_encoding: Status encoding
            target_trajs: Target trajectories (for training)
            global_img: Global image features
            cfg_guidance_scale: Classifier-Free Guidance scale (for inference)
                - None: Use default from config (self.cfg_guidance_scale)
                - 1.0: Disable CFG (only use conditional prediction)
                - > 1.0: Enable CFG with specified scale
        """
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, target_trajs, global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img, cfg_guidance_scale=cfg_guidance_scale)

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img=None, solver_type='rk4', use_ot_guidance=False, target_trajs=None, cfg_guidance_scale=None) -> Dict[str, torch.Tensor]:
        """
        Flow matching inference using numerical integration with OT path compatibility.
        
        The model learns a velocity field v(x, t) that points from any point x towards the target.
        During inference, we integrate: dx/dt = v(x, t) from t=0 to t=1, starting from the anchor.
        
        """
        bs = ego_query.shape[0]
        device = ego_query.device
        
        # Get guidance scale from parameter or config
        if cfg_guidance_scale is None:
            cfg_guidance_scale = getattr(self, 'cfg_guidance_scale', 2.0)
        
        # Check if CFG should be used during inference
        use_cfg = getattr(self, 'use_cfg_inference', True) and cfg_guidance_scale > 1.0
        
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        plan_anchor = self.bezier_xyyaw(plan_anchor)
        plan_anchor = plan_anchor.to(device)
        
        current_state = plan_anchor.clone()  # [bs, num_modes, ts, 3]
        
        num_integration_steps = 20
        dt = 1.0 / num_integration_steps
        
        # Helper function to compute velocity field at any state and time
        def compute_velocity(state, t_val, use_conditions=True):
            """
            Compute velocity field v(x, t) at given state and time.
            
            Args:
                state: Current state [bs, num_modes, ts, 3]
                t_val: Current time value (float)
                use_conditions: If True, use conditions; if False, use null conditions (for CFG)
                
            Returns:
                vel_field: Velocity field [bs, num_modes, ts, 3]
                poses_reg_list: List of trajectory predictions
                poses_cls_list: List of classification scores
            """
           

            t = torch.full((bs,), t_val, device=device, dtype=state.dtype)
            t_scaled = t.unsqueeze(-1) * 1000
            
            # Project current state to the query (denormalized space, matching forward_train)
            traj_pos_embed = gen_sineembed_for_position(state, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, self.ego_fut_mode, -1)
            
            # Create time embedding for flow matching
            time_embed = self.time_mlp(t_scaled)  # [bs, d_model]
            time_embed = time_embed.view(bs, 1, -1)  # [bs, 1, d_model]
            
            # Determine which conditions to use
            if use_conditions:
                # Use actual conditions (conditional prediction)
                agents_q = agents_query
                ego_q = ego_query
                bev_f = bev_feature
            else:
                # Use null conditions (unconditional prediction for CFG)
                # Create zero tensors matching the original shapes
                agents_q = torch.zeros_like(agents_query)
                ego_q = torch.zeros_like(ego_query)
                bev_f = torch.zeros_like(bev_feature)
            
            poses_reg_list, poses_cls_list, vel_field_list, _ = self.diff_decoder(
                traj_feature, state, bev_f, bev_spatial_shape, 
                agents_q, ego_q, time_embed, status_encoding, global_img
            )
            assert not torch.isnan(vel_field_list[0]).any(), f"vel_field NaN at t={t_val}, use_conditions={use_conditions}"
            assert not torch.isinf(vel_field_list[0]).any(), f"vel_field Inf at t={t_val}, use_conditions={use_conditions}"
            
            # Return the velocity field
            return vel_field_list[0], poses_reg_list, poses_cls_list  # [bs, num_modes, ts, 3]
        
        # Helper function for CFG-guided velocity computation
        def compute_velocity_cfg(state, t_val):
            """
            Compute velocity field with Classifier-Free Guidance.
            
            If CFG is enabled:
                vel_cfg = vel_uncond + guidance_scale * (vel_cond - vel_uncond)
            Otherwise:
                vel_cfg = vel_cond (standard conditional prediction)
                
            Returns:
                vel_field: CFG-guided velocity field [bs, num_modes, ts, 3]
                poses_reg_list: List of trajectory predictions (from conditional)
                poses_cls_list: List of classification scores (from conditional)
            """
            # Always compute conditional velocity
            vel_cond, poses_reg_list, poses_cls_list = compute_velocity(state, t_val, use_conditions=True)
            
            if use_cfg:
                # Compute unconditional velocity (null conditions)
                vel_uncond, _, _ = compute_velocity(state, t_val, use_conditions=False)
                
                # Apply CFG formula: vel_cfg = vel_uncond + scale * (vel_cond - vel_uncond)
                # This is equivalent to: vel_cfg = (1 - scale) * vel_uncond + scale * vel_cond
                # The scale > 1 boosts the influence of conditions
                vel_cfg = vel_uncond + cfg_guidance_scale * (vel_cond - vel_uncond)
                
                return vel_cfg, poses_reg_list, poses_cls_list
            else:
                # No CFG, just use conditional prediction
                return vel_cond, poses_reg_list, poses_cls_list
        
        # Choose the solver method
        if solver_type == 'euler':
            # Euler method (first-order)
            # x_{t+dt} = x_t + dt * v(x_t, t)
            for i in range(num_integration_steps):
                t_val = float(i) / num_integration_steps
                vel_field, poses_reg_list, poses_cls_list = compute_velocity_cfg(current_state, t_val)
                current_state = current_state + dt * vel_field
                
        elif solver_type == 'heun':
            # Heun's method (second-order Runge-Kutta, also known as improved Euler)
            # k1 = v(x_t, t)
            # k2 = v(x_t + dt * k1, t + dt)
            # x_{t+dt} = x_t + dt/2 * (k1 + k2)
            for i in range(num_integration_steps):
                t_val = float(i) / num_integration_steps
                
                # Compute k1 at current state
                k1, _, _ = compute_velocity_cfg(current_state, t_val)
                
                # Compute k2 at predicted state
                predicted_state = current_state + dt * k1
                k2, poses_reg_list, poses_cls_list = compute_velocity_cfg(predicted_state, t_val + dt)
                
                # Heun's update
                current_state = current_state + (dt / 2.0) * (k1 + k2)
                
        elif solver_type == 'rk4':
            # Fourth-order Runge-Kutta method
            # k1 = v(x_t, t)
            # k2 = v(x_t + dt/2 * k1, t + dt/2)
            # k3 = v(x_t + dt/2 * k2, t + dt/2)
            # k4 = v(x_t + dt * k3, t + dt)
            # x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            for i in range(num_integration_steps):
                t_val = float(i) / num_integration_steps
                
                # Compute k1 at current state
                k1, _, _ = compute_velocity_cfg(current_state, t_val)
                
                # Compute k2 at midpoint using k1
                k2, _, _ = compute_velocity_cfg(current_state + (dt / 2.0) * k1, t_val + dt / 2.0)
                
                # Compute k3 at midpoint using k2
                k3, _, _ = compute_velocity_cfg(current_state + (dt / 2.0) * k2, t_val + dt / 2.0)
                
                # Compute k4 at end using k3
                k4, poses_reg_list, poses_cls_list = compute_velocity_cfg(current_state + dt * k3, t_val + dt)
                
                # RK4 update
                current_state = current_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
              
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}. Choose from 'euler', 'heun', 'rk4'")
        
        final_poses_cls = poses_cls_list[-1]
        if final_poses_cls.dim() == 3:
            final_poses_cls = final_poses_cls.squeeze(1)  # [bs, num_modes]
        
        all_trajectories = poses_reg_list[-1]  # [bs, num_modes, ts, 3] - already in denormalized space
        all_cls_normalized = F.softmax(final_poses_cls, dim=-1)  # [bs, num_modes] in [0, 1]
        
        if use_ot_guidance and target_trajs is not None:
            try:
                # Normalize for OT computation
                normalized_all_trajs = self.norm_odo(all_trajectories)
                normalized_target = self.norm_odo(target_trajs)
                
                # Compute coupling weights
                _, ot_coupling = self.compute_minibatch_ot(
                    normalized_all_trajs,
                    normalized_target,
                    t=torch.zeros(bs, 1, 1, 1, device=device),  # t=0 for pure cost computation
                    epsilon=0.1
                )
                # ot_coupling: [bs, num_modes, 1] - higher for modes closer to target
                
                # Use OT coupling for mode selection (instead of classification scores)
                mode_idx = ot_coupling.squeeze(-1).argmax(dim=-1)  # [bs]
                
            except Exception as e:
                # Fallback to classification-based selection
                print(f"OT-guided selection failed, using classification scores: {e}")
                mode_idx = final_poses_cls.argmax(dim=-1)  # [bs]
        else:
            # Standard mode selection based on classification scores
            mode_idx = final_poses_cls.argmax(dim=-1)  # [bs]
        
        mode_idx = mode_idx[:, None, None, None].expand(-1, 1, self._num_poses, 3)  # [bs, 1, ts, 3]
        
        best_trajectory = torch.gather(current_state, 1, mode_idx).squeeze(1)  # [bs, ts, 3]
        
        return {
            "best_trajectory": best_trajectory, 
            "all_trajectories": all_trajectories, 
            "all_cls_normalized": all_cls_normalized,
            "cfg_guidance_scale": cfg_guidance_scale,  # Return the guidance scale used
            "cfg_enabled": use_cfg,  # Whether CFG was applied
        }

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, target_trajs=None, global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)

        plan_anchor = self.bezier_xyyaw(plan_anchor)
        target_trajs = self.bezier_xyyaw(target_trajs.unsqueeze(1))
        plan_anchor = plan_anchor.to(device)
        target_trajs = target_trajs.squeeze(1).to(device)
        
        normalized_targets = self.norm_odo(target_trajs)
        normalized_plan_anchor = self.norm_odo(plan_anchor)
        normalized_targets_expanded = normalized_targets.unsqueeze(1).expand(-1, self.ego_fut_mode, -1, -1)
        
        t = torch.rand((bs,), device=device, dtype=normalized_targets.dtype).view(-1, 1, 1, 1)
        
        try:
            interpolated_traj, coupling_weights = self.compute_minibatch_ot(
                normalized_plan_anchor,  # [bs, num_modes, ts, 3]
                normalized_targets,       # [bs, ts, 3]
                t,                        # [bs, 1, 1, 1]
                epsilon=0.1               # OT regularization
            )
            # interpolated_traj: [bs, num_modes, ts, 3]
            # coupling_weights: [bs, num_modes, 1] - higher for anchors closer to target
            
        except Exception as e:
            # Fallback to linear interpolation if OT fails
            print(f"OT path computation failed, falling back to linear interpolation: {e}")
            interpolated_traj = (1 - t) * normalized_plan_anchor + t * normalized_targets_expanded
            coupling_weights = torch.ones(bs, self.ego_fut_mode, 1, device=device, dtype=normalized_targets.dtype) / self.ego_fut_mode
        
        # Add small amount of noise to stabilize training
        # Scale noise inversely with coupling (higher coupling = less noise = more direct path)
        noise_scale = 0.01 * (1.0 - coupling_weights.unsqueeze(-1))  # [bs, num_modes, 1, 1]
        noise = torch.randn_like(interpolated_traj) * (noise_scale + 0.005)  # Add small base noise
        noisy_interpolated = torch.clamp(interpolated_traj + noise, min=-1, max=1)
        
        noisy_interpolated = self.denorm_odo(noisy_interpolated)

        ego_fut_mode = noisy_interpolated.shape[1]
        
        traj_pos_embed = gen_sineembed_for_position(noisy_interpolated, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        
        t_scaled = t.squeeze(-1).squeeze(-1) * 1000  # Scale to [0, 1000] range
        time_embed = self.time_mlp(t_scaled)
        time_embed = time_embed.view(bs, 1, -1)

        cfg_dropout = getattr(self, 'cfg_dropout_prob', 0.15)
        
        # Random mask for dropping conditions per sample in batch
        cfg_drop_mask = torch.rand(bs, device=device) < cfg_dropout  # [bs] boolean
        cfg_drop_mask_expanded = cfg_drop_mask.view(-1, 1, 1)  # [bs, 1, 1] for broadcasting
        
        if cfg_drop_mask.any():
            # Null condition for agents (zero tensor)
            agents_query_cfg = torch.zeros_like(agents_query)
            agents_query = torch.where(cfg_drop_mask_expanded.expand_as(agents_query), 
                                       agents_query_cfg, agents_query)
            
            # Null condition for ego query (zero tensor)
            ego_query_cfg = torch.zeros_like(ego_query)
            ego_query = torch.where(cfg_drop_mask_expanded.expand_as(ego_query), 
                                    ego_query_cfg, ego_query)
            
            # Null condition for BEV feature (zero tensor)
            # bev_feature shape: [bs, C, H, W] or similar
            bev_feature_cfg = torch.zeros_like(bev_feature)
            cfg_drop_mask_bev = cfg_drop_mask.view(-1, 1, 1, 1).expand_as(bev_feature)
            bev_feature = torch.where(cfg_drop_mask_bev, bev_feature_cfg, bev_feature)
        
        poses_reg_list, poses_cls_list, vel_field_list, _ = self.diff_decoder(traj_feature, noisy_interpolated, bev_feature, bev_spatial_shape, 
                                                                         agents_query, ego_query, time_embed, status_encoding, global_img)
      
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        
        expanded_targets = normalized_targets.unsqueeze(1).expand(-1, self.ego_fut_mode, -1, -1)
        
        try:
            velocity_target = self.compute_velocity_target_ot(
                self.denorm_odo(normalized_plan_anchor),  # source (denormalized anchors)
                target_trajs,                               # target (denormalized ground truth)
                noisy_interpolated,                         # current interpolated state
                coupling_weights,                           # OT coupling weights
                t                                           # current time
            )
        except Exception as e:
            # Fallback to standard velocity computation
            print(f"OT velocity computation failed, using standard: {e}")
            velocity_target = self.denorm_odo(expanded_targets) - noisy_interpolated
        
        for idx, (poses_reg, poses_cls, vel_field) in enumerate(zip(poses_reg_list, poses_cls_list, vel_field_list)):
            
            vel_field_loss = F.mse_loss(vel_field, velocity_target, reduction='none')
            vel_field_loss = vel_field_loss.mean(dim=(2, 3))  # [bs, num_modes]
            
            if 'coupling_weights' in dir() and coupling_weights is not None:
                ot_weighted_vel_loss = (vel_field_loss * coupling_weights.squeeze(-1)).mean()
            else:
                ot_weighted_vel_loss = vel_field_loss.mean()
            
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, target_trajs, plan_anchor)
            
            combined_loss = trajectory_loss + ot_weighted_vel_loss
            
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = combined_loss
            trajectory_loss_dict[f"velocity_loss_{idx}"] = ot_weighted_vel_loss
            ret_traj_loss += combined_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, ].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)

        all_trajectories = poses_reg_list[-1]

        final_poses_cls = poses_cls_list[-1]
        if final_poses_cls.dim() == 3:
            final_poses_cls = final_poses_cls.squeeze(1)  # [bs, num_modes]
        all_cls_normalized = F.softmax(final_poses_cls, dim=-1)  # [bs, num_modes] in [0, 1]

        return {
            "best_trajectory": best_reg, 
            "trajectory_loss": ret_traj_loss, 
            "trajectory_loss_dict": trajectory_loss_dict,
            "all_trajectories": all_trajectories,  # [bs, num_modes, ts, 3]
            "all_cls_normalized": all_cls_normalized,  # [bs, num_modes] in [0, 1] range
            "ot_coupling_weights": coupling_weights,  # [bs, num_modes, 1] - OT coupling weights
            "cfg_drop_mask": cfg_drop_mask,  # [bs] - which samples had conditions dropped (CFG training)
        }