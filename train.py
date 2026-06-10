import os
import math
import json
import copy
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymnasium.envs.registration import register

from utils import set_bn_eval
from reconsimulator.envs import nus_config as nus_cfg

from model_infra.backbone.backbone_config import TransfuserConfig
from model_infra.backbone.transfuser_backbone import TransfuserBackbone
from model_infra.backbone.modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, gen_sineembed_for_position_1d, GridSampleCrossBEVAttentionScorer
from model_infra.compute_advantage import compute_advantages_and_metrics
from model_infra.LSS import LSS
from model_infra.traj_head import TrajectoryHead
from model_infra.blocks import *
from model_infra.rewarding import *
from env import *



TRAJ_ANCHOR_PATH = "./assets/nus/kmeans/plan_recon_6.npy"
ACTION_ANCHOR_PATH = "./assets/nus/anchor/traj_anchor_05s_3721.npy"
ACTION_ANCHOR_YAW_PATH = "./assets/nus/anchor/traj_anchor_05s_3721_yaw.npy"

class ModelWrapper(nn.Module):
    def __init__(self, config, obs_shape, action_space, plan_anchor_path=TRAJ_ANCHOR_PATH, training_stage=True):
        super().__init__()
        self.config = config
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.nvec = action_space.nvec
        self.training = training_stage
        self.traj_anchor_path = plan_anchor_path

        self._backbone = TransfuserBackbone(config)

        self._query_splits = [1, config.num_bounding_boxes]

        self._keyval_embedding = nn.Parameter(torch.randn(80, 128, 128)) 
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self.bev_head = LSS(
            grid_size=(128, 128, 16),
            pc_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
            img_h=112,
            img_w=200,
            num_cameras=6,
            feature_dim=512,
            bev_output_channels=64,)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        
        self._status_encoding = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(12*9, config.tf_d_model),
                                nn.ReLU(),
                                nn.BatchNorm1d(config.tf_d_model)
                                )

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 112),
                        )
        self.bev_align = nn.Linear(80, config.tf_d_model)
        self.bev_upscaler = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                            nn.Conv2d(64, 32, kernel_size=3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                            )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
                            )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._trajectory_head = TrajectoryHead(
            num_poses=6,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=self.traj_anchor_path,
            config=config,
            training=self.training,
        )

        self._action_head = TrajectoryGuidedActionHead(
            config, self.nvec,
            num_anchors_x=int(self.nvec[0]),
            num_anchors_y=int(self.nvec[0]),
        )

        self.value_head = PPOValueHead(config)
        
        # Used in residual connection: final_logits = α * traj_logits + (1-α) * direct_logits
        self.direct_policy_head = DirectPolicyHead(
            config=config,
            num_anchors=int(action_space.nvec[0]),
            hidden_dim=512,
        )
        
        # Residual weight: α controls balance between trajectory-derived and direct policy
        self.register_buffer('policy_residual_alpha', torch.tensor(0.8))

        self.tau = nn.Parameter(torch.tensor(1.0))

    def _process_observation(self, obs):
        """Process observation into camera features"""
        B, C, H, W = obs.shape
        assert C == 18, f"Expected 18 channels (6 cameras * 3), got {C}"

        camera_views = obs.view(B, 6, 3, H, W)

        return {
            "camera_views": camera_views,
        }
    
    def _extract_features(self, obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, cfg_guidance_scale):
        """
        Extract common features used by forward(), assess_action_and_value(), and sample_trajectory_modes().
        """
        obs = self._process_observation(obs)
        camera_views = obs["camera_views"]
        batch_size = agents.shape[0]
        
        # Backbone forward pass
        flatten_image_features = self._backbone(camera_views)
    
        # BEV features
        bev_feature = self.bev_head(camera_views, flatten_image_features, camera_intrinsics, camera_extrinsics)
        bev_feature_upscale = self.bev_upscaler(bev_feature)
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        
        # Status encoding
        agents = agents.view(batch_size, -1)
        assert torch.isnan(agents).sum() == 0, "NaNs in agents"

        status_encoding = self._status_encoding(agents)
        status_spatial = F.interpolate(
            status_encoding.view(batch_size, 16, 4, 4), 
            size=(128, 128), 
            mode='bilinear', 
            align_corners=False
        )

        # Key-value preparation
        keyval = torch.concatenate([bev_feature, status_spatial], dim=1)
        keyval += self._keyval_embedding[None, ...]
    
        # Cross BEV feature
        concat_cross_bev = F.interpolate(keyval, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, bev_feature_upscale], dim=1)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]
        )
   
        # Query preparation
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        keyval_seq = keyval.flatten(2).permute(0, 2, 1)
        keyval_align = self.bev_align(keyval_seq)
        
        query_out = self._tf_decoder(query, keyval_align)
        
        # Split queries
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)
        
        # Trajectory generation
        trajectory_dict = self._trajectory_head(
            trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding[:, None], target_trajs, cfg_guidance_scale=cfg_guidance_scale
        )
        trajectory = trajectory_dict["best_trajectory"]
        trajectory_loss = trajectory_dict.get("trajectory_loss", None)
        
        return {
            'flatten_image_features': flatten_image_features,
            'trajectory_query': trajectory_query,
            'agents_query': agents_query,
            'trajectory_dict': trajectory_dict,
            'trajectory': trajectory,
            'trajectory_loss': trajectory_loss,
            'status_encoding': status_encoding,
            'cross_bev_feature': cross_bev_feature,
            'bev_spatial_shape': bev_spatial_shape,
            'batch_size': batch_size,
        }

    def forward(self, obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, cfg_guidance_scale=2.0):
        """
        Forward pass for training/inference.
        """
        # Extract common features
        features = self._extract_features(obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, cfg_guidance_scale)
        
        flatten_image_features = features['flatten_image_features']
        trajectory_query = features['trajectory_query']
        agents_query = features['agents_query']
        trajectory_dict = features['trajectory_dict']
        trajectory = features['trajectory']
        trajectory_loss = features['trajectory_loss']
        batch_size = features['batch_size']
        
        all_trajectories = trajectory_dict["all_trajectories"]
        all_cls_normalized = trajectory_dict["all_cls_normalized"]
        
        x_min, x_max = -4.16, 3.51
        y_min, y_max = -0.65, 10.09
        num_anchors = int(self.nvec[0])
        max_dist_x = x_max - x_min
        max_dist_y = y_max - y_min
        
        traj_endpoints = all_trajectories[:, :, 1, :]
        
        traj_x = traj_endpoints[:, :, 0].clamp(x_min, x_max)
        traj_y = traj_endpoints[:, :, 1].clamp(y_min, y_max)
        
        # Create anchor positions
        anchor_x = torch.linspace(x_min, x_max, num_anchors, device=traj_x.device)
        anchor_y = torch.linspace(y_min, y_max, num_anchors, device=traj_y.device)
        
        # Compute normalized absolute distances
        dist_x = torch.abs(traj_x.unsqueeze(-1) - anchor_x.unsqueeze(0).unsqueeze(0)) / max_dist_x
        dist_y = torch.abs(traj_y.unsqueeze(-1) - anchor_y.unsqueeze(0).unsqueeze(0)) / max_dist_y
        
        # Convert distance to similarity score using exponential decay
        tau = self.tau
        sim_x = torch.exp(-dist_x / tau)
        sim_y = torch.exp(-dist_y / tau)
        
        # # Weight by cls scores and aggregate across modes
        cls_weights = all_cls_normalized.unsqueeze(-1)
        best_mode_idx = cls_weights.argmax(dim=1)
        best_idx_expanded = best_mode_idx.view(batch_size, 1, 1).expand(-1, -1, 9)

        log_prob_x_per_mode = torch.log_softmax(sim_x / 0.5, dim=-1)
        log_cls_weights = torch.log(cls_weights + 1e-8)
        log_prob_x = torch.logsumexp(log_prob_x_per_mode + log_cls_weights, dim=1)
        traj_logits_x = log_prob_x
        log_prob_y_per_mode = torch.log_softmax(sim_y / 0.5, dim=-1)
        
        log_prob_y = torch.logsumexp(log_prob_y_per_mode + log_cls_weights, dim=1)
        traj_logits_y = log_prob_y

        direct_logits_x, direct_logits_y = self.direct_policy_head(
            trajectory_query, agents_query, flatten_image_features
        )
        
        alpha = self.policy_residual_alpha
        logits_x = alpha * traj_logits_x + (1.0 - alpha) * direct_logits_x
        logits_y = alpha * traj_logits_y + (1.0 - alpha) * direct_logits_y

        value = self.value_head(trajectory_query, agents_query, flatten_image_features)

        dist_x = Categorical(logits=logits_x)
        dist_y = Categorical(logits=logits_y)

        ax = dist_x.sample()
        ay = dist_y.sample()
        action = torch.stack([ax, ay], dim=-1)

        logprob = dist_x.log_prob(action[..., 0]) + dist_y.log_prob(action[..., 1])
        entropy = dist_x.entropy() + dist_y.entropy()

        if self.training:
            return trajectory, trajectory_loss, logits_x, logits_y, action, logprob, entropy, value, trajectory_dict
        else:
            return trajectory, logits_x, logits_y, action, logprob, entropy, value, trajectory_dict
        
    def assess_action_and_value(self, obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, action, cfg_guidance_scale=1.0):
        """
        Assess action and value for RL training.
        """

        features = self._extract_features(obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, cfg_guidance_scale)
        
        flatten_image_features = features['flatten_image_features']
        trajectory_query = features['trajectory_query']
        agents_query = features['agents_query']
        trajectory = features['trajectory']
        trajectory_loss = features['trajectory_loss']
        status_encoding = features['status_encoding']
        cross_bev_feature = features['cross_bev_feature']
        bev_spatial_shape = features['bev_spatial_shape']
        batch_size = features['batch_size']
        trajectory_dict = features['trajectory_dict']

        # Get trajectory outputs
        all_trajectories = trajectory_dict["all_trajectories"]
        all_cls_normalized = trajectory_dict["all_cls_normalized"]

        x_min, x_max = -4.16, 3.51
        y_min, y_max = -0.65, 10.09
        num_anchors = int(self.nvec[0])
        max_dist_x = x_max - x_min
        max_dist_y = y_max - y_min

        traj_endpoints = all_trajectories[:, :, 1, :]
        traj_x = traj_endpoints[:, :, 0].clamp(x_min, x_max)
        traj_y = traj_endpoints[:, :, 1].clamp(y_min, y_max)

        # Create anchors
        anchor_x = torch.linspace(x_min, x_max, num_anchors, device=traj_x.device)
        anchor_y = torch.linspace(y_min, y_max, num_anchors, device=traj_y.device)

        # Compute similarity
        tau = self.tau
        dist_x = torch.abs(traj_x.unsqueeze(-1) - anchor_x.unsqueeze(0).unsqueeze(0)) / max_dist_x
        dist_y = torch.abs(traj_y.unsqueeze(-1) - anchor_y.unsqueeze(0).unsqueeze(0)) / max_dist_y
        sim_x = torch.exp(-dist_x / tau)
        sim_y = torch.exp(-dist_y / tau)

        cls_weights = all_cls_normalized.unsqueeze(-1)
        best_mode_idx = cls_weights.argmax(dim=1)
        best_idx_expanded = best_mode_idx.view(batch_size, 1, 1).expand(-1, -1, 9)
        
        log_prob_x_per_mode = torch.log_softmax(sim_x / 0.5, dim=-1)
        log_cls_weights = torch.log(cls_weights + 1e-8)
        log_prob_x = torch.logsumexp(log_prob_x_per_mode + log_cls_weights, dim=1)
       
        traj_logits_x = log_prob_x
        log_prob_y_per_mode = torch.log_softmax(sim_y / 0.5, dim=-1)
        log_prob_y = torch.logsumexp(log_prob_y_per_mode + log_cls_weights, dim=1)
      
        traj_logits_y = log_prob_y
        
        direct_logits_x, direct_logits_y = self.direct_policy_head(
            trajectory_query, agents_query, flatten_image_features
        )
        
        alpha = self.policy_residual_alpha
        logits_x = alpha * traj_logits_x + (1.0 - alpha) * direct_logits_x
        logits_y = alpha * traj_logits_y + (1.0 - alpha) * direct_logits_y
       
        value = self.value_head(trajectory_query, agents_query, flatten_image_features)
      
        dist_x = Categorical(logits=logits_x)
        dist_y = Categorical(logits=logits_y)

        action_indices = action.long()
        logprob = dist_x.log_prob(action_indices[..., 0]) + dist_y.log_prob(action_indices[..., 1])
        entropy = dist_x.entropy() + dist_y.entropy()

        return trajectory, trajectory_loss, logprob, entropy, value, trajectory_dict
    
    def sample_trajectory_modes(self, obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, num_modes=5):
        """
        Sample K distinct trajectory modes from the Flow Matching model.
        """
        
        features = self._extract_features(obs, agents, target_trajs, camera_intrinsics, camera_extrinsics, cfg_guidance_scale=1.0)
        
        trajectory_query = features['trajectory_query']
        agents_query = features['agents_query']
        cross_bev_feature = features['cross_bev_feature']
        bev_spatial_shape = features['bev_spatial_shape']
        status_encoding = features['status_encoding']
        
        trajectories = []
        mode_scores = []
        
        with torch.no_grad():
            for k in range(num_modes):
                trajectory_dict = self._trajectory_head(
                    trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
                    status_encoding[:, None], target_trajs
                )
                
                traj = trajectory_dict["best_trajectory"]
                trajectories.append(traj)
                
                if "trajectory_loss" in trajectory_dict:
                    loss = trajectory_dict["trajectory_loss"]
                    score = -loss.item() if torch.is_tensor(loss) else -loss
                else:
                    score = 0.0
                mode_scores.append(score)
        
        trajectories = torch.stack(trajectories, dim=0)
        
        mode_scores = torch.tensor(mode_scores, device=obs.device)
        mode_scores = mode_scores - mode_scores.min()
        
        return trajectories, mode_scores

def arrange_agents(agents_buf):
    target_rows = 12
    
    if isinstance(agents_buf, torch.Tensor):
        if agents_buf.dim() == 2:  
            agents_buf = agents_buf[:target_rows, :]  # Crop if larger than 12 rows
            
            curr_rows = agents_buf.size(0)
            if curr_rows < target_rows:
                agents_buf = F.pad(agents_buf, (0, 0, 0, target_rows - curr_rows), mode='constant', value=0)
            
            return agents_buf
        
    else:
        processed_agents_buf = []
        for item in agents_buf:
            item = item[:target_rows, :] 
            curr_rows = item.size(0)
            if curr_rows < target_rows:
                item = F.pad(item, (0, 0, 0, target_rows - curr_rows), mode='constant', value=0)
            
            processed_agents_buf.append(item)
        return torch.stack(processed_agents_buf, dim=0)

def make_env(cuda=0, scene=0, debug=False, resize_shape=(112, 200)):
    env = ReconNusPPOEnv(cuda=cuda, scene=scene, debug=debug, resize_shape=resize_shape)
    return env

# ============================================================================
# RL Training Function
# ============================================================================

def train_ppo(
    cuda=0,
    scene=0,
    total_timesteps=100000,
    learning_rate=1e-5,  
    num_steps=16,  
    num_envs=8,
    minibatch_size=4,
    device_ids=None,
    gamma=0.99,
    gae_lambda=0.95,
    update_epochs=2,
    clip_coef=0.15,
    vf_coef=0.1,
    ent_coef=0.05,
    max_grad_norm=0.5,
    resize_shape=(112, 200),
    device="cuda",
    save_path="ppo_reconsimulator_v1.pt",
    init_model_path=None,
    scenes=None,
    # Trajectory Probe Reward parameters
    use_trajectory_probe=True,
    probe_num_steps=3,
    probe_discount_factor=0.9,
    probe_collision_weight=2.0,
    probe_deviation_weight=1.0,
    probe_comfort_weight=0.5,
    probe_success_bonus=1.0,
    probe_reward_weight=0.1,
    clear_cache_every=5,  
    # KL divergence penalty parameters
    use_kl_penalty=True,
    kl_coef=0.5,
    kl_target=0.04,
    # Entropy coefficient schedule parameters
    use_entropy_schedule=True,
    ent_coef_start=0.3,
    ent_coef_end=0.1,
    ent_coef_decay_updates=2000,
    clip_vcoef=0.4,
    # Checkpoint averaging parameters
    use_checkpoint_averaging=True,
    checkpoint_avg_start_ratio=0.9, 
    checkpoint_avg_frequency=5, 
    checkpoint_avg_max_count=10, 
    checkpoint_avg_save_path=None,
    use_traj_probe=True, 
    traj_probe_num_modes=6,  
    traj_probe_num_probe_steps=6,
    traj_probe_discount_factor=0.9,
    traj_probe_temperature=1.0,
    traj_probe_min_reward_gap=0.1,
    traj_probe_reward_weight=0.5,
):
    
    torch.autograd.set_detect_anomaly(True)

    if scenes is None:
        try:
            base = nus_cfg.BASE_DATA_DIR
            scenes = sorted([
                int(d) for d in os.listdir(base)
                if d.isdigit() and os.path.isdir(os.path.join(base, d))
            ])
        except Exception:
            scenes = [scene]

    envs = []
    for i in range(num_envs):
        sc = scenes[i % len(scenes)]
        envs.append(make_env(cuda=cuda, scene=int(sc), debug=False, resize_shape=resize_shape))

    obs_shape = envs[0].observation_space.shape
    action_space = envs[0].action_space
    main_device = torch.device(device if torch.cuda.is_available() else "cpu")

    config = TransfuserConfig()

    base_model = ModelWrapper(
        config, obs_shape, action_space, 
        plan_anchor_path=TRAJ_ANCHOR_PATH,
    ).to(main_device)

    model = base_model

    set_bn_eval(model, exclude_names=["value_head", "_action_head"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_updates = math.ceil(total_timesteps / (num_steps * num_envs))
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=learning_rate * 0.1)

    trajectory_probe = TrajectoryProbeReward(
        num_probe_steps=probe_num_steps,
        discount_factor=probe_discount_factor,
        collision_penalty_weight=probe_collision_weight,
        deviation_penalty_weight=probe_deviation_weight,
        comfort_weight=probe_comfort_weight,
        success_bonus=probe_success_bonus,
    ) if use_trajectory_probe else None
    
    
    traj_probe = ProbingModule(
        num_modes=traj_probe_num_modes,
        num_probe_steps=traj_probe_num_probe_steps,
        discount_factor=traj_probe_discount_factor,
        temperature=traj_probe_temperature,
        min_reward_gap=traj_probe_min_reward_gap,
    ) if use_traj_probe else None
    
    if use_traj_probe:
        print(f"[Probing] Enabled with {traj_probe_num_modes} trajectory modes, {traj_probe_num_probe_steps} probe steps")
        print(f"[Probing] Temperature: {traj_probe_temperature}, Min reward gap: {traj_probe_min_reward_gap}")
        print(f"[Probing] Reward weight: {traj_probe_reward_weight}")

    obs_list = []
    target_trajs_list = []
    traj_anchors_list = []
    agents_list = []
    camera_intrinsics_list = []
    camera_extrinsics_list = []

    for env in envs:
        reset_obs, reset_info = env.reset()
        obs_list.append(reset_obs.astype(np.float32))
        assert np.isnan(reset_info["agents"]).sum() == 0, "NaNs in reset_info, agents" 
        target_trajs_list.append(reset_info["target_trajs"])
        traj_anchors_list.append(reset_info["traj_anchors"])
        agents_list.append(reset_info["agents"])
        camera_intrinsics_list.append(reset_info["intrinsics"])
        camera_extrinsics_list.append(reset_info["extrinsics"])

    agents_list = arrange_agents(agents_list)

    obs = np.stack(obs_list, axis=0)
  
    target_trajs = np.stack(target_trajs_list, axis=0)
    traj_anchors = np.stack(traj_anchors_list, axis=0)
    agents = np.stack(agents_list, axis=0)
    assert np.isnan(agents).sum() == 0, "NaNs in agents"
    camera_intrinsics = torch.stack(camera_intrinsics_list, dim=0)
    camera_extrinsics = torch.stack(camera_extrinsics_list, dim=0)
  
    if init_model_path is not None and os.path.exists(init_model_path):
        state_dict = torch.load(init_model_path, map_location=main_device)
        model.load_state_dict(state_dict, strict=False)

    global_step = 0
    num_updates = math.ceil(total_timesteps / (num_steps * num_envs))
    print(f'num_updates: {num_updates}')

    training_logs = []

    # Checkpoint averaging setup
    checkpoint_states = []  # Store state dicts for averaging
    checkpoint_avg_start_update = int(num_updates * checkpoint_avg_start_ratio)
    checkpoint_avg_save_path = checkpoint_avg_save_path or save_path.replace('.pt', '_averaged.pt')
    
    if use_checkpoint_averaging:
        print(f"[Checkpoint Averaging] Will start saving checkpoints at update {checkpoint_avg_start_update}/{num_updates}")
        print(f"[Checkpoint Averaging] Frequency: every {checkpoint_avg_frequency} updates, max {checkpoint_avg_max_count} checkpoints")
        print(f"[Checkpoint Averaging] Final averaged model will be saved to: {checkpoint_avg_save_path}")

    for update in range(num_updates):
        obs_buf = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        actions_buf = np.zeros((num_steps, num_envs, len(action_space.nvec)), dtype=np.int64)
        agents_buf = np.zeros((num_steps, num_envs, agents.shape[1], agents.shape[2]), dtype=np.float32)
        target_trajs_buf = np.zeros((num_steps, num_envs, target_trajs.shape[1], target_trajs.shape[2]), dtype=np.float32)
        camera_intrinsics_buf = np.zeros((num_steps, num_envs, 6, 3, 3), dtype=np.float32)
        camera_extrinsics_buf = np.zeros((num_steps, num_envs, 6, 4, 4), dtype=np.float32)

        logprobs_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        entropy_buf = np.zeros((num_steps, num_envs), dtype=np.float32)  # Track entropy for exploration monitoring
        metrics_buf = [[None for _ in range(num_envs)] for _ in range(num_steps)]
        plan_x_probs_buf = np.zeros((num_steps, num_envs, int(action_space.nvec[0])), dtype=np.float32)
        plan_y_probs_buf = np.zeros((num_steps, num_envs, int(action_space.nvec[1])), dtype=np.float32)
        
        
        # Probing reward buffers
        traj_probe_best_reward_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        traj_probe_best_idx_buf = np.zeros((num_steps, num_envs), dtype=np.int32)
        traj_probe_num_modes_sampled_buf = np.zeros((num_steps, num_envs), dtype=np.int32)
        traj_probe_reward_gap_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        traj_probe_mean_reward_buf = np.zeros((num_steps, num_envs), dtype=np.float32)
        traj_probe_all_rewards_buf = np.zeros((num_steps, num_envs, traj_probe_num_modes), dtype=np.float32)

        model.eval()

        for step in range(num_steps):
            obs_buf[step] = obs
            agents_buf[step] = agents
            target_trajs_buf[step] = target_trajs
            camera_intrinsics_buf[step] = camera_intrinsics.cpu()
            camera_extrinsics_buf[step] = camera_extrinsics.cpu()

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=main_device)
            agents_batch = torch.tensor(agents, dtype=torch.float32, device=main_device)
            assert torch.isnan(agents_batch).sum() == 0, "Nan in agents_batch"
            target_trajs_batch = torch.tensor(target_trajs, dtype=torch.float32, device=main_device)
            camera_intrinsics_batch = camera_intrinsics.to(dtype=torch.float32, device=main_device)
            camera_extrinsics_batch = camera_extrinsics.to(dtype=torch.float32, device=main_device)
          
            with torch.no_grad():
                # Disable CFG during rollout by setting cfg_guidance_scale=1.0
                trajectory, logits_x, logits_y, action_tensor, logprob_tensor, \
                    entropy_tensor, value_tensor, trajectory_dict = model(obs_tensor, agents_batch, target_trajs_batch, camera_intrinsics_batch, 
                                                                                                         camera_extrinsics_batch, cfg_guidance_scale=1.0)
            
            traj_probe_results = {
                'best_rewards': np.zeros(num_envs),
                'best_trajectories': trajectory.cpu().numpy(),
                'best_indices': np.zeros(num_envs, dtype=np.int32),
                'all_rewards': np.zeros((num_envs, traj_probe_num_modes)),
            }
            
            if use_traj_probe and traj_probe is not None:
    
                # Sample K trajectory modes
                sampled_trajectories, mode_scores = model.sample_trajectory_modes(
                    obs_tensor, agents_batch, target_trajs_batch,
                    camera_intrinsics_batch, camera_extrinsics_batch,
                    num_modes=traj_probe_num_modes
                )
                
                # Probe all K trajectories for each environment
                for env_idx in range(num_envs):
                    traj_modes = sampled_trajectories[:, env_idx, :, :]
                    expert_traj = target_trajs[env_idx]
                    
                    # Probe each trajectory mode
                    mode_rewards = []
                    for k in range(traj_probe_num_modes):
                        traj = traj_modes[k]
                        
                        # Probe this trajectory in the environment
                        if trajectory_probe is not None:
                            probe_r, _ = trajectory_probe.compute_future_reward(
                                env=envs[env_idx],
                                trajectory=traj,
                                expert_trajectory=expert_traj,
                                current_action=None
                            )
                        else:
                            probe_r = mode_scores[k].item() if k < len(mode_scores) else 0.0
                        
                        mode_rewards.append(probe_r)
                    
                    mode_rewards = np.array(mode_rewards)
                    
                    best_idx = np.argmax(mode_rewards)
                    best_reward = mode_rewards[best_idx]
                    mean_reward = np.mean(mode_rewards)
                    reward_gap = best_reward - np.min(mode_rewards)
                    
                    # Store results
                    traj_probe_results['best_rewards'][env_idx] = best_reward
                    traj_probe_results['best_indices'][env_idx] = best_idx
                    traj_probe_results['all_rewards'][env_idx] = mode_rewards
                    traj_probe_results['best_trajectories'][env_idx] = traj_modes[best_idx].cpu().numpy()
                    
                    # Store in buffers
                    traj_probe_best_reward_buf[step, env_idx] = best_reward
                    traj_probe_best_idx_buf[step, env_idx] = best_idx
                    traj_probe_num_modes_sampled_buf[step, env_idx] = traj_probe_num_modes
                    traj_probe_reward_gap_buf[step, env_idx] = reward_gap
                    traj_probe_mean_reward_buf[step, env_idx] = mean_reward
                    traj_probe_all_rewards_buf[step, env_idx] = mode_rewards
                        
            actions = action_tensor.cpu().numpy()
            logprobs = logprob_tensor.cpu().numpy()
            values = value_tensor.cpu().numpy()

            # Sanitize values to prevent NaN propagation
            values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)

            probs_x = torch.softmax(logits_x, dim=-1).cpu().numpy()
            probs_y = torch.softmax(logits_y, dim=-1).cpu().numpy()

            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            values_buf[step] = values.squeeze()
            entropy_buf[step] = entropy_tensor.cpu().numpy()  # Track entropy for exploration monitoring
            plan_x_probs_buf[step] = probs_x
            plan_y_probs_buf[step] = probs_y

            next_obs_list = []
            next_target_trajs_list = []
            next_traj_anchors_list = []
            next_agents_list = []
            next_camera_intrinsics_list = []
            next_camera_extrinsics_list = []

            for env_idx, env in enumerate(envs):
                act = actions[env_idx]
                
                # Compute trajectory probe reward BEFORE stepping
                probe_reward = 0.0
                probe_info = {}
                if trajectory_probe is not None and use_trajectory_probe:
                   
                    # Get the trajectory for this environment
                    current_traj = trajectory[env_idx]  # [T, 3]
                    expert_traj = target_trajs[env_idx]  # [T, 2]
                    
                    # Compute future reward from trajectory probe
                    probe_reward, probe_info = trajectory_probe.compute_future_reward(
                        env=env,
                        trajectory=current_traj,
                        expert_trajectory=expert_traj,
                        current_action=(act[0], act[1])
                    )
                    # Weight the probe reward
                    probe_reward = probe_reward_weight * probe_reward
                    
                
                next_obs, reward, terminated, truncated, info = env.step(act)
              
                probe_weight = 0.2
               
                traj_probe_weight = traj_probe_reward_weight
                traj_probe_best_reward = traj_probe_best_reward_buf[step, env_idx]
                
                combined_reward = (
                    reward +
                    probe_weight * probe_reward +
                    traj_probe_weight * traj_probe_best_reward
                )
                rewards_buf[step, env_idx] = combined_reward
                
                done = terminated or truncated
                dones_buf[step, env_idx] = float(done)
                collision = info.get("collision", None)
                
                is_dynamic = bool(info.get("is_dynamic_collision_box", False))
                static_score = float(info.get("static_collision_score", 0.0))
                distance_dev = float(info.get("distance", 0.0))
                
                metrics_buf[step][env_idx] = {
                    "ego2match_yaw_degrees": float(info.get("ego2match_yaw_degrees", 0.0)),
                    "distance_deviation": distance_dev,
                    "is_dynamic_collision_box": is_dynamic,
                    "static_collision_score": static_score,
                    "collision_position": np.array([0.0, 0.0]),
                    "linear_v": float(env.prev_speed),
                    "yaw_v": float(info.get("yaw_v", 0.0)),
                    "expert_timestamp": int(env.base_env.now_frame),
                    "rl_value_function": float(values[env_idx].item()) if not np.isnan(values[env_idx]) else 0.0,
                    "env_reward": float(reward),
                    "combined_reward": float(combined_reward),
                    "traj_probe_best_reward": float(traj_probe_best_reward),
                    "traj_probe_best_idx": int(traj_probe_best_idx_buf[step, env_idx]),
                    "traj_probe_reward_gap": float(traj_probe_reward_gap_buf[step, env_idx]),
                    "traj_probe_mean_reward": float(traj_probe_mean_reward_buf[step, env_idx]),
                }
                global_step += 1

                if done:
                    cur_sc_idx = (update + env_idx) % len(scenes)
                    env.set_scene(int(scenes[cur_sc_idx]))
                    next_obs, _ = env.reset()

                next_obs_list.append(next_obs.astype(np.float32))
                next_target_trajs_list.append(info["target_trajs"])
                next_traj_anchors_list.append(info["traj_anchors"])
                next_agents_list.append(info["agents"])
                next_camera_intrinsics_list.append(info["intrinsics"])
                next_camera_extrinsics_list.append(info["extrinsics"])

            next_agents_list = arrange_agents(next_agents_list)

            obs = np.stack(next_obs_list, axis=0)
            target_trajs = np.stack(next_target_trajs_list, axis=0)
            traj_anchors = np.stack(next_traj_anchors_list, axis=0) 
            agents = np.stack(next_agents_list, axis=0)
            camera_intrinsics = torch.stack(next_camera_intrinsics_list, dim=0)
            camera_extrinsics = torch.stack(next_camera_extrinsics_list, dim=0)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=main_device)
        agents_batch = torch.tensor(agents, dtype=torch.float32, device=main_device)
        target_trajs_batch = torch.tensor(target_trajs, dtype=torch.float32, device=main_device)
        camera_intrinsics_batch = camera_intrinsics.to(dtype=torch.float32, device=main_device)
        camera_extrinsics_batch = camera_extrinsics.to(dtype=torch.float32, device=main_device)

        with torch.no_grad():
            
            # Disable CFG for last_value computation
            trajectory, logits_x, logits_y, action_tensor, logprob_tensor, entropy_tensor, last_value, trajectory_dict = model(obs_tensor, agents_batch, target_trajs_batch, camera_intrinsics_batch, 
                                                                                                         camera_extrinsics_batch, cfg_guidance_scale=1.0)
        last_value = last_value.squeeze().cpu().numpy()
        
        # Sanitize last_value to prevent NaN propagation
        last_value = np.nan_to_num(last_value, nan=0.0, posinf=10.0, neginf=-10.0)

        advantages = np.zeros_like(rewards_buf, dtype=np.float32)
        lastgaelam = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones_buf[t]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_buf

        obs_batch = torch.tensor(
            obs_buf.reshape(num_steps * num_envs, *obs_shape),
            dtype=torch.float32,
            device=main_device,
        )
        
        actions_batch = torch.tensor(
            actions_buf.reshape(num_steps * num_envs, len(action_space.nvec)),
            dtype=torch.long,
            device=main_device,
        )
        agents_batch = torch.tensor(
            agents_buf.reshape(num_steps * num_envs, 12, 9),
            dtype=torch.float32,
            device=main_device,
        )
       
        target_trajs_batch = torch.tensor(
            target_trajs_buf.reshape(num_steps * num_envs, 6, 2),
            dtype=torch.float32,
            device=main_device,
        )
       
        camera_intrinsics_batch = torch.tensor(
            camera_intrinsics_buf.reshape(num_steps * num_envs, 6, 3, 3),
            dtype=torch.float32,
            device=main_device,
        )
        camera_extrinsics_batch = torch.tensor(
            camera_extrinsics_buf.reshape(num_steps * num_envs, 6, 4, 4),
            dtype=torch.float32,
            device=main_device,
        )

        logprobs_batch = torch.tensor(
            logprobs_buf.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        advantages_batch = torch.tensor(
            advantages.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        returns_batch = torch.tensor(
            returns.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        values_batch = torch.tensor(
            values_buf.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )

        returns_batch = torch.nan_to_num(returns_batch, nan=0.0, posinf=10.0, neginf=-10.0)
        values_batch = torch.nan_to_num(values_batch, nan=0.0, posinf=10.0, neginf=-10.0)

        
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        batch_size = num_steps * num_envs
        idxs = np.arange(batch_size)
     

        d_max_eval = 5.0
        custom_advantages = np.zeros_like(rewards_buf, dtype=np.float32)
        dec_adv_dyn = np.zeros_like(rewards_buf, dtype=np.float32)
        dec_adv_sta = np.zeros_like(rewards_buf, dtype=np.float32)
        dec_adv_dist = np.zeros_like(rewards_buf, dtype=np.float32)
        dec_adv_ang = np.zeros_like(rewards_buf, dtype=np.float32)
        total_timestamps = 0
        total_not_gameover_timestamps = 0
        collision_gameover_count = 0
        static_collision_gameover_count = 0
        deviation_gameover_count = 0
        direction_gameover_count = 0
        success_count = 0
        front_collision_count = 0
        back_collision_count = 0
        longitudinal_jerk_vals = []
        yaw_jerk_vals = []
        clips_for_update = {}

        for env_idx in range(num_envs):
            clips_infos = {}
            timestamp_list = []
            for t in range(num_steps):
                info_dict = metrics_buf[t][env_idx]
                if info_dict is None:
                    continue
                ts = t
                timestamp_list.append(ts)
                clips_infos[ts] = info_dict
            if len(timestamp_list) == 0:
                continue
            clip_data = {
                "clip_id": f"update_{update}_env_{env_idx}",
                "timestamp_list": timestamp_list,
                "clips_infos": clips_infos,
            }
            clips_for_update[clip_data["clip_id"]] = clip_data

            (
                clip_not_gameover_list,
                dynamic_collision_gameover,
                static_collision_gameover,
                distance_deviation_gameover,
                angle_deviation_gameover,
                first_collision_position,
                expert_timestamp,
                front_collision,
                back_collision,
                longitudinal_jerk_mean,
                yaw_jerk_mean,
                combined_advantages_list,
                dynamic_advantages_list,
                static_advantages_list,
                distance_advantages_list,
                angle_advantages_list,
                progress_list,
            ) = compute_advantages_and_metrics(clip_data, d_max_eval)

            total_timestamps += len(timestamp_list)
            total_not_gameover_timestamps += len(clip_not_gameover_list)
            for i, ts in enumerate(timestamp_list):
                if i < len(combined_advantages_list):
                    custom_advantages[ts, env_idx] = float(combined_advantages_list[i])
                if i < len(dynamic_advantages_list):
                    dec_adv_dyn[ts, env_idx] = float(dynamic_advantages_list[i])
                if i < len(static_advantages_list):
                    dec_adv_sta[ts, env_idx] = float(static_advantages_list[i])
                if i < len(distance_advantages_list):
                    dec_adv_dist[ts, env_idx] = float(distance_advantages_list[i])
                if i < len(angle_advantages_list):
                    dec_adv_ang[ts, env_idx] = float(angle_advantages_list[i])

            if dynamic_collision_gameover and not (distance_deviation_gameover or static_collision_gameover or angle_deviation_gameover):
                collision_gameover_count += 1
            elif distance_deviation_gameover and not (dynamic_collision_gameover or static_collision_gameover or angle_deviation_gameover):
                deviation_gameover_count += 1
            elif angle_deviation_gameover and not (dynamic_collision_gameover or static_collision_gameover or distance_deviation_gameover):
                direction_gameover_count += 1
            elif static_collision_gameover and not (dynamic_collision_gameover or distance_deviation_gameover or angle_deviation_gameover):
                static_collision_gameover_count += 1
            elif not (dynamic_collision_gameover or distance_deviation_gameover or static_collision_gameover or angle_deviation_gameover):
                success_count += 1

            if dynamic_collision_gameover and front_collision:
                front_collision_count += 1
            elif dynamic_collision_gameover and back_collision:
                back_collision_count += 1

            longitudinal_jerk_vals.append(longitudinal_jerk_mean)
            yaw_jerk_vals.append(yaw_jerk_mean)

        if total_timestamps > 0:
            success_rate = success_count / num_envs
            collision_rate = collision_gameover_count / num_envs
            front_collision_rate = front_collision_count / num_envs
            back_collision_rate = back_collision_count / num_envs
            static_collision_rate = static_collision_gameover_count / num_envs
            deviation_rate = deviation_gameover_count / num_envs
            direction_rate = direction_gameover_count / num_envs
            gameover_ratio = total_not_gameover_timestamps / total_timestamps
            v_jerk = float(np.array(longitudinal_jerk_vals).mean()) if len(longitudinal_jerk_vals) > 0 else 0.0
            yaw_jerk = float(np.array(yaw_jerk_vals).mean()) if len(yaw_jerk_vals) > 0 else 0.0
          
            print(
                "SR:" + f"{success_rate:.5f}" +
                ",\tdynamic_collision_rate:" + f"{collision_rate:.5f}" +
                ",\tfront_collision_rate:" + f"{front_collision_rate:.5f}" +
                ",\tback_collision_rate:" + f"{back_collision_rate:.5f}" +
                ",\tstatic_collision_rate:" + f"{static_collision_rate:.5f}" +
                ",\tdeviation_rate:" + f"{deviation_rate:.5f}" +
                ",\tdirection_rate:" + f"{direction_rate:.5f}" +
                ",\tSDTR:" + f"{gameover_ratio:.5f}" +
                ",\tv_jerk:" + f"{v_jerk:.5f}" +
                ",\tyaw_jerk:" + f"{yaw_jerk:.5f}"
            )
    
        gae_advantages_batch = torch.tensor(
            advantages.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        custom_advantages_batch = torch.tensor(
            custom_advantages.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        
        values_tensor = torch.tensor(
            values_buf.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        
        returns_batch = gae_advantages_batch + values_tensor
        
        custom_scale = 0.5 
        advantages_batch = gae_advantages_batch + custom_scale * custom_advantages_batch
       
        gae_mean = float(gae_advantages_batch.mean().item())
        gae_std = float(gae_advantages_batch.std().item())
        custom_mean = float(custom_advantages_batch.mean().item())
        custom_std = float(custom_advantages_batch.std().item())
        
        adv_mean = float(advantages_batch.mean().item())
        adv_std = float(advantages_batch.std().item())
        adv_min = float(advantages_batch.min().item())
        adv_max = float(advantages_batch.max().item())
        val_mean = float(values_tensor.mean().item())
        val_min = float(values_tensor.min().item())
        val_max = float(values_tensor.max().item())

        if adv_std < 1e-4:
            print(f"[CRITICAL] Advantage std is {adv_std}. Agent sees no difference in actions. Forcing exploration...")
            
        if adv_std > 1e-6:
            advantages_batch = (advantages_batch - adv_mean) / (adv_std + 1e-8)
        else:
            # If variance is zero, skip standardization to avoid NaN
            print(f"[WARNING] Low advantage variance. Mean: {adv_mean}, Std: {adv_std}")
            # Optionally, add a small amount of artificial noise to kickstart learning
            advantages_batch = advantages_batch - adv_mean
        
        # Also check raw rewards from env
        rewards_tensor = torch.tensor(
            rewards_buf.reshape(num_steps * num_envs),
            dtype=torch.float32,
            device=main_device,
        )
        rew_mean = float(rewards_tensor.mean().item())
        rew_std = float(rewards_tensor.std().item())
        
        if update % 5 == 0:
            print(f"[DEBUG] GAE_adv (from rewards): mean={gae_mean:.5f}, std={gae_std:.5f}")
            print(f"[DEBUG] custom_adv (sparse): mean={custom_mean:.5f}, std={custom_std:.5f}")
            print(f"[DEBUG] combined_adv: mean={adv_mean:.5f}, std={adv_std:.5f}, min={adv_min:.5f}, max={adv_max:.5f}")
            print(f"[DEBUG] values: mean={val_mean:.5f}, min={val_min:.5f}, max={val_max:.5f}")
            print(f"[DEBUG] raw_rewards: mean={rew_mean:.5f}, std={rew_std:.5f}")
        
        if update % 5 == 0:
            ret_mean = float(returns_batch.mean().item())
            ret_std = float(returns_batch.std().item())
            print(f"[DEBUG] returns (from GAE only): mean={ret_mean:.5f}, std={ret_std:.5f}")
        
        if update % 5 == 0:
            ret_mean = float(returns_batch.mean().item())
            ret_std = float(returns_batch.std().item())
            print(f"[DEBUG] returns (before adv standardization): mean={ret_mean:.5f}, std={ret_std:.5f}")
        
        # Standardize advantages ONLY for policy gradient loss stability
        # This does NOT affect returns or value loss
        if adv_std > 1e-6:
            advantages_batch = (advantages_batch - advantages_batch.mean()) / advantages_batch.std()
        else:
            print(f"[WARNING] advantages std too small ({adv_std:.6f}), skipping normalization")
            advantages_batch = advantages_batch - advantages_batch.mean()

        
        target_entropy = -np.log(1.0 / action_space.nvec[0]) * 2
        
        # Use stored entropy from rollout buffer
        current_entropy = float(entropy_buf.mean())
        
        # Minimum entropy constraint: maintain exploration
        min_entropy = target_entropy * 0.35
        
        if use_entropy_schedule:
            # Linear decay from ent_coef_start to ent_coef_end over ent_coef_decay_updates
            progress = min(1.0, update / ent_coef_decay_updates)
            scheduled_ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * progress
        else:
            scheduled_ent_coef = ent_coef
        
        ent_coef_now = scheduled_ent_coef
      
        if current_entropy < target_entropy * 0.45: 
            # Entropy is getting low - increase entropy bonus
            ent_coef_now = scheduled_ent_coef * 2.5  
            print(f"[ENTROPY] Entropy {current_entropy:.3f} < {target_entropy * 0.45:.3f}, ent_coef={ent_coef_now:.4f}")
        elif current_entropy < target_entropy * 0.6:  
            # Entropy is moderately low - slightly increase entropy bonus
            ent_coef_now = scheduled_ent_coef * 1.5
            print(f"[ENTROPY] Entropy {current_entropy:.3f} < {target_entropy * 0.6:.3f}, ent_coef={ent_coef_now:.4f}")
        elif update % 10 == 0:
            print(f"[ENTROPY] Current: {current_entropy:.3f}, Target: {target_entropy:.3f}, scheduled_ent_coef={scheduled_ent_coef:.4f}, ent_coef={ent_coef_now:.4f}")

        update_pg_sum = 0.0
        update_v_sum = 0.0
        update_ent_sum = 0.0
        update_loss_sum = 0.0
        update_mb_count = 0

        # Switch back to train mode for gradient updates
        model.train()

        for epoch in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_logprobs = logprobs_batch[mb_idx]
                mb_advantages = advantages_batch[mb_idx]
                mb_returns = returns_batch[mb_idx]
                mb_values = values_batch[mb_idx]
                mb_agents = agents_batch[mb_idx]
                mb_target_trajs = target_trajs_batch[mb_idx]
                mb_camera_in = camera_intrinsics_batch[mb_idx]
                mb_camera_ex = camera_extrinsics_batch[mb_idx]
                
                # Sanitize minibatch inputs to prevent NaN propagation
                mb_returns = torch.nan_to_num(mb_returns, nan=0.0, posinf=10.0, neginf=-10.0)
                mb_values = torch.nan_to_num(mb_values, nan=0.0, posinf=10.0, neginf=-10.0)
                mb_logprobs = torch.nan_to_num(mb_logprobs, nan=0.0, posinf=10.0, neginf=-10.0)
                mb_advantages = torch.nan_to_num(mb_advantages, nan=0.0, posinf=10.0, neginf=-10.0)

                
                trajectory, trajectory_loss, new_logprob, entropy, new_values, trajectory_dict = model.assess_action_and_value(mb_obs, mb_agents, mb_target_trajs, mb_camera_in, 
                                                                                                         mb_camera_ex, mb_actions, cfg_guidance_scale=1.0)

                if update_mb_count == 0 and epoch == 0:
                    print(f"[DEBUG] new_values: mean={new_values.mean().item():.5f}, std={new_values.std().item():.5f}")
                    print(f"[DEBUG] mb_returns: mean={mb_returns.mean().item():.5f}, std={mb_returns.std().item():.5f}")
                    print(f"[DEBUG] mb_advantages: mean={mb_advantages.mean().item():.5f}, std={mb_advantages.std().item():.5f}")
                
                logratio = new_logprob - mb_logprobs
                logratio = torch.clamp(logratio, min=-10.0, max=10.0)
                ratio = torch.exp(logratio)  

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                mb_values_detached = mb_values.detach()
                mb_returns_detached = mb_returns.detach()
            
                mb_values_clamped = torch.clamp(mb_values_detached, min=-10.0, max=10.0)
                mb_returns_clamped = torch.clamp(mb_returns_detached, min=-10.0, max=10.0)
                new_values_clamped = torch.clamp(new_values, min=-10.0, max=10.0)
                
                value_loss_unclipped = F.smooth_l1_loss(new_values_clamped.squeeze(), mb_returns_clamped, reduction='none')
                value_pred_clipped = mb_values_clamped + torch.clamp(
                        new_values_clamped.squeeze() - mb_values_clamped, 
                        -0.05,
                        0.05
                    )
                value_loss_clipped = F.smooth_l1_loss(value_pred_clipped, mb_returns_clamped, reduction='none')

                value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                
                # Additional safety check for value loss
                if torch.isnan(value_loss).any() or torch.isinf(value_loss).any():
                    print(f"NaN or Inf in value_loss after clamping: value_loss={value_loss}, "
                          f"new_values range: [{new_values.min().item()}, {new_values.max().item()}], "
                          f"mb_returns range: [{mb_returns.min().item()}, {mb_returns.max().item()}]")
                    continue
                
                entropy_loss = entropy.mean()
                
                # Safety check for entropy loss
                if torch.isnan(entropy_loss).any() or torch.isinf(entropy_loss).any():
                    print(f"NaN or Inf in entropy_loss: {entropy_loss}")
                    continue

             
                pg_loss_safe = torch.mean(pg_loss)

                kl_penalty_loss = torch.tensor(0.0, device=main_device)
                approx_kl = torch.tensor(0.0, device=main_device)
                early_stop_kl = False
                gradient_scale = 1.0
                if use_kl_penalty:
                    
                    approx_kl = (ratio - 1.0 - logratio).mean()
    
               
                    kl_val = approx_kl.item()
                    
                    # Track KL for adaptive control (using simple moving average approximation)
                    if not hasattr(train_ppo, '_kl_ema'):
                        train_ppo._kl_ema = kl_val
                    train_ppo._kl_ema = 0.9 * train_ppo._kl_ema + 0.1 * kl_val
                    
                    # Adaptive penalty coefficient
                    # If KL is trending high, increase penalty; if low, decrease it
                    adaptive_kl_coef = kl_coef
                    if train_ppo._kl_ema > kl_target:
                        # KL is trending high - increase penalty aggressively
                        adaptive_kl_coef = kl_coef * (1.0 + 2.0 * (train_ppo._kl_ema - kl_target) / kl_target)
                        adaptive_kl_coef = min(adaptive_kl_coef, kl_coef * 10.0)  # Cap at 10x
                    elif train_ppo._kl_ema < kl_target * 0.5:
                        # KL is low - can relax penalty slightly
                        adaptive_kl_coef = kl_coef * 0.8
                    
                    gradient_scale = 1.0
                    if kl_val > kl_target * 0.5:
                        gradient_scale = min(1.0, kl_target / (kl_val + 1e-6))
                    if kl_val > kl_target:
                        # Scale down gradient proportionally to KL excess
                        gradient_scale = max(0.1, kl_target / kl_val)
                    
                    # This prevents the policy from ever getting too far
                    if kl_val > kl_target * 0.25:
                        # Start applying penalty at 50% of target (proactive)
                        excess = max(0.0, kl_val - kl_target * 0.5)
                        linear_penalty = adaptive_kl_coef * excess * 2.0
                        quadratic_penalty = adaptive_kl_coef * 0.5 * excess ** 2
                        kl_penalty_loss = linear_penalty + quadratic_penalty
                        
                        if update_mb_count == 0 and epoch == 0:
                            print(f"[KL] approx_kl={kl_val:.5f}, ema={train_ppo._kl_ema:.5f}, "
                                  f"adaptive_coef={adaptive_kl_coef:.3f}, grad_scale={gradient_scale:.3f}, "
                                  f"penalty={kl_penalty_loss:.5f}")
                    
                    # If KL is way too high, stop this epoch
                    if kl_val > kl_target * 1.5:
                        early_stop_kl = True
                        print(f"[KL EARLY STOP] approx_kl={kl_val:.5f} > {kl_target * 1.5}, stopping epoch {epoch}")
                    elif kl_val > kl_target * 2.0:
                        # Warning zone - apply extra penalty but don't stop yet
                        kl_penalty_loss = kl_penalty_loss + adaptive_kl_coef * (kl_val - kl_target * 2.0) * 5.0
                        if update_mb_count == 0:
                            print(f"[KL WARNING] approx_kl={kl_val:.5f} > {kl_target * 2.0}, extra penalty applied")
                
                # Early stop this epoch if KL is too high
                if early_stop_kl:
                    break

                loss = pg_loss_safe + vf_coef * value_loss - ent_coef_now * entropy_loss + kl_penalty_loss

                # Final check before backward pass
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"NaN or Inf in total loss: {loss}, "
                          f"pg_loss: {pg_loss_safe}, value_loss: {value_loss}, entropy_loss: {entropy_loss}")
                    continue
                
                # Apply gradient scaling based on KL (soft trust region)
                # gradient_scale is computed above (default 1.0 if KL penalty disabled)
                loss = loss * gradient_scale
                
                # Scale loss to prevent gradient explosion
                loss_scaled = loss / 10.0
                
                optimizer.zero_grad()
                try:
                    loss_scaled.backward()
                except RuntimeError as e:
                    if 'nan' in str(e).lower():
                        print(f"NaN detected during backward pass, skipping update: {e}")
                        continue
                    raise
                
                # Unscale gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(10.0)
                
                # Check for NaN gradients and skip the update if detected
                nan_found = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient detected in parameter {name}")
                            nan_found = True
                            break
                        elif torch.isinf(param.grad).any():
                            print(f"Inf gradient detected in parameter {name}")
                            nan_found = True
                            break
                
                if nan_found:
                    print("Skipping update due to NaN/Inf gradients")
                    continue
                
                print(f'PPO_loss: {loss.item():.5f}, Value_loss: {value_loss.item():.5f}')
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                # Step the learning rate scheduler after each update
                lr_scheduler.step()

                update_pg_sum += float(pg_loss.item())
                update_v_sum += float(value_loss.item())
                update_ent_sum += float(entropy_loss.item())
                update_loss_sum += float(loss.item())
                update_mb_count += 1

        if (update + 1) % clear_cache_every == 0:
            torch.cuda.empty_cache()

        if (update + 1) % 10 == 0:
            mean_return = float(returns_batch.mean().item())
            print(f"Update {update + 1}/{num_updates}, mean return {mean_return:.5f}")

        if update_mb_count > 0:
            avg_pg = update_pg_sum / update_mb_count
            avg_v = update_v_sum / update_mb_count
            avg_ent = update_ent_sum / update_mb_count
            avg_loss = update_loss_sum / update_mb_count
            mean_return = float(returns_batch.mean().item())
            avg_rollout_entropy = float(entropy_buf.mean())  # Average entropy from rollout
            print(f"[Update {update+1}] avg_pg_loss:{avg_pg:.5f}, avg_value_loss:{avg_v:.5f}, avg_entropy:{avg_ent:.5f}, rollout_entropy:{avg_rollout_entropy:.3f}, avg_total_loss:{avg_loss:.5f}, mean_return:{mean_return:.5f}")

            log_entry = {
                "update": int(update + 1),
                "avg_pg_loss": float(avg_pg),
                "avg_value_loss": float(avg_v),
                "avg_entropy": float(avg_ent),
                "avg_total_loss": float(avg_loss),
                "mean_return": float(mean_return),
                "success_rate": round(float(success_rate), 5) if total_timestamps > 0 else 0.0,
                "collision_rate": round(float(collision_rate), 5) if total_timestamps > 0 else 0.0,
                "static_collision_rate": round(float(static_collision_rate), 5) if total_timestamps > 0 else 0.0,
                "deviation_rate": round(float(deviation_rate), 5) if total_timestamps > 0 else 0.0,
                "direction_rate": round(float(direction_rate), 5) if total_timestamps > 0 else 0.0,
                "SDTR": round(float(gameover_ratio), 5) if total_timestamps > 0 else 0.0,
                "longitudinal_jerk_mean": float(v_jerk) if total_timestamps > 0 else 0.0,
                "yaw_jerk_mean": float(yaw_jerk) if total_timestamps > 0 else 0.0,
                "current_lr": float(optimizer.param_groups[0]['lr']),
                "entropy_coefficient": float(ent_coef_now),
                "rollout_entropy": float(avg_rollout_entropy),
                # Probing metrics:
                "traj_probe_best_reward": round(float(traj_probe_best_reward_buf.mean()), 5),
                "traj_probe_reward_gap": round(float(traj_probe_reward_gap_buf.mean()), 5),
                "traj_probe_mean_reward": round(float(traj_probe_mean_reward_buf.mean()), 5),
            }
            training_logs.append(log_entry)
            
            logs_path = os.path.join(os.getcwd(), "training_logs_v1.json")
            with open(logs_path, "w") as f:
                json.dump(training_logs, f, indent=2)

        
        current_lr = optimizer.param_groups[0]['lr']
        if (update + 1) % 10 == 0:
            print(f"[LR] Current learning rate: {current_lr:.2e}")

        # Checkpoint averaging: save checkpoints near end of training
        if use_checkpoint_averaging and update >= checkpoint_avg_start_update:
            if (update - checkpoint_avg_start_update) % checkpoint_avg_frequency == 0:
                if len(checkpoint_states) < checkpoint_avg_max_count:
                    # Deep copy the state dict to avoid reference issues
                    checkpoint_states.append(copy.deepcopy(model.state_dict()))
                    print(f"[Checkpoint Averaging] Saved checkpoint {len(checkpoint_states)}/{checkpoint_avg_max_count} at update {update + 1}")

    state = model.state_dict()
    torch.save(state, save_path)
    print(f"Saved PPO agent to {save_path}")

    # Checkpoint averaging: compute and save averaged model
    if use_checkpoint_averaging and len(checkpoint_states) > 0:
        print(f"\n[Checkpoint Averaging] Averaging {len(checkpoint_states)} checkpoints...")
        
        # Initialize averaged state dict with zeros
        averaged_state = {}
        for key in checkpoint_states[0].keys():
            averaged_state[key] = torch.zeros_like(checkpoint_states[0][key])
        
        # Sum all checkpoint states
        for checkpoint in checkpoint_states:
            for key in checkpoint.keys():
                averaged_state[key] += checkpoint[key]
        
        # Divide by number of checkpoints to get average
        num_checkpoints = len(checkpoint_states)
        for key in averaged_state.keys():
            averaged_state[key] = averaged_state[key] / num_checkpoints
        
        # Save averaged checkpoint
        torch.save(averaged_state, checkpoint_avg_save_path)
        print(f"[Checkpoint Averaging] Saved averaged checkpoint ({num_checkpoints} checkpoints) to {checkpoint_avg_save_path}")
    else:
        print("[Checkpoint Averaging] No checkpoints collected, skipping averaging")

    


# ============================================================================
# BC Training Function
# ============================================================================

def train_bc(
    cuda=0,
    scene=0,
    total_steps=20000,
    learning_rate=3e-4,
    batch_size=4,
    resize_shape=(112, 200),
    device="cuda",
    save_path="bc_reconsimulator_v1.pt",
    device_ids=None,
    scenes=None,
):
    if scenes is None:
        try:
            base = nus_cfg.BASE_DATA_DIR
            scenes = sorted([
                int(d) for d in os.listdir(base)
                if d.isdigit() and os.path.isdir(os.path.join(base, d))
            ])
        except Exception:
            scenes = [scene]

    env = make_env(cuda=cuda, scene=int(scenes[0]), debug=True, resize_shape=resize_shape)

    obs_shape = env.observation_space.shape
    action_space = env.action_space

    main_device = torch.device(device if torch.cuda.is_available() else "cpu")

    config = TransfuserConfig()
    base_model = ModelWrapper(
        config, obs_shape, action_space,
        plan_anchor_path=TRAJ_ANCHOR_PATH,
        training_stage=True,
    ).to(main_device)

    model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    steps_per_scene = max(1, math.ceil(total_steps / max(1, len(scenes))))
    loss_history = []

    print(f'BC training with {len(scenes)} scenes')

    for sc in scenes:
        
        env.set_scene(int(sc))
        
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        prev_ego = env.base_env.start_ego.copy()

        X_buf = []
        yx_buf = []
        yy_buf = []
        camera_intrinsics_buf = []
        camera_extrinsics_buf = []
        target_trajs_buf = []
        agents_buf = []
        steps = 0

        while steps < steps_per_scene:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=main_device).unsqueeze(0)
            
            next_obs, terminated, truncated, _info = env.base_env.step([0, 0, 1])
            
            camera_intrinsics, camera_extrinsics, agents = _info["intrinsics"], _info["extrinsics"], _info["agents"]
            
            next_obs = env._obs_dict_to_tensor(next_obs).astype(np.float32)
            next_ego = env.base_env.start_ego.copy()
            ax_idx, ay_idx = env.compute_expert_action(prev_ego, next_ego)
          
            X_buf.append(obs_tensor)
            yx_buf.append(ax_idx)
            yy_buf.append(ay_idx)
            camera_intrinsics_buf.append(camera_intrinsics)
            camera_extrinsics_buf.append(camera_extrinsics)
            target_trajs_buf.append(_info['target_trajs'])
            agents_buf.append(agents.to(main_device))

            prev_ego = next_ego
            obs = next_obs
            steps += 1

            if len(X_buf) >= batch_size:
                agents_batch = arrange_agents(agents_buf)
                
                x_batch = torch.cat(X_buf, dim=0)
           
                camera_intrinsics_batch = torch.stack(camera_intrinsics_buf, dim=0)
                camera_extrinsics_batch = torch.stack(camera_extrinsics_buf, dim=0)
                target_trajs_batch = torch.stack(target_trajs_buf, dim=0)

                yx = torch.tensor(yx_buf, dtype=torch.long, device=main_device)
                yy = torch.tensor(yy_buf, dtype=torch.long, device=main_device)
              

                trajectory, trajectory_loss, logits_x, logits_y, action, logprob, entropy, value, trajectory_dict = model(x_batch, agents_batch, target_trajs_batch, 
                                                                                             camera_intrinsics_batch, camera_extrinsics_batch)
           
                yx = yx.view(-1)
                yy = yy.view(-1)

                action_entropy_loss = F.cross_entropy(logits_x, yx) + F.cross_entropy(logits_y, yy)
                loss = 0.3 * action_entropy_loss + 1.0 * trajectory_loss
           
                current_loss = loss.item()
                loss_history.append(current_loss)
           
                if len(loss_history) % 10 == 0:
                    avg_loss = sum(loss_history[-50:]) / min(50, len(loss_history))
                    print(f'Step {len(loss_history)}, Avg Loss: {avg_loss:.5f}, Current Loss: {current_loss:.5f}')
                    print(f'Loss components, {action_entropy_loss.item():.5f}, {trajectory_loss.item():.5f}')
                    # Track additional metrics
                    action_accuracy_x = (torch.argmax(logits_x, dim=1) == yx).float().mean()
                    action_accuracy_y = (torch.argmax(logits_y, dim=1) == yy).float().mean()
                    print('Action metrics', action_accuracy_x.item(), action_accuracy_y.item())
                
                if current_loss > 1000:
                    print(f'Warning: Loss exploded to {current_loss}!')
                    break

                optimizer.zero_grad()
                loss.backward()
           
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                scheduler.step()

                X_buf.clear()
                yx_buf.clear()
                yy_buf.clear()
                camera_intrinsics_buf.clear()
                camera_extrinsics_buf.clear()
                target_trajs_buf.clear()
                agents_buf.clear()

            if terminated or truncated:
                obs, _ = env.reset()
                obs = obs.astype(np.float32)
                prev_ego = env.base_env.start_ego.copy()

    state = model.state_dict()
    torch.save(state, save_path)

    if loss_history:
        print(f"Final average loss: {sum(loss_history)/len(loss_history):.5f}")
        print(f"Min loss: {min(loss_history):.5f}, Max loss: {max(loss_history):.5f}")
        
    print(f"Saved BC model to {save_path}")


if __name__ == "__main__":
    register(
        id="ReconSimulator-v0",
        entry_point="reconsimulator.envs.nus:ReconSimulator",
    )

    bc_path = "bc_reconsimulator_v2.pt"
    ppo_path = "ppo_reconsimulator_v2.pt"
    po_ave_path = "ppo_reconsimulator_v2_averaged.pt"

    try:
        base = nus_cfg.BASE_DATA_DIR
        scenes = sorted([
            int(d) for d in os.listdir(base)
            if d.isdigit() and os.path.isdir(os.path.join(base, d))
        ])

    except Exception:
        scenes = [0]

    print(f"Total scenes: {len(scenes)}")

    # Stage 1: IL
    print("\n" + "="*80)
    print("IL Training")
    print("="*80)
    train_bc(
        cuda=0,
        scene=scenes[0] if len(scenes) > 0 else 0,
        total_steps=12000,
        save_path=bc_path,
        scenes=scenes,
    )

    # Stage 2: RL
    print("\n" + "="*80)
    print("RL Training")
    print("="*80)
    train_ppo(
        cuda=0,
        scene=scenes[0] if len(scenes) > 0 else 0,
        total_timesteps=32000,
        num_steps=8,
        save_path=ppo_path,
        init_model_path=bc_path,
        scenes=scenes,
    )