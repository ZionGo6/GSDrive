import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_infra.backbone.modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention, gen_sineembed_for_position_1d, GridSampleCrossBEVAttentionScorer



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ResidualMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and dropout"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        x = self.norm2(x)
        return x


class PPOValueHead(nn.Module):
    def __init__(self, config, d_model=256):
        super().__init__()
        self.agent_pool = nn.AdaptiveMaxPool1d(1) 
        self.img_pool = nn.AdaptiveAvgPool1d(1)

        input_dim = 256 + 256 + 512  # traj_context + agent_context + img_context
        hidden_dim = config.tf_d_ffn
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=0.1) for _ in range(4)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize value head with reasonable scale to match expected returns
        nn.init.orthogonal_(self.output_head[-1].weight, gain=0.5)
        nn.init.constant_(self.output_head[-1].bias, 0.0)

    def forward(self, traj_query, agent_query, img_feat):

        agent_context = self.agent_pool(agent_query.transpose(1, 2)).squeeze(-1)

        img_feat = img_feat.view(img_feat.shape[0] // 6, 6, 512)
        img_context = self.img_pool(img_feat.transpose(1, 2)).squeeze(-1)
        
        traj_context = traj_query.squeeze(1) # [B, 256]
       
        state_representation = torch.cat([traj_context, agent_context, img_context], dim=-1)
        
        # Input projection
        x = self.input_proj(state_representation)
        
        # Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Output head
        value = self.output_head(x)
        
        # Clamp value output to reasonable range and sanitize
        value = torch.clamp(value, min=-10.0, max=10.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return value


class ResidualPolicyBlock(nn.Module):
    """Residual block for policy head with pre-norm architecture."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class DirectPolicyHead(nn.Module):
    """
    Direct policy head that outputs action logits directly from features.
    Used in residual connection: final_logits = α * traj_logits + (1-α) * direct_logits
    """
    def __init__(self, config, num_anchors: int, hidden_dim: int = 512):
        super().__init__()
        self.num_anchors = num_anchors
        
        traj_dim = 256
        agent_dim = 256
        img_dim = 512
        
        input_dim = traj_dim + agent_dim + img_dim  # 1024
        
        self.agent_pool = nn.AdaptiveMaxPool1d(1)
        self.img_pool = nn.AdaptiveAvgPool1d(1)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        
        self.head_x = nn.Linear(hidden_dim, num_anchors)
        self.head_y = nn.Linear(hidden_dim, num_anchors)
        
        # Initialize with small weights for stable start
        nn.init.orthogonal_(self.head_x.weight, gain=0.5)
        nn.init.orthogonal_(self.head_y.weight, gain=0.5)
        nn.init.constant_(self.head_x.bias, 0.0)
        nn.init.constant_(self.head_y.bias, 0.0)
    
    def forward(self, traj_query, agents_query, img_feat):
        B = traj_query.shape[0]
        
        agent_context = self.agent_pool(agents_query.transpose(1, 2)).squeeze(-1)
        
        img_feat = img_feat.view(B, 6, -1)  # [B, 6, 512]
        img_context = self.img_pool(img_feat.transpose(1, 2)).squeeze(-1)  # [B, 512]
        
        traj_context = traj_query.squeeze(1)
        
        features = torch.cat([traj_context, agent_context, img_context], dim=-1)
        
        x = self.feature_proj(features)
        for block in self.residual_blocks:
            x = block(x)
        
        logits_x = self.head_x(x)
        logits_y = self.head_y(x)
        
        return logits_x, logits_y


class ActionHead(nn.Module):
    def __init__(self, config, action_nvec):
        super().__init__()
        d_model = 256
        hidden_dim = config.tf_d_ffn
        
        self.img_proj = nn.Linear(512, d_model)
        self.agent_proj = nn.Linear(256, d_model)
        self.agent_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.image_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Shared feature processing with residual blocks
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Residual blocks for shared features
        self.shared_residual_blocks = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(3)
        ])
        
        # Policy Heads with residual connections
        # Head X
        self.policy_head_x_proj = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head_x_residual = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        self.policy_head_x_out = nn.Linear(hidden_dim, action_nvec[0])
        
        # Head Y  
        self.policy_head_y_proj = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head_y_residual = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        self.policy_head_y_out = nn.Linear(hidden_dim, action_nvec[0])
        
        # Initialize policy heads
        nn.init.orthogonal_(self.policy_head_x_out.weight, gain=0.5)
        nn.init.orthogonal_(self.policy_head_y_out.weight, gain=0.5)
        nn.init.constant_(self.policy_head_x_out.bias, 0.0)
        nn.init.constant_(self.policy_head_y_out.bias, 0.0)

    def forward(self, img_feat, traj_query, agent_query):
     
        # Align shapes
        B = traj_query.shape[0]
        img_feat = self.img_proj(img_feat).view(B, 6, -1) # [B, 6, 256]
        agent_feat = self.agent_proj(agent_query)         # [B, 30, 256]
       
        # Cross-Attention: Traj looks at Agents
        x, _ = self.agent_attention(traj_query, agent_feat, agent_feat)
       
        # Cross-Attention: Traj looks at Images
        x, _ = self.image_attention(x, img_feat, img_feat)

        x = x.squeeze(1)
      
        # Shared feature processing with residual connections
        shared = self.shared_proj(x)
        for residual_block in self.shared_residual_blocks:
            shared = residual_block(shared)
        
        # Policy head X with residual connections
        x_feat = self.policy_head_x_proj(shared)
        for residual_block in self.policy_head_x_residual:
            x_feat = residual_block(x_feat)
        action_x = self.policy_head_x_out(x_feat)
        
        # Policy head Y with residual connections
        y_feat = self.policy_head_y_proj(shared)
        for residual_block in self.policy_head_y_residual:
            y_feat = residual_block(y_feat)
        action_y = self.policy_head_y_out(y_feat)
        
        return action_x, action_y


class TrajectoryGuidedActionHead(nn.Module):
    """
    Action head that uses ground truth trajectory to guide action prediction.
    """
    def __init__(self, config, action_nvec, num_anchors_x=9, num_anchors_y=9):
        super().__init__()
        d_model = 256
        hidden_dim = config.tf_d_ffn
        self.num_anchors_x = num_anchors_x
        self.num_anchors_y = num_anchors_y
        self.action_nvec = action_nvec
        
        self.traj_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(6 * d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.anchor_x_embed = nn.Embedding(num_anchors_x, d_model // 2)
        self.anchor_y_embed = nn.Embedding(num_anchors_y, d_model // 2)
        
        self.traj_anchor_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Feature projections
        self.img_proj = nn.Linear(512, d_model)
        self.agent_proj = nn.Linear(256, d_model)
        
        # Agent and image attention
        self.agent_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.image_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Trajectory guidance fusion
        self.traj_guidance_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # traj_feat + attended_traj
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Shared feature processing
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Residual blocks
        self.shared_residual_blocks = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(3)
        ])
        
        # Policy heads
        # Head X with trajectory guidance
        self.policy_head_x_proj = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head_x_residual = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        self.policy_head_x_out = nn.Linear(hidden_dim, action_nvec[0])
        
        # Head Y with trajectory guidance
        self.policy_head_y_proj = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head_y_residual = nn.ModuleList([
            ResidualPolicyBlock(hidden_dim, dropout=0.1) for _ in range(2)
        ])
        self.policy_head_y_out = nn.Linear(hidden_dim, action_nvec[0])
        
        self.anchor_scorer_x = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.anchor_scorer_y = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        # Initialize weights
        nn.init.orthogonal_(self.policy_head_x_out.weight, gain=0.5)
        nn.init.orthogonal_(self.policy_head_y_out.weight, gain=0.5)
        nn.init.constant_(self.policy_head_x_out.bias, 0.0)
        nn.init.constant_(self.policy_head_y_out.bias, 0.0)
    
    def compute_trajectory_anchor_scores(self, traj_embedding, device):
        """
        Compute similarity scores between trajectory and each anchor.
        """
        B = traj_embedding.shape[0]
        
        # Get all anchor embeddings
        anchor_x_idx = torch.arange(self.num_anchors_x, device=device)
        anchor_y_idx = torch.arange(self.num_anchors_y, device=device)
        
        anchor_x_embed = self.anchor_x_embed(anchor_x_idx)
        anchor_y_embed = self.anchor_y_embed(anchor_y_idx)

        traj_expanded = traj_embedding.unsqueeze(1)
        
        # Compute similarity with x anchors
        anchor_x_full = F.pad(anchor_x_embed, (0, anchor_x_embed.shape[-1]))
        anchor_x_scores = self.anchor_scorer_x(
            traj_expanded.expand(-1, self.num_anchors_x, -1) + anchor_x_full.unsqueeze(0)
        ).squeeze(-1)
        
        # Compute similarity with y anchors
        anchor_y_full = F.pad(anchor_y_embed, (anchor_y_embed.shape[-1], 0))
        anchor_y_scores = self.anchor_scorer_y(
            traj_expanded.expand(-1, self.num_anchors_y, -1) + anchor_y_full.unsqueeze(0)
        ).squeeze(-1)
        
        return anchor_x_scores, anchor_y_scores
    
    def forward(self, img_feat, traj_query, agent_query, target_traj=None, plan_anchors=None, plan_anchors_yaw=None):
        """
        Forward pass with trajectory-guided action prediction.
        """
        B = traj_query.shape[0]
        device = traj_query.device
        
        # Align shapes
        img_feat = self.img_proj(img_feat).view(B, 6, -1)  # [B, 6, 256]
        agent_feat = self.agent_proj(agent_query)  # [B, 30, 256]
        
        x, _ = self.agent_attention(traj_query, agent_feat, agent_feat)
        x, _ = self.image_attention(x, img_feat, img_feat)
        x = x.squeeze(1)
        
        guidance_info = {}
        
        if target_traj is not None:
            target_traj = target_traj.to(device)
            traj_embedding = self.traj_encoder(target_traj)

            traj_anchor_scores_x, traj_anchor_scores_y = self.compute_trajectory_anchor_scores(
                traj_embedding, device
            )
            
            guidance_info['traj_anchor_scores_x'] = traj_anchor_scores_x
            guidance_info['traj_anchor_scores_y'] = traj_anchor_scores_y
            guidance_info['traj_embedding'] = traj_embedding
            
            # Fuse trajectory guidance with attended features
            traj_guidance = self.traj_guidance_proj(
                torch.cat([x, traj_embedding], dim=-1)
            )
            x = x + traj_guidance * 0.5
        
        # Shared feature processing
        shared = self.shared_proj(x)
        for residual_block in self.shared_residual_blocks:
            shared = residual_block(shared)
        
        # Policy head X
        x_feat = self.policy_head_x_proj(shared)
        for residual_block in self.policy_head_x_residual:
            x_feat = residual_block(x_feat)
        action_x = self.policy_head_x_out(x_feat)
        
        # Policy head Y
        y_feat = self.policy_head_y_proj(shared)
        for residual_block in self.policy_head_y_residual:
            y_feat = residual_block(y_feat)
        action_y = self.policy_head_y_out(y_feat)
        
        if target_traj is not None:
            action_x = action_x + traj_anchor_scores_x * 0.3
            action_y = action_y + traj_anchor_scores_y * 0.3
        
        return action_x, action_y, guidance_info


class FMPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_mode=18, if_zeroinit_reg=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        
        self.velocity_field_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, ego_fut_ts * 3),
        )
        
        self.temporal_conv = nn.Conv1d(
            in_channels=ego_fut_ts, 
            out_channels=ego_fut_ts, 
            kernel_size=3, 
            padding=1,
            groups=1
        )
        
        # Residual connection for velocity prediction
        self.vel_residual_proj = nn.Linear(embed_dims, embed_dims // 2)

        self.flow_velocity_branch = nn.Sequential(
            nn.Linear(embed_dims + 3, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, 3)
        )
        
        self.if_zeroinit_reg = False
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
            nn.init.constant_(self.velocity_field_branch[-1].weight, 0)
            nn.init.constant_(self.velocity_field_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode, embed_dim = traj_feature.shape
        
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
        
        vel_input = traj_feature.view(bs * ego_fut_mode, -1)
        
        vel_res = self.vel_residual_proj(vel_input)
        vel_out = self.velocity_field_branch(vel_input)
        vel_out = vel_out.view(bs * ego_fut_mode, self.ego_fut_ts, 3)
        
        vel_out = self.temporal_conv(vel_out)
        vel_field = vel_out.view(bs, ego_fut_mode, self.ego_fut_ts, 3)
        
        return plan_reg, plan_cls, vel_field


class ModulationLayer(nn.Module):
    def __init__(self, embed_dims: int, condition_dims: int):
        super().__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(self, traj_feature, time_embed, global_cond=None, global_img=None):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([global_img, global_feature], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model, config.tf_num_head, num_points=num_poses,
            config=config, in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = FMPlanningRefinementModule(
            embed_dims=config.tf_d_model, ego_fut_ts=num_poses, ego_fut_mode=18,
        )

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, 
                agents_query, ego_query, time_embed, status_encoding, global_img=None):

        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query, agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)
        
        traj_feature = traj_feature.view(traj_feature.shape[0], -1, 18, traj_feature.shape[-1])
        bs, num_groups, _, _ = traj_feature.shape
        traj_feature = traj_feature.view(-1, 18, traj_feature.shape[-1])
        poses_reg, poses_cls, vel_field = self.task_decoder(traj_feature)

        poses_reg = poses_reg.view(bs, 18*num_groups, 6, 3)
        poses_cls = poses_cls.view(bs, -1, 18)
        vel_field = vel_field.view(bs, 18*num_groups, 6, 3)
 
        poses_reg = poses_reg + noisy_traj_points
        
        poses_reg[..., 2] = poses_reg[..., 2].tanh() * np.pi
        return poses_reg, poses_cls, vel_field, traj_feature


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, 
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        vel_field_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls, vel_field, traj_feature = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, 
                                                     agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            vel_field_list.append(vel_field)
            traj_points = poses_reg[..., :2].clone().detach()

        return poses_reg_list, poses_cls_list, vel_field_list, traj_feature