from typing import Dict, Tuple
import torch
import torch.nn as nn



class LSS(nn.Module):
    """
    Lift-Splat-Shoot module for transforming camera features to BEV representation.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (128, 128, 16),
        pc_range: Tuple[float, float, float, float, float, float] = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
        img_h: int = 112,
        img_w: int = 200,
        num_cameras: int = 6,
        feature_dim: int = 512,
        bev_output_channels: int = 64,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.img_h = img_h
        self.img_w = img_w
        self.feature_dim = feature_dim
        self.bev_output_channels = bev_output_channels
    
        self.feature_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Single depth value per pixel
        )
        
        # BEV channel projection
        self.bev_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, bev_output_channels, kernel_size=1),
        )
        
        # Pre-compute depth bins and coordinate grids
        self._setup_coordinate_grids()
    
    def _setup_coordinate_grids(self):
        """Pre-compute coordinate grids for camera to BEV transformation."""
        # Image pixel coordinates (normalized to [-1, 1])
        y_coords = torch.linspace(-1, 1, self.img_h)
        x_coords = torch.linspace(-1, 1, self.img_w)
        y, x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('pixel_coords', torch.stack([x, y, torch.ones_like(x)], dim=0))  # [3, H, W]
        
        # BEV grid coordinates (in meters)
        x_range = torch.linspace(self.pc_range[0], self.pc_range[3], self.grid_size[0])
        y_range = torch.linspace(self.pc_range[1], self.pc_range[4], self.grid_size[1])
        y_bev, x_bev = torch.meshgrid(y_range, x_range, indexing='ij')  # Note: ij indexing
        bev_grid = torch.stack([x_bev, y_bev, torch.zeros_like(x_bev)], dim=2)  # [grid_h, grid_w, 3]
        self.register_buffer('bev_grid', bev_grid)  # [grid_h, grid_w, 3]
    
    def get_3d_points(self, image_features: torch.Tensor, 
                      camera_intrinsics: torch.Tensor, 
                      camera_extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Convert image features to 3D points in world coordinates.
        """
        B, N, C, H, W = image_features.shape
        
        # Predict depth for each pixel
        features_flat = image_features.view(B * N, C, H, W)
        depth_map = self.depth_head(features_flat)  # [B*N, 1, H, W]
        depth_map = depth_map.view(B, N, 1, H, W)
        
        # Normalize depth to valid range [1, 60] meters
        depth_map = torch.sigmoid(depth_map) * 59 + 1
        
        # Get pixel coordinates
        pixel_coords = self.pixel_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        pixel_coords = pixel_coords.repeat(B, N, 1, 1, 1)  # [B, N, 3, H, W]
        
        # Back-project to camera coordinates
        K = camera_intrinsics
        eps = 1e-4
        K_stable = K + torch.eye(3, device=K.device, dtype=K.dtype).unsqueeze(0).unsqueeze(0) * eps
        K_inv = K_stable.inverse()  # [B, N, 3, 3]
        # Clamp inverse values to prevent explosion
        K_inv = torch.clamp(K_inv, min=-1e6, max=1e6)
     
        # Reshape for batched matmul
        pixel_coords_flat = pixel_coords.view(B * N, 3, H * W)  # [B*N, 3, H*W]
        depth_flat = depth_map.view(B * N, 1, H * W)  # [B*N, 1, H*W]
        K_inv_flat = K_inv.view(B * N, 3, 3)  # [B*N, 3, 3]
        
        # Scale by depth and apply K_inv
        points_cam_flat = pixel_coords_flat * depth_flat  # [B*N, 3, H*W]
     
        points_cam_flat = torch.clamp(points_cam_flat, min=-1000.0, max=1000.0)
        points_cam_flat = torch.bmm(K_inv_flat, points_cam_flat)  # [B*N, 3, H*W]
        # Clamp again after transformation to prevent explosion
        points_cam_flat = torch.clamp(points_cam_flat, min=-1000.0, max=1000.0)
        points_cam_flat = points_cam_flat.view(B, N, 3, H, W)  # [B, N, 3, H, W]
        
        # Transform to world coordinates
        T = camera_extrinsics  # [B, N, 4, 4]
        
        # Convert to homogeneous coordinates
        ones = torch.ones(B, N, 1, H, W, device=image_features.device)
        points_cam_homo = torch.cat([points_cam_flat, ones], dim=2)  # [B, N, 4, H, W]
        
        # Apply transformation
        T_flat = T.view(B * N, 4, 4)
        points_cam_homo_flat = points_cam_homo.view(B * N, 4, H * W)
        points_world_flat = torch.bmm(T_flat, points_cam_homo_flat)  # [B*N, 4, H*W]
        points_world_flat = points_world_flat[:, :3, :].view(B, N, 3, H, W)  # [B, N, 3, H, W]
        # Clamp world coordinates to prevent explosion
        points_world_flat = torch.clamp(points_world_flat, min=-200.0, max=200.0)
        
        return points_world_flat, depth_map
    
    def splat_to_bev(self, image_features: torch.Tensor, 
                     points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points onto BEV grid and accumulate features.
        
        Args:
            image_features: [B, N, C, H, W]
            points_3d: [B, N, 3, H, W]
            
        Returns:
            bev_features: [B, C, grid_h, grid_w]
        """
        B, N, C, H, W = image_features.shape
        grid_h, grid_w = self.grid_size[1], self.grid_size[0]  # y, x
        
        # Flatten spatial dimensions
        points_flat = points_3d.view(B, N, 3, -1)  # [B, N, 3, H*W]
        features_flat = image_features.view(B, N, C, -1)  # [B, N, C, H*W]
        
        # Convert world coordinates to BEV grid indices
        # x index: (x - x_min) / voxel_size
        # y index: (y - y_min) / voxel_size
        voxel_size_x = (self.pc_range[3] - self.pc_range[0]) / grid_w
        voxel_size_y = (self.pc_range[4] - self.pc_range[1]) / grid_h
        
        x_idx = ((points_flat[..., 0, :] - self.pc_range[0]) / voxel_size_x).long()
        y_idx = ((points_flat[..., 1, :] - self.pc_range[1]) / voxel_size_y).long()
        
        # Clamp indices to valid range
        x_idx = torch.clamp(x_idx, 0, grid_w - 1)
        y_idx = torch.clamp(y_idx, 0, grid_h - 1)
        
        # Initialize BEV accumulator
        bev_features = torch.zeros(B, C, grid_h, grid_w, 
                                   device=image_features.device, dtype=image_features.dtype)
        bev_counts = torch.zeros(B, 1, grid_h, grid_w, 
                                 device=image_features.device, dtype=image_features.dtype)
        
        # Accumulate features
        for b in range(B):
            for n in range(N):
                # Get valid indices (within range)
                valid_mask = (x_idx[b, n] >= 0) & (x_idx[b, n] < grid_w) & \
                             (y_idx[b, n] >= 0) & (y_idx[b, n] < grid_h)
                
                if valid_mask.any():
                    # Get valid indices and features
                    valid_x = x_idx[b, n][valid_mask]
                    valid_y = y_idx[b, n][valid_mask]
                    valid_features = features_flat[b, n, :, valid_mask]  # [C, num_valid]
                    
                    # Accumulate features using index_add_
                    flat_indices = valid_y * grid_w + valid_x  # Flatten 2D to 1D
                    bev_flat = bev_features[b].view(C, -1)  # [C, grid_h*grid_w]
                    bev_counts_flat = bev_counts[b].view(1, -1)  # [1, grid_h*grid_w]
                    
                    bev_flat.index_add_(1, flat_indices, valid_features)
                    bev_counts_flat.index_add_(1, flat_indices, 
                                               torch.ones(1, valid_features.shape[1], 
                                                         device=image_features.device))
        
        # Normalize by count
        bev_counts = bev_counts.clamp(min=1e-6)
        bev_features = bev_features / bev_counts
        
        return bev_features
    
    def forward(
        self,
        images: torch.Tensor,
        image_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of LSS.
        
        Args:
            images: Input images [B, N, 3, H, W]
            image_features: Image features [B*N, C] or [B, N, C, H, W]
            camera_intrinsics: Camera intrinsic matrix [B, N, 3, 3]
            camera_extrinsics: Camera extrinsic matrix [B, N, 4, 4]
            
        Returns:
            bev_features: BEV features [B, bev_output_channels, grid_h, grid_w]
        """
        B, N, _, H, W = images.shape
        
        # Handle flattened features [B*N, C] -> reshape to [B, N, C, H, W]
        if image_features.dim() == 2:
            # Features are flattened: [B*N, C]
            C_flat = image_features.shape[1]
            assert C_flat == self.feature_dim, f"Feature dimension mismatch: expected {self.feature_dim}, got {C_flat}"

            image_features = image_features.view(B, N, C_flat, 1, 1).expand(B, N, C_flat, H, W)
        else:
            # Features are already spatial: [B, N, C, H, W]
            _, _, C_flat, _, _ = image_features.shape
            assert C_flat == self.feature_dim, f"Feature dimension mismatch: expected {self.feature_dim}, got {C_flat}"
        
        # Project features
        features_flat = image_features.view(B * N, C_flat, H, W).contiguous()
        features_proj = self.feature_proj(features_flat)  # [B*N, C, H, W]
        features_proj = features_proj.view(B, N, C_flat, H, W)
        
        # Get 3D points from depth prediction
        points_3d, depth_map = self.get_3d_points(features_proj, camera_intrinsics, camera_extrinsics)
        
        # Splat features to BEV grid
        bev_features = self.splat_to_bev(features_proj, points_3d)  # [B, C, grid_h, grid_w]
        
        # Project to output channels
        bev_features = self.bev_proj(bev_features)  # [B, bev_output_channels, grid_h, grid_w]
        
        return bev_features