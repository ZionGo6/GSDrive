import os
import copy
import math
import json
import pickle
import torch
import numpy as np
import gymnasium as gym
from reconsimulator.envs.tool import get_splat, get_sky_view, move_to_device, slerp
from reconsimulator.envs import nus_config as cfg
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.spatial.distance import cdist

TRANSFORM_MATRIX = torch.eye(4, dtype=torch.float32).cuda()

class ReconSimulator(gym.Env):
    def __init__(self, cuda=0, scene=0, debug=True):
        self.device = f"cuda:{cuda}"
        self.debug = debug
        self.scene = scene
        self.w, self.h = 800, 450
        self.last_collision_frame = None
        self.last_collision_type = None

        # Observation space: 6 camera RGB views
        self.observation_space = gym.spaces.Dict({
            name: gym.spaces.Box(low=0, high=255, shape=(self.h, self.w, 3), dtype=np.uint8)
            for name in ["front", "front_left", "front_right", "back_left", "back_right", "back"]
        })

        # Action space: discrete anchor indices
        self.action_space = gym.spaces.MultiDiscrete([9, 9])

        # Load trainer
        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        # Frame control
        self.step_frames = 5
        self.final_frame = 186
        self.now_frame = 0

        # Load all data
        self._load_camera_and_images()
        self._load_ego_and_cam_matrices()
        self._load_expert_ego_frames()
        self._load_plan_anchors()
        self._load_token_mappings()

        self.all_camera_now = []
        self.get_all_point_for_expert()

    # ------------------------- Private loading functions ------------------------ #
    def _load_camera_and_images(self):
        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

    def _load_ego_and_cam_matrices(self):
        cam2ego = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.camera_front_start = ego2world @ cam2ego
        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world

        # Load all camera-to-ego matrices
        self.cam2ego = [
            np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
            for i in range(6)
            if os.path.exists(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt"))
        ]

    def _load_expert_ego_frames(self):
        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)

    def _load_plan_anchors(self):
        self.plan_anchors = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_FILE).astype(np.float32))
        self.plan_anchors_yaw = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_YAW_FILE).astype(np.float32)) * 5
        self.plan_anchors_mask = torch.from_numpy(np.load(cfg.PLAN_ANCHORS_MASK_FILE).reshape(-1))
        self.x_anchor = 9
        self.y_anchor = 9
        self.traj_anchors = torch.from_numpy(np.load(cfg.TRAJ_ANCHORS_FILE).astype(np.float32))

    def _load_token_mappings(self):
        frame2token_path = os.path.join(cfg.FRAME2TOKEN_DIR, f"{self.scene:03d}.json")
        with open(frame2token_path, 'r') as f:
            data = json.load(f)
            self.frame2token = {v: k for k, v in data.items()}
        with open(cfg.TOKEN2VAD_FILE, 'rb') as f:
            self.token2vad = pickle.load(f)

    # ------------------------- Observation & Info ------------------------ #
    def _get_obs(self):
        """
        Compute observation images from all active cameras using the trainer.
        """
        self.now_observe_image = []
        with torch.no_grad():
            for cam in self.all_camera_now:
                cam_info, img_info = cam
                results = self.trainer(img_info, cam_info)
                rgb = results['rgb'].clamp(0, 1).cpu().numpy()
                scaled_rgb = (rgb * 255).astype(np.uint8)
                self.now_observe_image.append(scaled_rgb)
        
        return {
            "front": self.now_observe_image[0],
            "front_left": self.now_observe_image[1],
            "front_right": self.now_observe_image[2],
            "back_left": self.now_observe_image[3],
            "back_right": self.now_observe_image[4],
            "back": self.now_observe_image[5],
        }

    def _get_info(self):
        intrinsics_list = []
        extrinsics_list = []
        
        for cam_info, _ in self.all_camera_now:
            intrinsics_list.append(cam_info['intrinsics'])
            c2w = cam_info['camera_to_world']
            w2c = torch.linalg.inv(c2w)
            extrinsics_list.append(w2c)

        info = {
            "intrinsics": torch.stack(intrinsics_list, dim=0),
            "extrinsics": torch.stack(extrinsics_list, dim=0),
            "traj_anchors": self.traj_anchors.view(-1, 6, 2)
        }
        
        token = self._get_current_token()
    
        traj = self._get_current_traj(token)
        agents = self._get_current_agents(token)
        agents = torch.nan_to_num(agents, nan=0.0, posinf=10.0, neginf=-10.0)

        cusum_traj = traj.cumsum(axis=-2)

        info["token"] = token
        info["target_trajs"] = traj
        info["cumsum_target_trajs"] = cusum_traj
        info["agents"] = agents
        
        vad_info = self.token2vad.get(token, {})
        info["is_dynamic_collision_box"] = bool(vad_info.get("is_dynamic_collision_box", False))
        info["static_collision_score"] = float(vad_info.get("static_collision_score", 0.0))
        
        return info

    def _get_current_token(self):
        """Get the token for the current frame."""
        frame_idx = self.now_frame
    
        token = self.frame2token[frame_idx]
        return token

    def _get_current_traj(self, token):
        """Get the trajectory data for the given token."""
        
        vad_info = self.token2vad.get(token)
       
        
        if 'gt_ego_fut_trajs' in vad_info:
            traj_data = vad_info['gt_ego_fut_trajs']
            
        return torch.from_numpy(traj_data.astype(np.float32))
    
    def _get_current_agents(self, token):
        """Get the agent data for the given token."""
        
        vad_info = self.token2vad.get(token)
       
        if 'gt_agent_lcf_feat' in vad_info:
            agent_data = vad_info['gt_agent_lcf_feat']
            
        return torch.from_numpy(agent_data.astype(np.float32))
        

    # ------------------------- Gym API ------------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.update(seed)
        else:
            self.now_frame = 0
            self.last_collision_frame = None
            self.last_collision_type = None
        self.start_ego = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.start_ego = np.linalg.inv(self.camera_front_start) @ self.start_ego

        self.all_camera_now = []
        for i in range(6):
            if self.now_frame > self.final_frame:
                self.now_frame = self.final_frame
            cam_info = copy.deepcopy(self.all_cams[i])
            cam_info = move_to_device(cam_info, self.device)
            cam_info['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i], device=self.device, dtype=torch.float32)
            cam_info['camera_to_world'] = cam_info['camera_to_world'] @ TRANSFORM_MATRIX

            img_info = copy.deepcopy(self.all_images[i])
            img_info = move_to_device(img_info, self.device)
            img_info['origins'], img_info['viewdirs'], img_info['direction_norm'] = get_sky_view(
                cam_info['camera_to_world'], cam_info['intrinsics'], self.device, self.h, self.w
            )
            img_info['normed_time'] = torch.tensor(self.trainer.normalized_timestamps[self.now_frame].item())
            self.all_camera_now.append((cam_info, img_info))

        

        obs = self._get_obs()
        info = self._get_info()
        self.all_camera_now = []

        return obs, info

    def step(self, action):
        self.now_frame += self.step_frames
        ax_index, ay_index, flag = action
      

        if self.debug or flag:
            self.start_ego = np.linalg.inv(self.camera_front_start) @ np.loadtxt(
                os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{self.now_frame:03d}.txt")
            )
         

        else:
            selected_idx = ax_index * self.y_anchor + ay_index
         

            anchor = self.plan_anchors[selected_idx]
   

            if anchor.ndim == 2 and anchor.shape[-1] >= 2:
                future_x = float(anchor[-1, 0].item())
                future_y = float(anchor[-1, 1].item())
            elif anchor.ndim == 1:
                if anchor.numel() == 2:
                    future_x = float(anchor[0].item())
                    future_y = float(anchor[1].item())
                elif anchor.numel() % 2 == 0:
                    anchor2 = anchor.view(-1, 2)
                    future_x = float(anchor2[-1, 0].item())
                    future_y = float(anchor2[-1, 1].item())
                else:
                    future_x = float(anchor[0].item())
                    future_y = float(anchor[1].item()) if anchor.numel() > 1 else 0.0
            else:
                future_x, future_y = 0.0, 0.0
            future_yaw = float(self.plan_anchors_yaw[selected_idx].item())
            tpt = np.array([
                [math.cos(future_yaw), -math.sin(future_yaw), 0, future_x],
                [math.sin(future_yaw), math.cos(future_yaw), 0, future_y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            self.start_ego = self.start_ego @ tpt
        self.start_ego[1][-1] = self.updateGroundDistance()
            
        
        w, h = 800, 450
        for i in range(6):
            if self.now_frame > self.final_frame:
                self.now_frame = self.final_frame
            loaded_cam_infos = copy.deepcopy(self.all_cams[i])
            loaded_cam_infos = move_to_device(loaded_cam_infos, self.device)
            loaded_cam_infos['camera_to_world'] = torch.tensor(self.start_ego @ self.cam2ego[i]).to(self.device).to(torch.float32)
            loaded_cam_infos['camera_to_world'] = loaded_cam_infos['camera_to_world'] @ TRANSFORM_MATRIX
            loaded_img_infos = copy.deepcopy(self.all_images[i])
            loaded_img_infos = move_to_device(loaded_img_infos, self.device)
            loaded_img_infos['origins'],\
            loaded_img_infos['viewdirs'], \
            loaded_img_infos['direction_norm'] = get_sky_view(loaded_cam_infos['camera_to_world'],\
                                                                  loaded_cam_infos['intrinsics'],\
                                                                    self.device,h,w)
            loaded_img_infos['normed_time'] = torch.tensor(self.trainer.normalized_timestamps[self.now_frame].item()) 
            self.all_camera_now.append((loaded_cam_infos, loaded_img_infos))

        observation = self._get_obs()
        info = self._get_info()
        self.all_camera_now = []

        terminated, truncated = False, False
        if self.now_frame == self.final_frame - 1:
            terminated = True
        else:
            terminated = False

        if self.check_coliision():
            truncated = True
        else:
            truncated = False

        return observation, terminated, truncated, info
    
    def check_coliision(self):
        self.last_collision_frame = None
        self.last_collision_type = None
        frame_idx = self.now_frame
        token = None
        if hasattr(self, "frame2token"):
            if frame_idx in self.frame2token:
                token = self.frame2token[frame_idx]
            elif str(frame_idx) in self.frame2token:
                token = self.frame2token[str(frame_idx)]
        if token is None or not hasattr(self, "token2vad"):
            return False
        vad_info = self.token2vad.get(token)
        if vad_info is None:
            return False
        is_dynamic_collision = bool(vad_info.get("is_dynamic_collision_box", False))
        static_score = float(vad_info.get("static_collision_score", 0.0))
        static_threshold = 25000.0
        is_static_collision = (not is_dynamic_collision) and static_score > static_threshold
        if is_dynamic_collision:
            self.last_collision_frame = frame_idx
            self.last_collision_type = "dynamic"
        elif is_static_collision:
            self.last_collision_frame = frame_idx
            self.last_collision_type = "static"
        return is_dynamic_collision or is_static_collision
    
    def update(self, scene: int, *, step_frames: int = None):
        self.scene = int(scene)
        if step_frames is not None:
            self.step_frames = int(step_frames)        
        self.now_frame = 0
        self.all_camera_now = []
        self.save = None

        self.trainer, self.num_timesteps = get_splat(self.device, self.scene)
        self.trainer.eval()

        with open(cfg.ALL_CAMS_FILE, "rb") as f:
            self.all_cams = pickle.load(f)
        with open(cfg.ALL_IMAGES_FILE, "rb") as f:
            self.all_images = pickle.load(f)

        cam2ego_0 = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/0.txt"))
        ego2world_0 = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/000.txt"))
        self.camera_front_start = ego2world_0 @ cam2ego_0

        self.start_ego = np.linalg.inv(self.camera_front_start) @ ego2world_0

        self.cam2ego = []
        for i in range(6):
            cam_path = os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/cam2ego/{i}.txt")
            if os.path.exists(cam_path):
                self.cam2ego.append(np.loadtxt(cam_path))

        self.all_expert_ego = []
        for i in range(0, self.final_frame + self.step_frames, self.step_frames):
            expert_world = np.loadtxt(os.path.join(cfg.BASE_DATA_DIR, f"{self.scene:03d}/ego_pose/{i:03d}.txt"))
            expert_world = np.linalg.inv(self.camera_front_start) @ expert_world
            self.all_expert_ego.append(expert_world)
        self.get_all_point_for_expert()

    def get_all_point_for_expert(self):
        self.expert_world_all = []
        for i in range(len(self.all_expert_ego) - 1):
            start_matrix = self.all_expert_ego[i]
            end_matrix = self.all_expert_ego[i + 1]
            for alpha in np.linspace(0, 1, 40): 
                translation = (1 - alpha) * start_matrix[:3, 3] + alpha * end_matrix[:3, 3]
                start_rot = R.from_matrix(start_matrix[:3, :3])
                end_rot = R.from_matrix(end_matrix[:3, :3])
                interp_rot = slerp(start_rot, end_rot, alpha)
                new_matrix = np.eye(4)
                new_matrix[:3, :3] = interp_rot.as_matrix()
                new_matrix[:3, 3] = translation
                self.expert_world_all.append(new_matrix)

        self.expert_pair = [matrix[:3, 3][[0, 2]] for matrix in self.expert_world_all]
        self.expert_altitude  = [matrix[:3, 3][[1]] for matrix in self.expert_world_all]


    def updateGroundDistance(self):
        start_ego_position = self.start_ego[:3, 3][[0, 2]]
        distances = cdist([start_ego_position], self.expert_pair, 'euclidean')[0]
        nearest_indices = np.argsort(distances)[:1] 
       
        return self.expert_altitude[nearest_indices[0]]
