# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import numpy as np
# from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.sensors.camera import Camera, CameraCfg, save_images_to_file

from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
import math

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# from omni.isaac.lab_tasks.manager_based.manipulation.push.feature_extractor import FeatureExtractor, FeatureExtractorCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def object_pose_in_robot_frame(env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        camera_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera") ) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Extract position and quaternion (with shape [batch_size, 3] and [batch_size, 4] respectively)
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]

    object_pos_w = obj.data.root_pos_w[:, :3]
    object_quat_w = obj.data.root_state_w[:, 3:7]  # Ensure correct extraction

    # Compute relative position
    object_pos_b, _ = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, object_pos_w
    )

    # Compute relative orientation
    robot_quat_w_conj = torch.cat([robot_quat_w[:, :1], -robot_quat_w[:, 1:]], dim=-1)
    # print("Robot Quaternion (Conjugate):", robot_quat_w_conj.shape)
    # print("Object Quaternion:", object_quat_w.shape)
    
    object_quat_b = quat_multiply(robot_quat_w_conj, object_quat_w)
    # object_pos_b[:, 2] += 0.003  # Adjust Z position to account for the table being slightly below the robot
    # This is not great but I am also hardcoding the object height when running on robot

    # This is hacky but just set object pos z to 0.04
    object_pos_b[:, 2] = 0.04
    # TODO make sure to remember to set this on real robot too

    # Return combined relative pose
    object_pose_b = torch.cat([object_pos_b, object_quat_b], dim=-1)
    # print('object_pose_b:', object_pose_b)

    # object_physical_properties(env, object_cfg)
    return object_pose_b

def quat_multiply(q1, q2):
    """Multiplies two quaternions (expects shape [batch_size, 4])."""
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4, "Quaternions must have shape [batch_size, 4]"

    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dim=-1)

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    # # Call image_embeddings to train CNN to regress on object poses but return object_pos_b
    # image_embeddings(env)

    return object_pos_b

def ee_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_base_frame"),
    # ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
) -> torch.Tensor:
    ### NOTE this is to the ee base not the tip
    """The pose (position and orientation) of the end-effector in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    
    # Get world position and orientation of the end-effector
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :3]
    ee_rot_w = ee_frame.data.target_quat_w[..., 0, :4]  # in [w, x, y, z] format
    
    # Transform to the robot's root frame
    ee_pos_b, ee_rot_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w, ee_rot_w
    )

    # Combine position and orientation into a single tensor
    ee_pose_b = torch.cat((ee_pos_b, ee_rot_b), dim=-1)  # Shape: [batch_size, 7]
    # print('ee_pose_b', ee_pose_b)
    # if ee_pos_b[0, 2] < 0.035:
    #     print('EE POS Z IS TOO LOW:', ee_pos_b)
    
    return ee_pose_b

def ee_xy_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_base_frame")
) -> torch.Tensor:
    """The XY position of the end-effector in the robot's root frame."""
    # Get full pose
    ee_pos_b = ee_pose_in_robot_root_frame(env, robot_cfg, ee_frame_cfg)
    # Slice to XY only
    ee_pos_b_xy = ee_pos_b[:, :2]
    return ee_pos_b_xy

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return object_ee_distance

def object_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The velocity of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_vel_w

def image_embeddings(env: ManagerBasedRLEnv,
                     camera_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
                     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),) -> torch.Tensor:
    # global frame_buffer

    camera = env.scene[camera_cfg.name]
    # rgb_image = camera.data.output["rgb"]
    # # print('rgb_image shape:', rgb_image.shape)

    # # print('min and max of rgb_image before:', torch.min(rgb_image), torch.max(rgb_image))
    # rgb_image = rgb_image.float() / 255.0
    # # print('min and max of rgb_image after:', torch.min(rgb_image), torch.max(rgb_image))

    # if rgb_image.dim() == 4:
    #     # Case when the image already includes a batch dimension
    #     rgb_image = rgb_image.permute(0, 3, 1, 2)  # From (N, H, W, C) to (N, C, H, W)
    # elif rgb_image.dim() == 3:
    #     # Case when the image is without a batch dimension
    #     rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)  # From (H, W, C) to (1, C, H, W)
    # else:
    #     raise ValueError(f"Unexpected number of dimensions in rgb_image: {rgb_image.dim()}")

    # # Ensure the batch size is consistent with the initialized buffer
    # if rgb_image.size(0) != batch_size:
    #     raise ValueError(f"Inconsistent batch size. Expected {batch_size}, but got {rgb_image.size(0)}")

    # # # Visualize the stack of frames in the buffer
    # # for i, frame in enumerate(frame_buffer):
    # #     frame = frame * 255.0
    # #     plt.subplot(1, buffer_size, i + 1)
    # #     plt.imshow(frame[0].permute(1, 2, 0).cpu().numpy())
    # #     plt.axis("off")
    # # plt.show()

    # # rgb_image = normalize(rgb_image).to(device)
    # # print('min and max of rgb_image after:', torch.min(rgb_image), torch.max(rgb_image))
    # # rgb_image = rgb_image.to(device)

    # # Add the most recent frame to frame_buffer (FIFO queue)
    # frame_buffer.append(rgb_image.clone())

    # # Concatenate the frames in the buffer along the channel dimension
    # stacked_frames = torch.cat(list(frame_buffer), dim=1)  # Shape: (batch_size, C * buffer_size, H, W)

    # # Flatten the concatenated frames into a single feature vector
    # # feature_size = C * H * W
    # feature_size = C * buffer_size * H * W

    # flattened_vector = stacked_frames.reshape(batch_size, feature_size)  # Shape: (batch_size, feature_size)


    # """The position of the object in the robot's root frame."""
    # robot: RigidObject = env.scene[robot_cfg.name]
    # object: RigidObject = env.scene[object_cfg.name]
    # object_pos_w = object.data.root_pos_w[:, :3]
    # object_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    # )

    # train CNN to regress on object poses
    gt_object_pose = object_pose_in_robot_frame(env, robot_cfg, object_cfg)
    pose_loss, embeddings = env.feature_extractor.step(
        camera.data.output["rgb"],
        camera.data.output["depth"],
        camera.data.output["semantic_segmentation"][..., :3],
        gt_object_pose,
    )

    
    # if not hasattr(env, "extras"):
    #     env.extras = {"log": dict()}

    # if "log" not in env.extras:
    #     env.extras["log"] = dict()
    env.extras["log"]["pose_loss"] = pose_loss

    if pose_loss:
        print('pose_loss:', pose_loss.item())

        # Write to a csv file
        file_folder = "/home/maggiewang/Workspace/IsaacLab/logs"
        file_path = os.path.join(file_folder, "pose_loss_02_03_25__07.csv")
        with open(file_path, 'a') as f:
            f.write(f'{pose_loss}\n')

    # print('pose_loss:', pose_loss)

    # Print embeddings shape
    # print('embeddings shape:', embeddings.shape)
    # Reshape to batch_size, feature_size
    embeddings = embeddings.reshape(embeddings.size(0), -1)
    # print('embeddings shape after reshape:', embeddings.shape)

    # Reshape to batch_size, feature_size where feature size is C*H*W
    # return embeddings
    return gt_object_pose


def object_physical_properties(env: ManagerBasedRLEnv,
                               object_cfg: SceneEntityCfg = SceneEntityCfg("object"),) -> torch.Tensor:
    """The physics material of the object."""
    object: RigidObject = env.scene[object_cfg.name]

    # materials: (5, 2, 3) => (num_envs, num_shapes, 3)
    # 3 means [static_friction, dynamic_friction, restitution]
    materials = object.root_physx_view.get_material_properties()
    # masses: (5, 1) => (num_envs, num_bodies)
    # masses = object.root_physx_view.get_masses()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    materials = materials.to(device)
    # masses = masses.to(device)

    # Extract static friction across *all shapes* => shape (5, 2)
    # This slices out the 0-th property dimension from the last axis
    all_static_friction = materials[:, :, 0]  # (5, 2)
    # Similarly for dynamic friction:
    all_dynamic_friction = materials[:, :, 1]  # (5, 2)
    # print('all static friction:', all_static_friction[0, :])
    # print('all dynamic friction:', all_dynamic_friction[0, :])

    # Average across the shape dimension (dim=1), resulting in shape (5,)
    avg_sf = all_static_friction.mean(dim=1)   # (5,)
    avg_df = all_dynamic_friction.mean(dim=1)  # (5,)

    # For mass, if there is only 1 body, shape is (5, 1).
    # Squeeze out the second dimension so masses is shape (5,)
    # mass = masses.squeeze(dim=1)               # (5,)

    # Now stack them together: shape => (5, 3)
    # out = torch.stack([avg_sf, avg_df, mass], dim=1)
    out = torch.stack([avg_sf, avg_df], dim=1)
    # out = torch.tensor([[0.5, 0.5]], device=device)

    # print('out:', out[0:, :])
    print('obs object_physical_properties:', out[4:, :])
    return out

def rma_inputs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.rma_inputs.to(env.device)  # Shape: [num_envs, embedding_dim]

# def rma_inputs(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # COM x
#     com_x = get_rigid_body_com_x(env)
#     # print('com_x:', com_x)
#     return com_x

# Pre-compute T-block template (do this once at module level)
def get_t_block_template():
    """Get the canonical T-block vertices (scaled and centered)."""
    t_coords = np.array([
        [-1.2, 0.6], [1.2, 0.6], [1.2, 0.1], [0.3, 0.1],
        [0.3, -1.5], [-0.3, -1.5], [-0.3, 0.1], [-1.2, 0.1], [-1.2, 0.6]
    ], dtype=np.float32)
    
    # Center and scale (same as original)
    center_offset = (t_coords.min(axis=0) + t_coords.max(axis=0)) / 2
    t_coords = t_coords - center_offset
    scale_x = 0.2 / (t_coords[:, 0].max() - t_coords[:, 0].min())
    scale_y = 0.2 / (t_coords[:, 1].max() - t_coords[:, 1].min())
    t_coords = t_coords * [scale_x, scale_y]
    
    return torch.tensor(t_coords, dtype=torch.float32)  # (9, 2)

# Pre-compute template once
T_BLOCK_TEMPLATE = get_t_block_template()

def batched_point_in_polygon(points, polygons):
    """Test N points against N polygons (one-to-one).
    
    Args:
        points: (N, 2) tensor of points
        polygons: (N, V, 2) tensor of N polygons with V vertices each
    
    Returns:
        (N,) boolean tensor
    """
    N, V, _ = polygons.shape
    
    # Get edges: v1 -> v2 for each polygon
    v1 = polygons  # (N, V, 2)
    v2 = torch.roll(polygons, -1, dims=1)  # (N, V, 2) - next vertex
    
    # Expand points for broadcasting with edges
    points_exp = points.unsqueeze(1)  # (N, 1, 2)
    
    # Ray casting: horizontal ray to the right
    # Check if edge crosses horizontal line through point
    y_crosses = ((v1[:, :, 1] > points_exp[:, :, 1]) != 
                 (v2[:, :, 1] > points_exp[:, :, 1]))  # (N, V)
    
    # For crossing edges, check if intersection is to the right
    dy = v2[:, :, 1] - v1[:, :, 1]  # (N, V)
    dx = v2[:, :, 0] - v1[:, :, 0]  # (N, V)
    
    # Avoid division by zero
    dy = torch.where(torch.abs(dy) < 1e-10, torch.sign(dy) * 1e-10, dy)
    
    # X-coordinate of intersection
    x_intersect = v1[:, :, 0] + dx * (points_exp[:, :, 1] - v1[:, :, 1]) / dy
    x_right = x_intersect > points_exp[:, :, 0]  # (N, V)
    
    # Count crossings to the right for each environment
    crossings = (y_crosses & x_right).sum(dim=1)  # (N,)
    
    # Odd number of crossings = inside
    return (crossings % 2) == 1

def batched_distance_to_polygon(points, polygons):
    """Compute distance from N points to N polygons (one-to-one).
    
    Args:
        points: (N, 2) tensor of points
        polygons: (N, V, 2) tensor of N polygons
        
    Returns:
        (N,) tensor of distances
    """
    N, V, _ = polygons.shape
    
    # Get edges
    v1 = polygons  # (N, V, 2)
    v2 = torch.roll(polygons, -1, dims=1)  # (N, V, 2)
    
    # Vector from v1 to v2 for each edge
    edge_vec = v2 - v1  # (N, V, 2)
    
    # Vector from v1 to point for each environment  
    points_exp = points.unsqueeze(1)  # (N, 1, 2)
    point_vec = points_exp - v1  # (N, V, 2)
    
    # Project point onto each edge
    edge_len_sq = (edge_vec ** 2).sum(dim=-1, keepdim=True)  # (N, V, 1)
    edge_len_sq = torch.clamp(edge_len_sq, min=1e-10)
    
    dot_product = (point_vec * edge_vec).sum(dim=-1, keepdim=True)  # (N, V, 1)
    t = torch.clamp(dot_product / edge_len_sq, 0, 1)  # (N, V, 1)
    
    # Closest point on each edge
    closest_on_edge = v1 + t * edge_vec  # (N, V, 2)
    
    # Distance from point to closest point on each edge
    distances = torch.norm(points_exp - closest_on_edge, dim=-1)  # (N, V)
    
    # Minimum distance across all edges for each environment
    return distances.min(dim=1)[0]  # (N,)

# def t_block_contact_flag(
#     env,
#     contact_threshold: float = 0.01,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
# ) -> torch.Tensor:
#     """Fully vectorized T-block contact detection with exact accuracy."""
    
#     obj: RigidObject = env.scene[object_cfg.name]
#     ee: RigidObject = env.scene[ee_frame_cfg.name]
#     device = env.device
#     N = env.num_envs
    
#     # Get data (all on GPU)
#     obj_pos = obj.data.root_pos_w[:, :2]  # (N, 2)
#     obj_quat = obj.data.root_state_w[:, 3:7]  # (N, 4)
#     ee_pos = ee.data.target_pos_w[..., 0, :2]  # (N, 2)
    
#     # Compute rotation
#     rot_z = 2 * torch.atan2(obj_quat[:, 2], obj_quat[:, 0])  # (N,)
#     cos_rot = torch.cos(rot_z)  # (N,)
#     sin_rot = torch.sin(rot_z)  # (N,)
    
#     # Transform template to world coordinates for all environments
#     template = T_BLOCK_TEMPLATE.to(device)  # (9, 2)
#     template_exp = template.unsqueeze(0).expand(N, -1, -1)  # (N, 9, 2)
    
#     # Vectorized rotation
#     x, y = template_exp[..., 0], template_exp[..., 1]  # (N, 9), (N, 9)
#     cos_rot_exp = cos_rot.unsqueeze(1)  # (N, 1)
#     sin_rot_exp = sin_rot.unsqueeze(1)  # (N, 1)
    
#     x_rot = cos_rot_exp * x - sin_rot_exp * y  # (N, 9)
#     y_rot = sin_rot_exp * x + cos_rot_exp * y  # (N, 9)
#     rotated = torch.stack([x_rot, y_rot], dim=-1)  # (N, 9, 2)
    
#     # Translate to world position
#     world_polygons = rotated + obj_pos.unsqueeze(1)  # (N, 9, 2)
    
#     # Test if points are inside polygons (fully vectorized)
#     inside = batched_point_in_polygon(ee_pos, world_polygons)  # (N,)
    
#     # For points outside, compute distance to boundary
#     distances = batched_distance_to_polygon(ee_pos, world_polygons)  # (N,)
#     near_boundary = distances < contact_threshold  # (N,)
    
#     # Contact if inside OR near boundary
#     contact_flags = inside | near_boundary  # (N,)

#     # Print first contact flag for debugging
#     print('contact_flags:', contact_flags[0])
#     print('contact_flags shape:', contact_flags.shape)
    
#     return contact_flags.float().unsqueeze(1)  # (N, 1)

# def t_block_contact_flag(
#     env,
#     contact_threshold: float = 0.1,  # Larger threshold for meaningful signal
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
# ) -> torch.Tensor:
#     """Return continuous contact signal instead of binary flag."""
    
#     obj: RigidObject = env.scene[object_cfg.name]
#     ee: RigidObject = env.scene[ee_frame_cfg.name]
#     device = env.device
#     N = env.num_envs
    
#     obj_pos = obj.data.root_pos_w[:, :2]  # (N, 2)
#     obj_quat = obj.data.root_state_w[:, 3:7]  # (N, 4)
#     ee_pos = ee.data.target_pos_w[..., 0, :2]  # (N, 2)
    
#     # Compute rotation
#     rot_z = 2 * torch.atan2(obj_quat[:, 2], obj_quat[:, 0])  # (N,)
#     cos_rot = torch.cos(rot_z)  # (N,)
#     sin_rot = torch.sin(rot_z)  # (N,)
    
#     # Transform template to world coordinates
#     template = T_BLOCK_TEMPLATE.to(device)  # (9, 2)
#     template_exp = template.unsqueeze(0).expand(N, -1, -1)  # (N, 9, 2)
    
#     # Vectorized rotation
#     x, y = template_exp[..., 0], template_exp[..., 1]  # (N, 9), (N, 9)
#     cos_rot_exp = cos_rot.unsqueeze(1)  # (N, 1)
#     sin_rot_exp = sin_rot.unsqueeze(1)  # (N, 1)
    
#     x_rot = cos_rot_exp * x - sin_rot_exp * y  # (N, 9)
#     y_rot = sin_rot_exp * x + cos_rot_exp * y  # (N, 9)
#     rotated = torch.stack([x_rot, y_rot], dim=-1)  # (N, 9, 2)
    
#     # Translate to world position
#     world_polygons = rotated + obj_pos.unsqueeze(1)  # (N, 9, 2)
    
#     # Check if inside (distance = 0 if inside)
#     inside = batched_point_in_polygon(ee_pos, world_polygons)  # (N,)
    
#     # Compute actual distances to boundary
#     distances = batched_distance_to_polygon(ee_pos, world_polygons)  # (N,)
    
#     # Set distance to 0 for points inside the polygon
#     distances = torch.where(inside, torch.zeros_like(distances), distances)
    
#     # Smooth sigmoid
#     scale = 0.02
#     contact_signal = torch.sigmoid((contact_threshold - distances) / scale)
    
#     # print(f'contact_signal range: {contact_signal.min():.3f} - {contact_signal.max():.3f}')
#     # print(f'distances range: {distances.min():.3f} - {distances.max():.3f}')
#     # print("ee_pos_world:", ee_pos[0].cpu())
#     # print("obj_pos_world:", obj_pos[0].cpu())
#     # print("ee - obj dist:", torch.norm(ee_pos - obj_pos, dim=1)[0].item())
#     # print('distances:', distances[0])
#     print('contact_signal:', contact_signal[0])

#     # # Sanity check: make contact signal just 1 for all
#     # contact_signal = torch.ones_like(distances)  # For debugging
#     # print('contact_signal:', contact_signal[0])
    
#     return contact_signal.unsqueeze(1)  # (N, 1)

# def get_rigid_body_com_x(
#     env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Get the center of mass of the given rigid body in the environment frame."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     coms = object.root_physx_view.get_coms().clone()

#     # We only care about x axis for the object
#     coms = coms[:, 0:1]

#     # Flatten the tensor to 1D
#     coms = coms.view(coms.shape[0], -1)
#     # print('coms:', coms)

#     return coms

def get_rigid_body_com_x(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get the center of mass of the given rigid body in the environment frame."""
    object: RigidObject = env.scene[object_cfg.name]
    
    coms = object.root_physx_view.get_coms()[:, 0:1].contiguous()
    return coms