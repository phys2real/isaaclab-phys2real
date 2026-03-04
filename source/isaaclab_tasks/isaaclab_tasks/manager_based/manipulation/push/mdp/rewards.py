# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the object in the xy plane: (num_envs, 2)
    object_ee_distance_xy = torch.norm(cube_pos_w[..., :2] - ee_w[..., :2], dim=1)
    # Distance of the end-effector to the object in the z direction: (num_envs,)
    object_ee_distance_z = torch.abs(cube_pos_w[..., 2] - ee_w[..., 2])

    # print(f'object z: {cube_pos_w[..., 2]}, ee z: {ee_w[..., 2]}')
    
    # Apply threshold to the xy distance
    threshold_xy = 0.15
    # # i think threshold_xy = 0.15 doesn't work / very bad ??
    object_ee_distance_xy = torch.clamp(object_ee_distance_xy, min=threshold_xy)

    # # Combine xy and z distances
    # object_ee_distance = torch.sqrt(object_ee_distance_xy**2 + object_ee_distance_z**2)

    # NOTE just using the xy distance for the reward since we are now constraining z
    object_ee_distance = object_ee_distance_xy

    # Add object ee distance to log dictionary
    if "log" not in env.extras:
        env.extras["log"] = dict()
    env.extras["log"]["object_ee_distance"] = object_ee_distance
    env.extras["log"]["object_ee_distance_xy"] = object_ee_distance_xy
    env.extras["log"]["object_ee_distance_z"] = object_ee_distance_z

    reward = 1 - torch.tanh(object_ee_distance / std)
    return reward


def object_goal_distance(     
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    # distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # We only care about the xy distance
    distance = torch.norm(des_pos_w[..., :2] - object.data.root_pos_w[:, :2], dim=1)
    # print("object_goal_distance: ", distance)
    reward = 1 - torch.tanh(distance / std)
    return reward

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to yaw."""
    # Assuming q is [w, x, y, z]
    w, x, y, z = q.unbind(dim=-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    return torch.atan2(siny_cosp, cosy_cosp)

def object_goal_orientation_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    activation_start: float = 0.10,  # Start influencing at 15cm
    activation_end: float = 0.05,     # Full influence at 3cm
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for orientation alignment that ramps up as object approaches goal."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get current and desired positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    current_pos = obj.data.root_pos_w[:, :2]
    xy_distance = torch.norm(des_pos_w[:, :2] - current_pos, dim=1)
    
    # Calculate progressive activation weight [0-1]
    distance_from_end = torch.clamp(xy_distance - activation_end, min=0)
    activation_range = activation_start - activation_end
    progress = 1 - (distance_from_end / activation_range)
    orientation_weight = torch.clamp(progress, 0, 1)  # Linear ramp
    
    # Compute yaw error for all environments
    des_quat_b = command[:, 3:7]
    des_quat_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        des_pos_b,
        des_quat_b
    )[1]
    
    current_yaw = quat_to_yaw(obj.data.root_quat_w)
    desired_yaw = quat_to_yaw(des_quat_w)
    yaw_diff = torch.remainder(desired_yaw - current_yaw + np.pi, 2 * np.pi) - np.pi
    
    # Orientation reward component
    orientation_reward = (1 - torch.tanh(torch.abs(yaw_diff) / std)) * orientation_weight
    
    # # Logging
    # env.extras["log"].update({
    #     "goal_yaw_error": torch.rad2deg(torch.abs(yaw_diff)).mean(),
    #     "orientation_weight": orientation_weight,
    #     "effective_orientation_reward": orientation_reward
    # })
    
    return orientation_reward

def object_reached_goal_bonus(
    env: ManagerBasedRLEnv,
    threshold_pos: float,
    threshold_yaw_deg: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success bonus when both position and orientation are within thresholds."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get desired pose in world frame
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3], 
        robot.data.root_state_w[:, 3:7], 
        des_pos_b, 
        des_quat_b
    )
    
    # Position error (XY)
    current_pos = obj.data.root_pos_w[:, :2]
    pos_distance = torch.norm(des_pos_w[:, :2] - current_pos, dim=1)
    
    # Orientation error (yaw)
    current_yaw = quat_to_yaw(obj.data.root_quat_w)
    desired_yaw = quat_to_yaw(des_quat_w)
    yaw_diff = torch.remainder(desired_yaw - current_yaw + np.pi, 2 * np.pi) - np.pi
    yaw_error = torch.abs(yaw_diff)
    
    # Convert degrees to radians for threshold
    yaw_threshold = torch.deg2rad(torch.tensor(threshold_yaw_deg, device=yaw_error.device))
    
    # Success condition (both position and orientation)
    is_success_pos = pos_distance < threshold_pos
    is_success_yaw = yaw_error < yaw_threshold
    is_success = is_success_pos & is_success_yaw
    
    # Only reward the first success per episode (and avoid early false positives)
    is_first_success = is_success & (env.episode_length_buf > 10)

    return is_first_success.float()
