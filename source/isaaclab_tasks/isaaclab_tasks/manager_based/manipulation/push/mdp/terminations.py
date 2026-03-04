# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.orbit.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# TODO should be if any part of the robot touches the table
# Robot touches table with end effector, terminate episode
    
# Array of success across environments
# NUM_ENVS = 4096  # TODO Change this to the number of environments
NUM_ENVS = 1
SUCCESS_ACROSS_ENVIRONMENTS = torch.zeros(NUM_ENVS, device='cuda')
FINAL_DISTANCE_FROM_GOAL = torch.ones(NUM_ENVS, device='cuda') * 0.4
FINAL_YAW_ERROR = torch.ones(NUM_ENVS, device='cuda') * 180.0

# Track previous values to only print when they change
PREV_SUCCESS_RATE = -1.0
PREV_MEAN_DISTANCE = -1.0
PREV_MEAN_YAW_ERROR = -1.0

def robot_touched_table(
    env: RLTaskEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_tip_frame"),
    table_height: float = 0.0,
    threshold: float = 0.0001,
) -> torch.Tensor:
    """Detect if the end-effector touched the table and persistently store failure across episodes."""

    # Extract end-effector position
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]  # Shape: (num_envs, 3)

    # Compute whether the EE is too low
    ee_too_low = ee_w[..., 2] < (table_height + threshold)

    # # If EE touches the table, mark failure persistently across episodes
    # # Track sticky failures
    # if "EE_TOUCHED_TABLE" not in env.extras:
    #     env.extras["EE_TOUCHED_TABLE"] = torch.zeros(env.num_envs, device="cuda", dtype=torch.bool)

    # env.extras["EE_TOUCHED_TABLE"] |= ee_too_low  # If it fails once, it remains failed

    # Debugging print
    if torch.any(ee_too_low):
        print(f"EE touched table: {torch.sum(ee_too_low).item()} robots")

    return ee_too_low

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to yaw."""
    # Assuming q is [w, x, y, z]
    w, x, y, z = q.unbind(dim=-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    return torch.atan2(siny_cosp, cosy_cosp)

def object_reached_goal(
    env: RLTaskEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:  # Always returns False now
    """Termination condition that NEVER triggers early (only tracks success at timeout)."""
    global PREV_SUCCESS_RATE, PREV_MEAN_DISTANCE, PREV_MEAN_YAW_ERROR
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
    yaw_error_deg = torch.abs(torch.rad2deg(yaw_diff))
    
    # Velocity check
    velocity = torch.norm(obj.data.root_vel_w[:, :3], dim=1)
    
    # Success thresholds (only used at timeout)
    pos_threshold = 0.03
    yaw_threshold_deg = 10.0
    vel_threshold = 0.01
    
    # Success condition
    succeeded = (
        (pos_distance < pos_threshold) &
        (yaw_error_deg < yaw_threshold_deg) & 
        (velocity < vel_threshold)
    )
    
    # Handle timeout
    timed_out = env.episode_length_buf >= env.max_episode_length - 1
    
    # Update success tracking ONLY at timeout
    SUCCESS_ACROSS_ENVIRONMENTS[timed_out] = succeeded[timed_out].float()
    FINAL_DISTANCE_FROM_GOAL[timed_out] = pos_distance[timed_out]
    FINAL_YAW_ERROR[timed_out] = yaw_error_deg[timed_out]

    # Logging - only print when values change significantly
    success_rate = SUCCESS_ACROSS_ENVIRONMENTS.sum() / NUM_ENVS
    mean_distance = FINAL_DISTANCE_FROM_GOAL.mean().item()
    mean_yaw_error = FINAL_YAW_ERROR.mean().item()
    
    # Only print if values have changed significantly (tolerance for floating point precision)
    tolerance = 1e-4
    if (abs(success_rate.item() - PREV_SUCCESS_RATE) > tolerance or
        abs(mean_distance - PREV_MEAN_DISTANCE) > tolerance or
        abs(mean_yaw_error - PREV_MEAN_YAW_ERROR) > tolerance):
        
        print(f"Success Rate (%): {success_rate.item() * 100:.4f}")
        print(f"Mean final distance: {mean_distance:.4f}")
        print(f"Mean yaw error: {mean_yaw_error:.2f}°")
        
        # Update previous values
        PREV_SUCCESS_RATE = success_rate.item()
        PREV_MEAN_DISTANCE = mean_distance
        PREV_MEAN_YAW_ERROR = mean_yaw_error

    env.extras["log"].update({
        "percentage_success": success_rate,
        "final_xy_error": FINAL_DISTANCE_FROM_GOAL.mean(),
        "final_yaw_error": FINAL_YAW_ERROR.mean(),
    })
    
    return torch.zeros_like(pos_distance, dtype=torch.bool)  # Always False