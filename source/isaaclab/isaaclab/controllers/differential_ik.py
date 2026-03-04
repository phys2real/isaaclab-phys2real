# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg

from rl_games.algos_torch.encoders_cfg import USE_HAMMER


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "position_xy":
            # print('COMMAND TYPE POSITIONXY')
            return 2  # (x, y)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position_xy":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                
                my_command = self._command
                # if self.cfg.command_type == "position_xy":
                #     # Append a 0 to the z axis of command
                #     my_command = torch.cat((my_command, torch.zeros(self.num_envs, 1, device=self._device)), dim=1)

                # self.ee_pos_des[:] = ee_pos + my_command
                # self.ee_quat_des[:] = ee_quat

                # self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, my_command)
                # # Set z to 0.020 and set orientation to z down 
                # from omni.isaac.lab.utils.math import quat_from_euler_xyz
                # ee_orientation = quat_from_euler_xyz(
                #                     torch.tensor(0.0),
                #                     torch.tensor(3.1415),
                #                     torch.tensor(0.0),
                #                 )
                # # ee_orientation = quat_from_euler_xyz(
                # #     torch.tensor(-3.1415),
                # #     torch.tensor(0.0),
                # #     torch.tensor(-3.1415),
                # # )
                
                # self.ee_pos_des[:, 2] = 0.010 + 0.195  # To the base of the EE

                # self.ee_quat_des = torch.cat([ee_orientation.unsqueeze(0) for _ in range(self.num_envs)], dim=0)
                # # Reorder the quaternion to be in the form of (w, x, y, z)
                # # self.ee_quat_des = self.ee_quat_des[:, [1, 2, 3, 0]]

                ##########
                # DIRECTLY SET DESIRED POSE WITHOUT apply_delta_pose()
                self.ee_pos_des = ee_pos.clone()
                self.ee_pos_des[:, :2] += command  # Only apply XY delta
                
                # Set fixed Z and downward orientation
                if USE_HAMMER:
                    # FOR HAMMER
                    self.ee_pos_des[:, 2] = 0.015 + 0.195  # Your desired Z height
                else:
                    # FOR TBLOCK
                    self.ee_pos_des[:, 2] = 0.02 + 0.195  # Your desired Z height
                
                self.ee_quat_des = torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0]],  # Fixed downward orientation (x,y,z,w)
                    device=self._device
                ).repeat(self.num_envs, 1)
                ##########

                # # Store initial Z position on first command
                # if not self.first_command_received.all():
                #     first_envs = ~self.first_command_received
                #     self.initial_z[first_envs] = ee_pos[first_envs, 2].clone()
                #     self.first_command_received[first_envs] = True

                # # Apply XY delta while maintaining initial Z position
                # self.ee_pos_des = ee_pos.clone()
                # self.ee_pos_des[:, :2] += command  # Apply XY delta
                # self.ee_pos_des[:, 2] = self.initial_z  # Maintain initial Z

                # # Fixed downward orientation
                # # self.ee_quat_des = torch.tensor(
                # #     [[0.0, 0.0, 1.0, 0.0]],  # Downward orientation (x,y,z,w)
                # #     device=self._device
                # # ).repeat(self.num_envs, 1)

                # quat_des = torch.tensor([[0., 0., 1., 0.]], device=self._device)
                # quat_des = quat_des / quat_des.norm(dim=1, keepdim=True)
                # self.ee_quat_des = quat_des.repeat(self.num_envs, 1)

                # Change device to cuda
                self.ee_quat_des = self.ee_quat_des.to(self._device)
                self.ee_pos_des = self.ee_pos_des.to(self._device)

                # print('--ee_pos:', ee_pos)
                # # # # # # # # # See if z of ee_pos < 0.035
                # # # # # # # # # if ee_pos[0, 2] < 0.035:
                # # # # # # # # #     print('--EE Z IS BELOW 0.035')
                # # # # # print('--ee_quat:', ee_quat)
                # print('--ee_pos_des:', self.ee_pos_des)
                # # # # # # # # # if self.ee_pos_des[0, 2] < 0.035:
                # # # # # # # # #     print('--EE DESIRED Z IS BELOW 0.35')
                # # # # # print('--ee_quat_des:', self.ee_quat_des)
                # # # # # print(f'ee_pos: {ee_pos}, ee_pos_des: {self.ee_pos_des}')
                # # # # # print(f'ee_quat: {ee_quat}, ee_quat_des: {self.ee_quat_des}')
        elif self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                self.ee_pos_des = self._command[:, 0:3]
                self.ee_quat_des = self._command[:, 3:7]

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        # compute the delta in joint-space
        if self.cfg.command_type == "position_xy":  # NOTE lol dumb to use "in" bc this has to go first
            # position_error, axis_angle_error = compute_pose_error(
            #     ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            # )
            # # print('position_error:', position_error)
            # pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )

            # position_error[:, 2] *= 8.0  # Amplifying correction in z
            # position_error[:, :2] *= 1.0  # Amplifying correction in xyz
            # rotation_scale = 5.0         # Ensure exact tracking in orientation
            # axis_angle_error = rotation_scale * axis_angle_error

            pose_error = torch.cat((position_error, axis_angle_error), dim=1)

            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        elif "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_joint_pos
