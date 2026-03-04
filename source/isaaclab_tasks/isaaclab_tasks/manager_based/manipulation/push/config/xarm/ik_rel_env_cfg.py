# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab.assets import RigidObjectCfg
from isaaclab_tasks.manager_based.manipulation.push import mdp
# from isaaclab.feature_extractor import FeatureExtractorCfg

from . import joint_pos_env_cfg

from rl_games.algos_torch.encoders_cfg import USE_HAMMER
SCALE = 0.0
if USE_HAMMER:
    SCALE = 0.05
    # SCALE = 0.1   # PHASE 1 TRAINING
else:
    SCALE = 0.03

##
# Pre-defined configs
##
from isaaclab_assets.robots.xarm import XARM_HIGH_PD_CFG  # isort: skip

@configclass
class XArmPushEnvCfg(joint_pos_env_cfg.XArmPushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set XArm as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = XARM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/xarm6_with_gripper")

        self.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            debug_vis=True,
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            body_name="link6",
            # body_name="push_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="position_xy", use_relative_mode=True, ik_method="dls"),
            # scale=0.02,
            # scale=0.06,
            # scale=0.03,  # TBLOCk

            # scale=0.05,  # HAMMER FINE TUNING
            # scale=0.03,
            scale=SCALE,     # HAMMER INITIAL

            # scale=0.08,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                # pos=(0.10, 0.0, 0.0),
                # pos=(0.0, 0.0, 0.195),
                pos=(0.0, 0.0, 0.0),
            ),  # THIS IS TO THE BASE OF THE GRIPPER - make sure that's the case when i transfer to real
        )

@configclass
class XArmPushEnvCfg_PLAY(XArmPushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
