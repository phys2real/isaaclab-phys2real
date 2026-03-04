# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg

from rl_games.algos_torch.encoders_cfg import USE_HAMMER

# from pxr import Usd, UsdShade, UsdPhysics

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.xarm import XARM_CFG  # isort: skip

# from isaaclab.feature_extractor import FeatureExtractorCfg
# from omni.isaac.lab.rma import HistoryEncoderCfg
# from pxr import PhysxSchema, Sdf, Semantics

##
# OBJECT SELECTION - Toggle this flag to switch between objects
##
# USE_HAMMER = True  # Set to True for hammer, False for tblock

def get_object_config(object_type="tblock"):
    """Get object configuration based on type"""
    configs = {
        "tblock": {
            "usd_path": "/home/maggiewang/Workspace/sim-to-real-rl/assets/diffusion_tblock/big_tblock_rotated_centered_mass200g.usda",
            "scale": (1.0, 1.0, 1.0),
            "init_pos": (0.1704, 0.0, 0.04),
            "init_rot": (1.0, 0.0, 0.0, 0.0),
            "rigid_props": RigidBodyPropertiesCfg(
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # "reward_weights": {
            #     "reaching_object": 10.0,
            #     "object_goal_tracking": 20.0,
            #     "object_goal_tracking_fine_grained": 160.0,
            #     "object_orientation_near_goal": 50.0,
            # },
        },
        "hammer": {
            "usd_path": "/home/maggiewang/Workspace/sim-to-real-rl/assets/hammer/hammer_mass620g.usda",
            # "scale": (0.105, 0.105, 0.105),  # TODO CHECK THIS SCALE
            # "scale": (0.105, 0.105, 0.105),  # TODO CHECK THIS SCALE
            "scale": (0.115, 0.115, 0.115),  # TODO CHECK THIS SCALE
            # "scale": (1.0, 1.0, 1.0),  # TODO CHECK THIS SCALE
            "init_pos": (0.3, 0.2, 0.028109 + 0.03),
            "init_rot": (1.0, 0.0, 0.0, 0.0),
            "rigid_props": RigidBodyPropertiesCfg(
                # solver_position_iteration_count=128,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5000.0,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            # "reward_weights": {
            #     "reaching_object": 10.0,  # Slightly higher for hammer (harder to reach)
            #     "object_goal_tracking": 20.0,  # Higher weight for position tracking
            #     "object_goal_tracking_fine_grained": 180.0,  # Higher fine-grained reward
            #     "object_orientation_near_goal": 75.0,  # Much higher orientation reward (hammer orientation matters more)
            # },
        },
    }
    return configs.get(object_type, configs["tblock"])

@configclass
class XArmPushEnvCfg(PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set XArm as robot
        self.scene.robot = XARM_CFG.replace(prim_path="{ENV_REGEX_NS}/xarm6_with_gripper")

        self.commands.object_pose.body_name = "Object"

        # Get object type from flag
        object_type = "hammer" if USE_HAMMER else "tblock"
        object_config = get_object_config(object_type)
        
        print(f"[Config] Using object type: {object_type}")

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=object_config["init_pos"], 
                rot=object_config["init_rot"]
            ),
            spawn=UsdFileCfg(
                usd_path=object_config["usd_path"],
                scale=object_config["scale"],
                rigid_props=object_config["rigid_props"],
                semantic_tags=[("class", "object")],
            ),
        )
        
        # # Apply object-specific reward weights
        # reward_weights = object_config["reward_weights"]
        # self.rewards.reaching_object.weight = reward_weights["reaching_object"]
        # self.rewards.object_goal_tracking.weight = reward_weights["object_goal_tracking"]
        # self.rewards.object_goal_tracking_fine_grained.weight = reward_weights["object_goal_tracking_fine_grained"]
        # if hasattr(self.rewards, 'object_orientation_near_goal'):
        #     self.rewards.object_orientation_near_goal.weight = reward_weights["object_orientation_near_goal"]

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_tip_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/xarm6_with_gripper/link1",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/xarm6_with_gripper/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=(0.0, 0.0075, 0.195),
                        pos=(0.0, 0.0, 0.195),
                        # rot=(0.0, 0.7071, 0.0, 0.7071)  # Franka has z pointing down while Viper has x pointing down, so rotate the frame
                    ),
                ),
            ],
        )

        self.scene.ee_base_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/xarm6_with_gripper/link1",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/xarm6_with_gripper/link6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        # rot=(0.0, 0.7071, 0.0, 0.7071)  # Franka has z pointing down while Viper has x pointing down, so rotate the frame
                    ),
                ),
            ],
        )

        # self.history_encoder_cfg = HistoryEncoderCfg(
        #     train=True,
        #     # load_checkpoint=False,
        #     # checkpoint_name="history_encoder_50000_0.0123.pth",
        #     log_dir="/home/maggiewang/Workspace/IsaacLab/logs/xarm_push/rma",
        # )

@configclass
class XArmPushEnvCfg_PLAY(XArmPushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # log_dir = os.path.join("logs", "viper_push", "feature_extractor", log_dir)

        # self.feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True, write_image_to_file=False)
