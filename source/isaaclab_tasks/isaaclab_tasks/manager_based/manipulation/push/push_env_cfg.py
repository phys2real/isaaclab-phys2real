# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from . import mdp

from dataclasses import dataclass
import os
import re
import sys

# Import encoder configuration to get phase information
sys.path.append('/home/maggiewang/Workspace/rl_games_isaacsim')
try:
    from rl_games.algos_torch.encoders_cfg import HistoryEncoderCfg
    encoder_cfg = HistoryEncoderCfg()
    current_phase = encoder_cfg.phase
except ImportError:
    print("Warning: Could not import HistoryEncoderCfg, using default episode length")
    current_phase = 1  # Default to phase 1

from rl_games.algos_torch.encoders_cfg import USE_HAMMER

FINAL_YAW = None
if USE_HAMMER:
    FINAL_YAW = (0.785398, 0.785398)

    # FINAL_YAW = (0.0, 0.0)  # FOR SANITY CHECKING
else:
    FINAL_YAW = (0.0, 0.0)

if USE_HAMMER:
    # Set phase-dependent episode length

    # Maybe should make this longer for hammer?
    if current_phase == 2:
        EPISODE_LENGTH = 10.0  # Phase 2: shorter episodes for RMA history encoder training
        print(f"[Config] Phase {current_phase}: Using episode length {EPISODE_LENGTH}s")
    else:
        EPISODE_LENGTH = 10.0  # Phase 1: longer episodes for base policy training  
        print(f"[Config] Phase {current_phase}: Using episode length {EPISODE_LENGTH}s")
else:
    if current_phase == 2:
        # Set phase-dependent episode length
        EPISODE_LENGTH = 50.0  # Phase 2: shorter episodes for RMA history encoder training
        print(f"[Config] Phase {current_phase}: Using episode length {EPISODE_LENGTH}s")
    elif current_phase == 1.0:
        EPISODE_LENGTH = 100.0  # Phase 1: longer episodes for base policy training  
        print(f"[Config] Phase {current_phase}: Using episode length {EPISODE_LENGTH}s")
    else:
        EPISODE_LENGTH = 60.0  # Phase 1.5: medium episodes for combined training
        print(f"[Config] Phase {current_phase}: Using episode length {EPISODE_LENGTH}s")

if USE_HAMMER:
    INITIAL_X_RANGE = (0.0, 0.0)
    INITIAL_Y_RANGE = (0.0, 0.0)
else:
    INITIAL_X_RANGE = (0.1296, 0.1296)
    INITIAL_Y_RANGE = (0.2, 0.2)

# Rewards    
if USE_HAMMER:
    GOAL_TRACKING_FINE_GRAINED_WEIGHT = 90.0
    ORIENTATION_NEAR_GOAL_WEIGHT = 200.0
    REACHED_GOAL_BONUS = 80.0
else:
    GOAL_TRACKING_FINE_GRAINED_WEIGHT = 160.0
    ORIENTATION_NEAR_GOAL_WEIGHT = 100.0
    REACHED_GOAL_BONUS = 60.0

if USE_HAMMER:
    DECIMATION = 10
    SIM_DT = 0.01
    STATIC_FRICTION = 1.0
    DYNAMIC_FRICTION = 1.0

    GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 16 * 4096
    ENABLE_CCD = True
else:
    DECIMATION = 25
    SIM_DT = 0.04
    STATIC_FRICTION = 0.5
    DYNAMIC_FRICTION = 0.5

    GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 4096 * 1024
    ENABLE_CCD = False

##
# Scene definition
##

@dataclass
class CameraConfig:
    focal_length: float
    focus_distance: float
    clipping_range: tuple[float, float]
    horizontal_aperture: float
    vertical_aperture: float
    horizontal_aperture_offset: float
    vertical_aperture_offset: float
    width: int
    height: int

def load_calibration_params(calibration_file, width, height, focal_length, original_width, original_height):
    # Manually parse OpenCV YAML to extract K matrix
    with open(calibration_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    print('lines:', lines)
    # Extract the intrinsic matrix (K)
    K_data = re.search(r'data:\s\[(.*?)]', ''.join(lines), re.DOTALL).group(1)
    K = [float(x.strip()) for x in K_data.split(",")]
    print('K:', K)
    
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    
    # Calculate scaling factors for resolution
    scale_x = width / original_width
    scale_y = height / original_height
    
    # Scale focal lengths and principal points
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    
    # Calculate apertures
    horizontal_aperture = width * focal_length / fx_scaled
    vertical_aperture = height * focal_length / fy_scaled
    
    # Calculate offsets
    horizontal_aperture_offset = (cx_scaled - width / 2) / fx_scaled
    vertical_aperture_offset = (cy_scaled - height / 2) / fy_scaled
    
    return CameraConfig(
        focal_length=focal_length,
        focus_distance=1.0,
        clipping_range=(0.1, 5.0),
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        horizontal_aperture_offset=horizontal_aperture_offset,
        vertical_aperture_offset=vertical_aperture_offset,
        width=width,
        height=height,
    )


# Compute scaled camera configuration lazily/safely to avoid import-time failures
CALIBRATION_FILE = "/home/maggiewang/Workspace/sim-to-real-rl/camera_calibration/calibration.yml"
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080
NEW_WIDTH = 192
NEW_HEIGHT = 108
FOCAL_LENGTH = 0.513  # focal length in cm

def _safe_camera_cfg() -> CameraConfig:
    try:
        if os.path.exists(CALIBRATION_FILE):
            return load_calibration_params(
                CALIBRATION_FILE, NEW_WIDTH, NEW_HEIGHT, FOCAL_LENGTH, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
            )
        # Fallback defaults if calibration file is missing
        return CameraConfig(
            focal_length=FOCAL_LENGTH,
            focus_distance=1.0,
            clipping_range=(0.1, 5.0),
            horizontal_aperture=float(NEW_WIDTH),
            vertical_aperture=float(NEW_HEIGHT),
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            width=NEW_WIDTH,
            height=NEW_HEIGHT,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Camera calibration not loaded: {e}")
        return CameraConfig(
            focal_length=FOCAL_LENGTH,
            focus_distance=1.0,
            clipping_range=(0.1, 5.0),
            horizontal_aperture=float(NEW_WIDTH),
            vertical_aperture=float(NEW_HEIGHT),
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            width=NEW_WIDTH,
            height=NEW_HEIGHT,
        )

camera_cfg = _safe_camera_cfg()

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_tip_frame: FrameTransformerCfg = MISSING
    ee_base_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Camera",
    #     # offset=TiledCameraCfg.OffsetCfg(
    #     #     pos=(0.86, 0.0, 0.47),
    #     #     rot=tuple(quat_from_matrix(create_rotation_matrix_from_view(
    #     #         torch.tensor([[0.86, 0.0, 0.47]]),
    #     #         torch.tensor([[0.463, 0.0, 0.1]])
    #     #     ))[0].tolist()), 
    #     #     convention="opengl"
    #     # ),

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.886, 0.0, 0.381),
    #         rot=(0.626, 0.322, 0.324, 0.632),
    #         convention="opengl"
    #     ),
    #     data_types=["rgb", "depth", "semantic_segmentation"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=camera_cfg.focal_length,
    #         focus_distance=camera_cfg.focus_distance,
    #         horizontal_aperture=camera_cfg.horizontal_aperture,
    #         vertical_aperture=camera_cfg.vertical_aperture,
    #         horizontal_aperture_offset=camera_cfg.horizontal_aperture_offset,
    #         vertical_aperture_offset=camera_cfg.vertical_aperture_offset,
    #         clipping_range=camera_cfg.clipping_range,
    #     ),
    #     width=camera_cfg.width,
    #     height=camera_cfg.height,
    #     colorize_semantic_segmentation=False,
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="object",
        # asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(EPISODE_LENGTH, EPISODE_LENGTH),  # Automatically matches episode length based on phase
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(0.2, 0.6), pos_y=(-0.3, 0.3), pos_z=(0.05, 0.05), 
            
            # this is the original setting change it back when test
            # pos_x=(0.5, 0.5), pos_y=(-0.2, -0.2), pos_z=(0.05, 0.05), 
            pos_x=(0.3, 0.3), pos_y=(-0.2, -0.2), pos_z=(0.04, 0.04), 
            # pos_x=(0.25, 0.35), pos_y=(-0.2, 0.2), pos_z=(0.04, 0.04), 

            # RANDOM
            # pos_x=(0.2, 0.4), pos_y=(-0.2, 0.2), pos_z=(0.005, 0.005), 

            # pos_x=(0.4, 0.6), pos_y=(-0.3, 0.3), pos_z=(0.05, 0.05), 
            # roll=(0.0, 0.0), pitch=(-1.5708, -1.5708), yaw=(0.0, 0.0)
            # pos_x=(0.2, 0.6), pos_y=(-0.3, 0.3), pos_z=(0.05, 0.05), 
            # roll=(0.0, 0.0), pitch=(-1.5708, -1.5708), yaw=(0.0, 0.0)
            # roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-3.14159, 3.14159)
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=FINAL_YAW
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    # body_joint_pos: mdp.JointPositionActionCfg = MISSING
    body_joint_pos : mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


# Removed unused deep learning imports

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class BaseObservationCfg:
        """Base class for shared observations between policy and critic."""

        # contact_flag = ObsTerm(func=mdp.t_block_contact_flag)

        object_pose = ObsTerm(func=mdp.object_pose_in_robot_frame)
        # object_pose = ObsTerm(func=mdp.object_pose_in_robot_frame_randomized)
        
        # embeddings = ObsTerm(func=mdp.image_embeddings)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # target_object_position = ObsTerm(func=mdp.generated_xyz_position_commands, params={"command_name": "object_pose"})
        target_object_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # contact_flag = ObsTerm(func=mdp.t_block_contact_flag)
        actions = ObsTerm(func=mdp.last_action)
        # ee_pose = ObsTerm(unc=mdp.ee_pose_in_robot_root_frame)
        ee_position = ObsTerm(func=mdp.ee_xy_position_in_robot_root_frame)
        # ee_position1 = ObsTerm(func=mdp.ee_xy_position_in_robot_root_frame)

        # # Observe friction and mass of object
        # object_physical_properties = ObsTerm(func=mdp.object_physical_properties)

        # contact_flag = ObsTerm(func=mdp.t_block_contact_flag)
        # contact_flag = ObsTerm(func=mdp.t_block_contact_flag)

        # rma_encoding = ObsTerm(func=mdp.rma_history_encoding)
        rma_inputs = ObsTerm(func=mdp.rma_inputs)

    @configclass
    class PolicyCfg(ObsGroup, BaseObservationCfg):
        """Observations for policy group."""

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup, BaseObservationCfg):
        """Observations for critic group."""
        
        # Add privileged terms specific to the critic
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_frame)
        object_velocity = ObsTerm(func=mdp.object_velocity)  # TODO is this messing up RMA in any way

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # FOR HAMMER
    if USE_HAMMER:
        robot_joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.7, 1.3),  # default: 3.0
                "damping_distribution_params": (0.75, 1.5),  # default: 0.1
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.0, 0.0), "y": (-0.3, 0.3), "z": (0.0, 0.0),},
            # "pose_range": {"x": (-0.2, -0.2), "y": (-0.3, 0.3), "z": (0.0, 0.0),},
            # "pose_range": {"x": (-0.2, -0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0),},
            # "pose_range": {"x": (-0.15, -0.25), "y": (0.15, 0.25), "z": (0.0, 0.0),
            #                "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
            #                "yaw": (-3.14159, 3.14159)},  # randomize the yaw
            # "pose_range": {"x": (-0.2, -0.2), "y": (0.2, 0.2), "z": (0.0, 0.0),
            #                 "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
            #                 "yaw": (-3.14159, 3.14159)},  # randomize the yaw

            # "pose_range": {"x": (-0.2, -0.2), "y": (0.2, 0.2), "z": (0.0, 0.0),
            # "pose_range": {"x": (0.116, 0.116), "y": (0.2, 0.2), "z": (0.0, 0.0),
            # "pose_range": {"x": (0.116, 0.116), "y": (0.2, 0.2), "z": (0.0, 0.0),  
            # "pose_range": {"x": (0.1296 - 0.025, 0.1296 + 0.025), "y": (0.2 - 0.025, 0.2 + 0.025), "z": (0.0, 0.0),
            
            "pose_range": {"x": INITIAL_X_RANGE, "y": INITIAL_Y_RANGE, "z": (0.0, 0.0),

            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            # "pose_range": {"x": (0.0796, 0.1796), "y": (0.1, 0.2), "z": (0.0, 0.0),
            # RANDOM
            # "pose_range": {"x": (-0.3, -0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0),
                
                # "roll": (-0.0120, -0.0120), "pitch": (-0.0127, -0.0127),
                # "yaw": (-3.14159, 3.14159)},  # randomize the yaw  
                # "yaw": (0.0346, 0.0346)},
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159)},
                # "yaw": (-2*3.14/4.0, -2*3.14/4.0)},
                # "yaw": (0.0, 0.0)},  # randomize the yaw

                # "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
                # "yaw": (0.0, 0.0)},  # randomize the yaw  


                # "yaw": (0.0, 0.0)},              
                            # "yaw": (0.0, 0.0)},  # randomize the yaw
            # # "pose_range": {"x": (-0.2, -0.2), "y": (-0.3, 0.3), "z": (0.1, 0.1), 
            #     "roll": (0.0, 0.0),  # 90 degrees in radians to lay flat on the XY plane
            #     "pitch": (-1.5708, -1.5708),      # No pitch
            #     "yaw": (0.0, 0.0)         # No yaw
            # },
            # "pose_range": {"x": (-0.1, -0.1), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    if not USE_HAMMER:
        # FOR TBLOCK
        # -- object
        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                # "static_friction_range": (0.3, 1.5),
                # "dynamic_friction_range": (0.3, 1.5),
                # "static_friction_range": (0.3, 1.5),
                # "dynamic_friction_range": (0.3, 1.5),
                "static_friction_range": (0.3, 0.7),
                "dynamic_friction_range": (0.3, 0.7),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 250,
                # "num_buckets": 4,
            },
        )
    else:
        # FOR HAMMER
        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                # "static_friction_range": (0.3, 1.5),
                # "dynamic_friction_range": (0.3, 1.5),
                # "static_friction_range": (0.3, 1.5),
                # "dynamic_friction_range": (0.3, 1.5),
                "static_friction_range": (0.5, 1.2),
                "dynamic_friction_range": (0.5, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 250,
                # "num_buckets": 4,
            },
        )

    if USE_HAMMER:
        # FOR HAMMER
        object_scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                # "mass_distribution_params": (0.3, 1.0),
                "mass_distribution_params": (0.8, 1.2),
                # "mass_distribution_params": (0.3, 0.3),       # Should be 0.3 but looks too light in sim
                # "mass_distribution_params": (0.6, 0.6),
                # "mass_distribution_params": (1.0, 1.0),
                # "mass_distribution_params": (15.0, 15.0),
                "operation": "scale",
            },
        )
    
    # object_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="reset", 
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #         # "com_range": {"x": (-0.11, 0.11), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    #         # "com_range": {"x": (-0.105, 0.08), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
    #         # "com_range": {"x": (-0.09, 0.09), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
    #         "com_range": {"x": (-0.07 - 0.0369, 0.07 - 0.0369), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    #     },
    # )


    # object_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com_tblock,
    #     mode="reset", 
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #         # "com_range": {"x": (-0.11, 0.11), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    #         # "com_range": {"x": (-0.105, 0.08), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
    #         # "com_range": {"x": (-0.09, 0.09), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
    #         # "com_range": {"x": (-0.07 - 0.0369, 0.07 - 0.0369), "y": (0.0, 0.0), "z": (0.0, 0.0)},

            
    #     },
    # )

    if not USE_HAMMER:
        ## FOR T BLOCK
        object_com = EventTerm(
            func=mdp.randomize_rigid_body_com_with_taped_weight,   # or mdp.randomize_rigid_body_com
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),

                # --- taped weight (point mass) ---
                "taped_weight_mass_range": (0.14316, 0.14316),   # kg
                # "taped_weight_mass_range": (0.0, 0.0),

                # Offset r from the ORIGINAL COM, in BODY frame [m]
                # "weight_offset_range": {"x": (0.1 - 0.0369, 0.1 - 0.0369), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                # "taped_weight_position_range": {"x": (0.09, 0.09), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "taped_weight_position_range": {"x": (-0.13 - 0.0343, 0.13 - 0.0343), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                # "taped_weight_position_range": {"x": (-0.065 - 0.0343, -0.065 - 0.0343), "y": (0.0, 0.0), "z": (0.0, 0.0)},

                # --- optional base mass scaling (density scaling) ---
                # "mass_scale_range": (0.9, 1.1),
                "base_mass_scale_range": (1.0, 1.0),

                # --- optional small COM jitter AFTER combining ---
                "com_perturbation_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        )
    else:
        ### FOR HAMMER
        object_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            # mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                # "com_range": {"x": (-0.11, 0.11), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                # "com_range": {"x": (-0.105, 0.08), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
                # "com_range": {"x": (-0.1388, 0.0812), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
                
                # TODO it's actually a bit larger than 0.025 need to recal with new scale
                # "com_range": {"x": (0.1 - 0.0250, 0.1 - 0.0250), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
                
                # For 0.0105 scale, use -0.0250
                # For 0.0115 scale, use -0.0288
                # "com_range": {"x": (-0.11 - 0.0250, 0.11 - 0.0250), "y": (0.0, 0.0), "z": (-0.01, -0.01)},

                # Better to expand com range on both sides to avoid bias
                "com_range": {"x": (-0.13 - 0.0288, 0.13 - 0.0288), "y": (0.0, 0.0), "z": (-0.01, -0.01)},

                # "com_range": {"x": (-0.12 - 0.0250, 0.12 - 0.0250), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
                # "com_range": {"x": (-0.13 - 0.0250, 0.13 - 0.0250), "y": (0.0, 0.0), "z": (-0.01, -0.01)},
            },
        )
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=10.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.4, "command_name": "object_pose"},
        weight=20.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.03, "command_name": "object_pose"},
        weight=GOAL_TRACKING_FINE_GRAINED_WEIGHT,
    )

    object_orientation_near_goal = RewTerm(
        func=mdp.object_goal_orientation_alignment,
        params={
            "std": 0.3, 
            "command_name": "object_pose", 
            "activation_start": 0.4, 
            "activation_end": 0.1
        },
        # FOR T BLOCK
        # weight=100.0,

        # FOR HAMMER
        weight=ORIENTATION_NEAR_GOAL_WEIGHT
    )

    # object_goal_progress = RewTerm(
    #     func=mdp.object_goal_progress_reward,
    #     params={"goal_threshold": 0.03, "command_name": "object_pose"},
    #     weight=100000.0,
    # )

    success_bonus = RewTerm(
        func=mdp.object_reached_goal_bonus,
        params={
            "threshold_pos": 0.03,        # 3 cm
            "threshold_yaw_deg": 10.0,    # 10 degrees
            "command_name": "object_pose"
        },
        # Tblock
        # weight=60.0

        # Hammer
        weight=REACHED_GOAL_BONUS
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # TODO terminate when object is at target (object_reached_goal)
    object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # # TODO terminate if robot touches table
    # robot_touched_table = DoneTerm(func=mdp.robot_touched_table)
    # TODO terminate if robot touches object (aggressively?)

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.0, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # # TODO tune these values
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 20000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 20000}
    )
    # pass


##
# Environment configuration
##


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pushing environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # sim: SimulationCfg = SimulationCfg(
    #     physics_material=RigidBodyMaterialCfg(
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     physx=PhysxCfg(
    #         bounce_threshold_velocity=0.2,
    #         gpu_max_rigid_contact_count=2**20,
    #         gpu_max_rigid_patch_count=2**23,
    #     ),
    # )
    
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            # bounce_threshold_velocity=0.2,
            bounce_threshold_velocity=0.1,
        ),
    )

    # # Simulation settings
    # sim: SimulationCfg = SimulationCfg(
    #     physics_material=RigidBodyMaterialCfg(
    #         static_friction=0.8,
    #         dynamic_friction=0.8,
    #         friction_combine_mode="multiply",
    #     ),
    #     physx=PhysxCfg(
    #         bounce_threshold_velocity=0.2,
    #         gpu_max_rigid_contact_count=2**20,
    #         gpu_max_rigid_patch_count=2**23,
    #     ),
    # )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # self.decimation = 25  # T BLOCK TRAINING

        self.decimation = DECIMATION  # initial hammer training

        # self.decimation = 35
        # self.decimation = 100
        # self.decimation = 50
        # self.decimation = 30
        # self.decimation = 20
        self.episode_length_s = EPISODE_LENGTH  # Automatically set based on phase: Phase 2=50s, Phase 1=80s

        # self.commands.object_pose.resampling_time_range = (
        #     self.episode_length_s, self.episode_length_s
        # )

        # self.episode_length_s = 8.0
        # self.episode_length_s = 20.0
        # self.episode_length_s = 20.0
        # self.episode_length_s = 40.0  TODO next
        # self.episode_length_s = 6.0

        # simulation settings
        # self.sim.dt = 0.04  # 100Hz    ## TBLOCK TRAINING

        self.sim.dt = SIM_DT  ## HAMMER TRAINING

        # self.sim.dt = 0.02

        # self.sim.physics_material.static_friction = 1.5
        # self.sim.physics_material.dynamic_friction = 1.5
        
        # FOR TBLOCK
        # self.sim.physics_material.static_friction = 0.5
        # self.sim.physics_material.dynamic_friction = 0.5
        
        # FOR HAMMER
        self.sim.physics_material.static_friction = STATIC_FRICTION
        self.sim.physics_material.dynamic_friction = DYNAMIC_FRICTION
        self.sim.physics_material.friction_combine_mode = "multiply"
        # self.sim.physics_material.friction_combine_mode = "average"
        self.sim.physics_material.restitution_combine_mode = "min"

        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.1
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 4096 * 1024
        
        self.sim.physx.gpu_total_aggregate_pairs_capacity = GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY

        self.sim.physx.friction_correlation_distance = 0.00625
        # self.sim.physx.friction_correlation_distance = 0.003
        # self.sim.physx.friction_correlation_distance = 0.02 

        # FOR HAMMER
        self.sim.physx.enable_ccd = ENABLE_CCD

        # self.scene.apply_semantics()