# Xarm 6 lite

"""Configuration for the ViperX 300s robots.

Reference: https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_control/config/vx300s.yaml
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math

##
# Configuration
##

XARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/maggiewang/Workspace/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/xArm/xArm_rotated_ee.usda",
        # usd_path="/home/maggiewang/Workspace/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/xArm/xArm.usda",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.28819,
            "joint3": -0.29546,
            "joint4": 1.3243e-05,
            "joint5": 0.58206,
            "joint6": -3.1416,
        },
    ),
    actuators={
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            # effort_limit=70.0,
            effort_limit=100.0,
            # effort_limit=10.0,
            # effort_limit=0.0,
            # velocity_limit=2.286,  # NOTE this actually doesn't set anything, need to set it in usd itself
            stiffness=25.0,
            damping=2.86,
            # armature=0.1,
        ),
        "joint2": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            # effort_limit=57.0,
            effort_limit=100.0,
            # effort_limit=20.0,
            # effort_limit=0.0,
            # velocity_limit=2.286,
            stiffness=76.0,
            damping=6.25,
            # armature=0.1,
        ),
        "joint3": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            # effort_limit=25.0,
            effort_limit=100.0,
            # effort_limit=15.0,
            # effort_limit=0.0,
            # velocity_limit_sim=2.286,
            stiffness=106.0,
            damping=8.15,
            # armature=0.1,
        ),
        "joint4": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            # effort_limit=10.0,
            effort_limit=100.0,
            # effort_limit=2.0,
            # effort_limit=0.0,
            # velocity_limit_sim=2.286,
            stiffness=35.0,
            damping=3.07,
            # armature=0.1,
        ),
        "joint5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            # effort_limit=35.0,
            effort_limit=100.0,
            # effort_limit=5.0,
            # effort_limit=0.0,
            # velocity_limit_sim=2.286,
            stiffness=8.0,
            damping=1.18,
            # armature=0.1,
        ),
        "joint6": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            # effort_limit=35.0,
            effort_limit=100.0,
            # effort_limit=1.0,
            # effort_limit=0.0,
            # velocity_limit_sim=2.286,
            stiffness=7.0,
            damping=0.78,
            # armature=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

XARM_HIGH_PD_CFG = XARM_CFG.copy()
XARM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True

# VIPER_HIGH_PD_CFG = VIPER_CFG.copy()
# VIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
XARM_HIGH_PD_CFG.actuators["joint1"].stiffness = 900.0
XARM_HIGH_PD_CFG.actuators["joint1"].damping = 80.0
XARM_HIGH_PD_CFG.actuators["joint2"].stiffness = 2000.0
XARM_HIGH_PD_CFG.actuators["joint2"].damping = 300.0
XARM_HIGH_PD_CFG.actuators["joint3"].stiffness = 2000.0
XARM_HIGH_PD_CFG.actuators["joint3"].damping = 300.0
XARM_HIGH_PD_CFG.actuators["joint4"].stiffness = 600.0
XARM_HIGH_PD_CFG.actuators["joint4"].damping = 20.0
XARM_HIGH_PD_CFG.actuators["joint5"].stiffness = 600.0
XARM_HIGH_PD_CFG.actuators["joint5"].damping = 20.0
XARM_HIGH_PD_CFG.actuators["joint6"].stiffness = 400.0
XARM_HIGH_PD_CFG.actuators["joint6"].damping = 20.0
"""Configuration of Viper robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""