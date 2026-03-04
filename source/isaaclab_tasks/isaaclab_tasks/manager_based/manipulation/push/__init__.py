# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Point env cfgs and agent cfgs to the xarm configs that actually exist
from .config.xarm import joint_pos_env_cfg, ik_rel_env_cfg, agents

from rl_games.algos_torch.encoders_cfg import USE_HAMMER

##
# Register Gym environments.
##


# gym.register(
#     id="Isaac-Push-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.push_env_cfg:PushEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

yaml_file = None
if USE_HAMMER:
    yaml_file = "rl_games_ppo_cfg_hammer.yaml"
else:
    yaml_file = "rl_games_ppo_cfg_tblock.yaml"

gym.register(
    id="Isaac-Push-XArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.XArmPushEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:{yaml_file}",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-XArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.XArmPushEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:{yaml_file}",
    },
    disable_env_checker=True,
)

# Vision-specific configs are not present in this package; skip registering those variants

# gym.register(... Vision-Play ...)  # removed due to missing configs

##
# Inverse Kinematics - Absolute Pose Control
##

# gym.register(
#     id="Isaac-Push-Franka-IK-Abs-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": ik_abs_env_cfg.FrankaPushEnvCfg,
#     },
#     disable_env_checker=True,
# )


##
# Inverse Kinematics - Relative Pose Control
##

# gym.register(... Vision-IK-Rel ...)  # removed due to missing configs

# gym.register(... Vision-IK-Rel-Play ...)  # removed due to missing configs

gym.register(
    id="Isaac-Push-XArm-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.XArmPushEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:{yaml_file}",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-XArm-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.XArmPushEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:{yaml_file}",
    },
    disable_env_checker=True,
)
